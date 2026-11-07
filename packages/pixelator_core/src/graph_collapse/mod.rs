mod stats;

use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;
use std::iter;

use crate::common::graph::Graph;
use crate::common::node_indexing::UmiToNodeIndexMapping;
use crate::common::types::{EdgeWeight, NodeIdx, UMI};

use crate::common::distances::{hamming_distance_2bit, within_hamming_distance};
pub use stats::CollapseStats;

/// Finds low-degree nodes that look like sequencing errors of a nearby UMI,
/// and the UMI each should be corrected to.
///
/// A sequencing error in a UMI creates a new node with few edges. A node is
/// investigated when its degree is at most `max_degree`. It is treated as an
/// error when some node at a positive even distance of at most `bfs_threshold`
/// hops has a UMI within `hamming_threshold` symbols of its own. Even distance
/// keeps the candidate on the investigated node's side of the bipartite graph.
/// `max_degree` of 1 investigates only leaves.
///
/// The first element of the result maps each collapsed UMI to its corrected
/// UMI. It is flat: a chain `A -> B -> H` is stored as `A -> H` and `B -> H`,
/// and a match that would close a loop is dropped.
///
/// The second element counts what the search did, for this graph only:
/// - `nodes_considered` is the number of nodes with degree at most `max_degree`.
/// - `targets_found` and both distance distributions describe the matches that
///   were kept, one entry per kept match. A match dropped because it would
///   close a loop is not counted.
/// - Each distance is that of the match as found, so `A -> B -> H` records the
///   distances of `A -> B` and `B -> H`, not of the flattened `A -> H`.
///
/// `total_components` is left at 0. The graph is not known to be one
/// component, so the caller counts components and adds them with
/// [`CollapseStats::add_components`].
pub fn graph_collapse<T>(
    graph: &Graph<T>,
    umi_mapping: &UmiToNodeIndexMapping,
    bfs_threshold: usize,
    hamming_threshold: u8,
    max_degree: usize,
) -> (HashMap<UMI, UMI>, CollapseStats)
where
    T: EdgeWeight + Sync,
{
    let mut stats = CollapseStats::default();

    let mut matches: Vec<(UMI, UMI, (usize, usize))> = graph
        .get_degrees()
        .par_iter()
        .enumerate()
        .filter_map(|(node_id, &degree)| (degree <= max_degree).then_some(node_id))
        // One queue and visited set per Rayon job, reused across its nodes,
        // instead of a fresh pair for every search.
        .map_init(
            || (VecDeque::new(), StampSet::new(graph.get_degrees().len())),
            |scratch, node| {
                find_collapse_target_with(
                    graph,
                    umi_mapping,
                    node,
                    bfs_threshold,
                    hamming_threshold,
                    scratch,
                )
                .map(|(umi, depth, hamming_dist)| {
                    (
                        umi_mapping.map_node_index_to_umi(node),
                        umi,
                        (depth, hamming_dist),
                    )
                })
            },
        )
        .flatten()
        .collect();
    // Sort matches to make the output deterministic
    matches.sort_unstable_by_key(|(original, _, _)| *original);

    stats.add_considered_nodes(
        graph
            .get_degrees()
            .par_iter()
            .filter(|&&degree| degree <= max_degree)
            .count(),
    );

    let mut targets: HashMap<UMI, UMI> = HashMap::default();
    for (original, target, (bfs_dist, hamming_dist)) in matches {
        // Skip the match if adding `original -> target` would create a loop.
        if end_of_chain(&mut targets, target) != original {
            targets.insert(original, target);
            stats.add_target_stats(bfs_dist, hamming_dist);
        }
    }

    let nodes: Vec<UMI> = targets.keys().copied().collect();
    for node in nodes {
        end_of_chain(&mut targets, node);
    }
    (targets, stats)
}

/// Follows arrows from `umi` to the node that has none, and points every node
/// visited along the way at that end.
fn end_of_chain(targets: &mut HashMap<UMI, UMI>, start: UMI) -> UMI {
    let mut path = Vec::new();
    let mut umi = start;
    while let Some(&next) = targets.get(&umi) {
        path.push(umi);
        umi = next;
    }
    for node in path {
        targets.insert(node, umi);
    }
    umi
}

/// Finds the nearest UMI that `node` could be a misread of, or `None` when the
/// bounded search finds none.
///
/// On a match, returns `(umi, hops, hamming)`: the UMI, the number of graph
/// hops from `node` to it, and the Hamming distance between the two UMIs.
pub fn find_collapse_target<T>(
    graph: &Graph<T>,
    umi_mapping: &UmiToNodeIndexMapping,
    node: NodeIdx,
    bfs_threshold: usize,
    hamming_threshold: u8,
    scratch: &mut (VecDeque<(NodeIdx, usize)>, HashSet<NodeIdx>),
) -> Option<(UMI, usize, usize)>
where
    T: EdgeWeight,
{
    find_collapse_target_with(graph, umi_mapping, node, bfs_threshold, hamming_threshold, scratch)
}

/// [`find_collapse_target`] with any [`VisitedSet`] (same result).
pub fn find_collapse_target_with<T, S>(
    graph: &Graph<T>,
    umi_mapping: &UmiToNodeIndexMapping,
    node: NodeIdx,
    bfs_threshold: usize,
    hamming_threshold: u8,
    scratch: &mut (VecDeque<(NodeIdx, usize)>, S),
) -> Option<(UMI, usize, usize)>
where
    T: EdgeWeight,
    S: VisitedSet,
{
    let node_umi = umi_mapping.map_node_index_to_umi(node);
    bounded_even_bfs(graph, node, Some(bfs_threshold), scratch)
        .map(|(node_id, depth)| (umi_mapping.map_node_index_to_umi(node_id), depth))
        .find(|&(umi, _)| within_hamming_distance(node_umi as u64, umi as u64, hamming_threshold))
        .map(|(umi, depth)| {
            (
                umi,
                depth,
                hamming_distance_2bit(node_umi as u64, umi as u64) as usize,
            )
        })
}

/// Visited set of the bounded search. `FxHashSet<NodeIdx>` works for any graph; [`StampSet`] is an
/// epoch-stamped array over `0..num_nodes` (no hashing, O(1) clear) with identical behaviour.
pub trait VisitedSet {
    fn clear(&mut self);
    fn contains(&self, node: &NodeIdx) -> bool;
    fn insert(&mut self, node: NodeIdx);
}

impl VisitedSet for HashSet<NodeIdx> {
    fn clear(&mut self) {
        HashSet::clear(self)
    }
    fn contains(&self, node: &NodeIdx) -> bool {
        HashSet::contains(self, node)
    }
    fn insert(&mut self, node: NodeIdx) {
        HashSet::insert(self, node);
    }
}

/// Epoch-stamped visited set over node indices `0..n`.
pub struct StampSet {
    stamp: Vec<u32>,
    epoch: u32,
}

impl StampSet {
    pub fn new(n: usize) -> Self {
        Self { stamp: vec![0; n], epoch: 1 }
    }
}

impl VisitedSet for StampSet {
    fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
    }
    #[inline]
    fn contains(&self, node: &NodeIdx) -> bool {
        self.stamp[*node as usize] == self.epoch
    }
    #[inline]
    fn insert(&mut self, node: NodeIdx) {
        self.stamp[node as usize] = self.epoch;
    }
}

pub trait GraphCollapseExt<T: EdgeWeight> {
    /// Yields each node at a positive even distance from `start_node` as
    /// `(node, hops)`, nearest first.
    ///
    /// `scratch` holds the queue and visited set, so callers can reuse their
    /// allocations across searches. It is cleared on entry.
    fn bounded_even_breadth_first_search_iter<'a>(
        &'a self,
        start_node: NodeIdx,
        bound: Option<usize>,
        scratch: &'a mut (VecDeque<(NodeIdx, usize)>, HashSet<NodeIdx>),
    ) -> impl Iterator<Item = (NodeIdx, usize)> + 'a;
}

impl<T: EdgeWeight> GraphCollapseExt<T> for Graph<T> {
    fn bounded_even_breadth_first_search_iter<'a>(
        &'a self,
        start_node: NodeIdx,
        bound: Option<usize>,
        scratch: &'a mut (VecDeque<(NodeIdx, usize)>, HashSet<NodeIdx>),
    ) -> impl Iterator<Item = (NodeIdx, usize)> + 'a {
        bounded_even_bfs(self, start_node, bound, scratch)
    }
}

/// Generic body of [`GraphCollapseExt::bounded_even_breadth_first_search_iter`].
fn bounded_even_bfs<'a, T: EdgeWeight, S: VisitedSet>(
    graph: &'a Graph<T>,
    start_node: NodeIdx,
    bound: Option<usize>,
    scratch: &'a mut (VecDeque<(NodeIdx, usize)>, S),
) -> impl Iterator<Item = (NodeIdx, usize)> + 'a {
    {
        let (node_queue, visited) = scratch;
        node_queue.clear();
        visited.clear();
        node_queue.push_back((start_node, 0));

        iter::from_fn(move || {
            while let Some((node, d)) = node_queue.pop_front() {
                if visited.contains(&node) {
                    continue;
                }
                visited.insert(node);

                if bound.is_none_or(|bound| d < bound) {
                    for neighbor in graph.neighbors_iter(node) {
                        if visited.contains(&neighbor) {
                            continue;
                        }
                        node_queue.push_back((neighbor, d + 1));
                    }
                }

                if d > 0 && d % 2 == 0 {
                    return Some((node, d));
                }
            }
            None
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::common::node_indexing::UmiToNodeIndexMapping;
    use crate::common::types::edges_from_tuples;

    use super::*;
    use crate::encoding::encode_dna_2bit;

    fn graph_from(edges: &[(NodeIdx, NodeIdx)], num_nodes: usize) -> Graph<u8> {
        Graph::<u8>::from_edges(edges_from_tuples(edges.iter().copied()), num_nodes)
    }

    fn umi(seq: &[u8]) -> UMI {
        encode_dna_2bit(seq) as UMI
    }

    fn mapped_graph(umi_edges: &[(UMI, UMI)]) -> (Graph<u8>, UmiToNodeIndexMapping) {
        let mapping = UmiToNodeIndexMapping::from_umi_pairs(umi_edges.iter().copied());
        let graph = Graph::<u8>::from_edges(
            mapping.map_umi_pair_iterator_to_edge(umi_edges.iter().copied()),
            mapping.get_num_of_nodes(),
        );
        (graph, mapping)
    }

    #[test]
    fn returns_positive_even_distances_in_bfs_order() {
        // 0-1-2-3-4-5    6 (other component)
        let graph = graph_from(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 7);

        let found: Vec<_> = graph
            .bounded_even_breadth_first_search_iter(0, None, &mut Default::default())
            .collect();

        // 2 and 4 are two and four hops away. The start, the odd-distance nodes,
        // and the other component stay out of the result.
        assert_eq!(found, vec![(2, 2), (4, 4)]);
    }

    #[test]
    fn uses_shortest_path_when_a_node_also_has_a_longer_walk() {
        // 3 is one hop from 0, and also three hops along 0-1-2-3.
        // 0-1-2-3
        // |     |
        // +-----+
        let graph = graph_from(&[(0, 1), (1, 2), (2, 3), (0, 3)], 4);

        let found: Vec<_> = graph
            .bounded_even_breadth_first_search_iter(0, None, &mut Default::default())
            .collect();

        assert_eq!(found, vec![(2, 2)]);
    }

    #[test]
    fn bound_keeps_nodes_at_that_depth_and_stops() {
        // 0-1-2-3-4-5
        let graph = graph_from(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 6);

        assert_eq!(
            graph
                .bounded_even_breadth_first_search_iter(0, Some(0), &mut Default::default())
                .collect::<Vec<_>>(),
            Vec::<(NodeIdx, usize)>::new()
        );

        assert_eq!(
            graph
                .bounded_even_breadth_first_search_iter(0, Some(2), &mut Default::default())
                .collect::<Vec<_>>(),
            vec![(2, 2)]
        );

        assert_eq!(
            graph
                .bounded_even_breadth_first_search_iter(0, Some(4), &mut Default::default())
                .collect::<Vec<_>>(),
            vec![(2, 2), (4, 4)]
        );
    }

    #[test]
    fn reused_scratch_does_not_leak_into_the_next_search() {
        // 0-1-2-3-4-5
        let graph = graph_from(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 6);
        let mut scratch = Default::default();

        // Stop after the first node, leaving 0, 1 and 2 visited and 3 queued.
        let first = graph
            .bounded_even_breadth_first_search_iter(0, None, &mut scratch)
            .next();
        assert_eq!(first, Some((2, 2)));

        let found: Vec<_> = graph
            .bounded_even_breadth_first_search_iter(5, None, &mut scratch)
            .collect();
        assert_eq!(found, vec![(3, 2), (1, 4)]);
    }

    /// Following `original -> target` must reach a node that is not itself collapsed.
    fn collapse_pairs_have_a_cycle(targets: &HashMap<UMI, UMI>) -> bool {
        for &start in targets.keys() {
            let mut seen = HashSet::default();
            let mut node = start;
            while let Some(&target) = targets.get(&node) {
                if !seen.insert(node) {
                    return true;
                }
                node = target;
            }
        }
        false
    }

    #[test]
    fn star_leaves_do_not_collapse_into_a_cycle() {
        //        1
        //        |
        //    2---0---3
        let umi_edges = [(0, 1), (0, 2), (0, 3)];
        let mapping = UmiToNodeIndexMapping::from_umi_pairs(umi_edges);
        let graph = graph_from(&umi_edges, mapping.get_num_of_nodes());

        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 1, 1);

        assert!(
            !collapse_pairs_have_a_cycle(&targets),
            "leaves collapsed into a loop: {targets:?}"
        );
        // Every leaf finds a match one symbol and two hops away, but the three
        // matches form a loop. Dropping the one that closes it leaves two, and
        // only those are counted.
        assert_eq!(stats.nodes_considered, 3);
        assert_eq!(stats.targets_found, 2);
        assert_eq!(stats.bfs_distance_counts(), [(2, 2)]);
        assert_eq!(stats.hamming_distance_counts(), [(1, 2)]);
    }

    #[test]
    fn chain_collapses_onto_its_final_node() {
        // A -- M -- H -- N
        //      |
        //      B
        //
        // A is one symbol from B and two from H, so with threshold 1 it matches B
        // and skips H. B is one symbol from H and matches H. The chain A -> B -> H
        // must come out already flattened, with both leaves naming H.
        let umi = |seq: &[u8]| encode_dna_2bit(seq) as UMI;
        let umi_m = umi(b"GGGG");
        let umi_h = umi(b"CCAA");
        let umi_a = umi(b"AAAA");
        let umi_b = umi(b"CAAA");
        let umi_n = umi(b"TTTT");
        let umi_edges = [
            (umi_m, umi_h),
            (umi_m, umi_a),
            (umi_m, umi_b),
            (umi_h, umi_n),
        ];
        let mapping = UmiToNodeIndexMapping::from_umi_pairs(umi_edges);
        let graph = Graph::<u8>::from_edges(
            mapping.map_umi_pair_iterator_to_edge(umi_edges),
            mapping.get_num_of_nodes(),
        );

        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 1, 1);

        assert_eq!(
            targets,
            HashMap::from_iter([(umi_a, umi_h), (umi_b, umi_h)])
        );
        // The leaves are A, B and N. N has no match. The recorded matches are
        // `A -> B` and `B -> H`, both two hops and one symbol apart. The
        // flattened `A -> H` is two symbols apart and is not what is counted.
        assert_eq!(stats.nodes_considered, 3);
        assert_eq!(stats.targets_found, 2);
        assert_eq!(stats.bfs_distance_counts(), [(2, 2)]);
        assert_eq!(stats.hamming_distance_counts(), [(1, 2)]);
    }

    #[test]
    fn chain_of_three_leaves_collapses_onto_the_final_node() {
        //      A
        //      |
        // B -- M -- C
        //      |
        //      H -- N
        //
        // With threshold 1 the raw matches are A -> B -> C -> H: each leaf is one
        // symbol from the next node and two from the one after. H is not a leaf.
        let umi_h = umi(b"AAAA");
        let umi_c = umi(b"CAAA");
        let umi_b = umi(b"CCAA");
        let umi_a = umi(b"CCCA");
        let (graph, mapping) = mapped_graph(&[
            (umi(b"GGGG"), umi_h),
            (umi(b"GGGG"), umi_c),
            (umi(b"GGGG"), umi_a),
            (umi(b"GGGG"), umi_b),
            (umi_h, umi(b"TTTT")),
        ]);

        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 1, 1);

        assert_eq!(
            targets,
            HashMap::from_iter([(umi_a, umi_h), (umi_b, umi_h), (umi_c, umi_h)])
        );
        // The leaves are A, B, C and N. A, B and C each match the next node.
        assert_eq!(stats.nodes_considered, 4);
        assert_eq!(stats.targets_found, 3);
        assert_eq!(stats.bfs_distance_counts(), [(2, 3)]);
        assert_eq!(stats.hamming_distance_counts(), [(1, 3)]);
    }

    #[test]
    fn degree_above_max_degree_is_not_investigated() {
        // D has degree 2 and S has degree 3. They share the neighbor GGGG, so S is
        // two hops from D, and their UMIs are one symbol apart. No other pair of
        // nodes on the same side is within one symbol.
        //
        // With max_degree 1, D is not searched and nothing collapses. With
        // max_degree 2, D is searched and collapses onto S. S has degree 3, so it
        // is never searched itself.
        let umi_d = umi(b"AAAA");
        let umi_s = umi(b"CAAA");
        let (graph, mapping) = mapped_graph(&[
            (umi_d, umi(b"GGGG")),
            (umi_d, umi(b"TTTT")),
            (umi_s, umi(b"GGGG")),
            (umi_s, umi(b"CCCC")),
            (umi_s, umi(b"ATAT")),
        ]);

        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 1, 1);
        assert!(targets.is_empty());
        // Only the three leaves are investigated, and none finds a match.
        assert_eq!(stats.nodes_considered, 3);
        assert_eq!(stats.targets_found, 0);

        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 1, 2);
        assert_eq!(targets, HashMap::from_iter([(umi_d, umi_s)]));
        // D and GGGG join the three leaves. D alone finds a match.
        assert_eq!(stats.nodes_considered, 5);
        assert_eq!(stats.targets_found, 1);
    }

    #[test]
    fn raising_max_degree_can_flip_the_end_of_a_chain() {
        // Same graph as `chain_of_three_leaves_collapses_onto_the_final_node`:
        //
        //      A
        //      |
        // B -- M -- C
        //      |
        //      H -- N
        //
        // UMIs of A, B, C and H are each one symbol from the next one in that order.
        //
        // With max_degree 1, only the leaves are searched. H has degree 2, so it
        // is not, and the chain ends at H.
        //
        // With max_degree 2, H is searched too. Its closest match is C, so the
        // raw matches contain both H -> C and C -> H. Matches are applied in
        // UMI order and H comes first, so H -> C is kept and C -> H is dropped
        // because it would close a loop. The chain now ends at C.
        let umi_h = umi(b"AAAA");
        let umi_c = umi(b"CAAA");
        let umi_b = umi(b"CCAA");
        let umi_a = umi(b"CCCA");
        let (graph, mapping) = mapped_graph(&[
            (umi(b"GGGG"), umi_h),
            (umi(b"GGGG"), umi_c),
            (umi(b"GGGG"), umi_a),
            (umi(b"GGGG"), umi_b),
            (umi_h, umi(b"TTTT")),
        ]);

        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 1, 2);

        assert_eq!(
            targets,
            HashMap::from_iter([(umi_a, umi_c), (umi_b, umi_c), (umi_h, umi_c)])
        );
        assert!(!collapse_pairs_have_a_cycle(&targets));
        // A, B, C, N and H are investigated. The search finds four matches, but
        // `C -> H` closes a loop with `H -> C` and is not counted.
        assert_eq!(stats.nodes_considered, 5);
        assert_eq!(stats.targets_found, 3);
    }

    #[test]
    fn distances_report_the_match_not_the_threshold() {
        // X -- P -- Y -- Q -- Z -- R
        //
        // X and Z are four hops apart and one symbol apart. Y is two hops from X
        // but far from it in sequence, so the search has to go on to Z. R is the
        // other leaf and has no near UMI on its own side.
        let umi_x = umi(b"AAAA");
        let umi_z = umi(b"CAAA");
        let (graph, mapping) = mapped_graph(&[
            (umi_x, umi(b"TTTT")),
            (umi(b"GGGG"), umi(b"TTTT")),
            (umi(b"GGGG"), umi(b"CGCG")),
            (umi_z, umi(b"CGCG")),
            (umi_z, umi(b"GCGC")),
        ]);

        let (targets, stats) = graph_collapse(&graph, &mapping, 4, 1, 1);

        assert_eq!(targets, HashMap::from_iter([(umi_x, umi_z)]));
        assert_eq!(stats.nodes_considered, 2);
        assert_eq!(stats.targets_found, 1);
        // The bound is 4 and the match is found at 4. Nothing is recorded at 2.
        assert_eq!(stats.bfs_distance_counts(), [(4, 1)]);
        assert_eq!(stats.hamming_distance_counts(), [(1, 1)]);

        // With a bound of 2, Z is out of reach and nothing is found.
        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 1, 1);
        assert!(targets.is_empty());
        assert_eq!(stats.nodes_considered, 2);
        assert_eq!(stats.targets_found, 0);
        assert!(stats.bfs_distance_counts().is_empty());
        assert!(stats.hamming_distance_counts().is_empty());
    }

    #[test]
    fn hamming_distance_two_is_recorded_when_the_threshold_allows_it() {
        // M -- H and M -- A, where A and H are two symbols apart and both are
        // leaves. With threshold 2, each matches the other at two hops.
        let umi_m = umi(b"GGGG");
        let umi_h = umi(b"CCAA");
        let umi_a = umi(b"AAAA");
        let (graph, mapping) = mapped_graph(&[(umi_m, umi_h), (umi_m, umi_a)]);

        let (targets, stats) = graph_collapse(&graph, &mapping, 2, 2, 1);

        // A comes first in UMI order, so `A -> H` is kept and `H -> A` closes a
        // loop and is dropped.
        assert_eq!(targets, HashMap::from_iter([(umi_a, umi_h)]));
        assert_eq!(stats.nodes_considered, 2);
        assert_eq!(stats.targets_found, 1);
        assert_eq!(stats.hamming_distance_counts(), [(2, 1)]);
    }

    #[test]
    fn end_of_chain_returns_a_node_that_has_no_arrow() {
        let mut targets = HashMap::default();

        assert_eq!(end_of_chain(&mut targets, 7), 7);
        assert!(targets.is_empty());
    }

    #[test]
    fn end_of_chain_points_every_visited_node_at_the_end() {
        let mut targets = HashMap::from_iter([(1, 2), (2, 3), (3, 4)]);

        assert_eq!(end_of_chain(&mut targets, 1), 4);
        assert_eq!(targets, HashMap::from_iter([(1, 4), (2, 4), (3, 4)]));
        // A later walk stops after one hop.
        assert_eq!(end_of_chain(&mut targets, 2), 4);
    }

    #[test]
    fn find_collapse_target_skips_even_nodes_beyond_the_hamming_threshold() {
        // From the three-leaf chain: A's even nodes are H, then C, then B.
        // Only B is within threshold 1.
        let umi_h = umi(b"AAAA");
        let umi_c = umi(b"CAAA");
        let umi_b = umi(b"CCAA");
        let umi_a = umi(b"CCCA");
        let (graph, mapping) = mapped_graph(&[
            (umi(b"GGGG"), umi_h),
            (umi(b"GGGG"), umi_c),
            (umi(b"GGGG"), umi_a),
            (umi(b"GGGG"), umi_b),
            (umi_h, umi(b"TTTT")),
        ]);
        let node = |seq: &[u8]| mapping.map_umi_to_node_index(umi(seq));

        assert_eq!(
            find_collapse_target(
                &graph,
                &mapping,
                node(b"CCCA"),
                2,
                1,
                &mut Default::default()
            ),
            Some((umi_b, 2, 1))
        );
        assert_eq!(
            find_collapse_target(
                &graph,
                &mapping,
                node(b"CCAA"),
                2,
                1,
                &mut Default::default()
            ),
            Some((umi_c, 2, 1))
        );
        assert_eq!(
            find_collapse_target(
                &graph,
                &mapping,
                node(b"CAAA"),
                2,
                1,
                &mut Default::default()
            ),
            Some((umi_h, 2, 1))
        );
        assert_eq!(
            find_collapse_target(
                &graph,
                &mapping,
                node(b"TTTT"),
                2,
                1,
                &mut Default::default()
            ),
            None
        );
    }

    #[test]
    fn find_collapse_target_returns_none_when_the_bound_stops_before_a_match() {
        let umi_a = umi(b"CCCA");
        let (graph, mapping) = mapped_graph(&[
            (umi(b"GGGG"), umi(b"AAAA")),
            (umi(b"GGGG"), umi(b"CAAA")),
            (umi(b"GGGG"), umi_a),
            (umi(b"GGGG"), umi(b"CCAA")),
            (umi(b"AAAA"), umi(b"TTTT")),
        ]);
        let node = mapping.map_umi_to_node_index(umi_a);

        // The match is two hops away, so bounds 0 and 1 never reach it.
        assert_eq!(
            find_collapse_target(&graph, &mapping, node, 0, 1, &mut Default::default()),
            None
        );
        assert_eq!(
            find_collapse_target(&graph, &mapping, node, 1, 1, &mut Default::default()),
            None
        );
        // Threshold 0 rejects the one-symbol match.
        assert_eq!(
            find_collapse_target(&graph, &mapping, node, 2, 0, &mut Default::default()),
            None
        );
    }
}
