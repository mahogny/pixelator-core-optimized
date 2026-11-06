use rustc_hash::FxHashSet as HashSet;
use sprs::CsMatI;

/// Adjacency storage: u32 column indices, usize row offsets. [pixelator-rs patch: was CsMat<T>
/// (usize indices); halves index memory, results unchanged]
pub type AdjMat<T> = CsMatI<T, u32, usize>;
use std::collections::VecDeque;

use crate::common::types::{Edge, EdgeWeight, NodeIdx};

mod statistics;

pub use statistics::GraphProperties;

/// A simple undirected weighted graph stored as a sparse adjacency matrix.
///
/// The input is interpreted as undirected: whenever you add an edge `(u, v)` with `u != v`,
/// we store it in both adjacency entries `(u, v)` and `(v, u)`. Self-loops `(u, u)` are stored
/// once as `(u, u)`.
///
/// The `Graph` exposes two common "edge size" metrics that differ in meaning:
/// - `edge_entry_count`: counts adjacency entries (non-zero matrix entries in the
///   upper-triangular part of the matrix, i.e. we count each undirected edge once).
/// - `total_edge_weight`: sums the weight values associated with those entries.
///
/// This distinction is important when edges can carry weights: many algorithms (and our
/// quality metrics) need the total *weight*, not just the number of adjacency entries.
///
/// In particular, when we build an aggregated graph during Leiden (`WeightedPartitionedGraph::aggregate`),
/// each "node" represents a set of original nodes (a community at the current level) and each
/// adjacency weight represents the sum of weights of all original edges between the two sets.
/// Therefore, `edge_entry_count` tells you only how many distinct neighbor-community pairs
/// exist, while `total_edge_weight` preserves how strongly communities are connected.
#[derive(Clone)]
pub struct Graph<T> {
    adjacency_matrix: AdjMat<T>,
    num_nodes: usize,
    edge_entry_count: usize,
    total_edge_weight: usize,
    norm: f64,
    degrees: Vec<usize>,
}

impl<T: EdgeWeight> Graph<T> {
    pub fn from_edges<I>(edges: I, num_nodes: NodeIdx) -> Self
    where
        I: Iterator<Item = Edge<T>>,
    {
        let adjacency_matrix = csr_from_undirected_edges(edges, num_nodes);
        Self::from_adjacency_matrix(adjacency_matrix)
    }

    /// Build a graph from a pre-collected undirected edge buffer.
    ///
    /// Each triplet should already be ordered as `(min(src, dest), max(src, dest), weight)`.
    /// Duplicate pairs are merged by summing weights inside CSR construction.
    pub fn from_undirected_triplets(
        triplets: Vec<(NodeIdx, NodeIdx, T)>,
        num_nodes: NodeIdx,
    ) -> Self {
        let adjacency_matrix = csr_from_undirected_triplets(triplets, num_nodes);
        Self::from_adjacency_matrix(adjacency_matrix)
    }

    pub fn from_adjacency_matrix(adjacency_matrix: AdjMat<T>) -> Self {
        let num_nodes = adjacency_matrix.rows();
        let edge_entry_count = adjacency_matrix
            .view()
            .iter_rbr()
            .filter(|(_, (node_1, node_2))| node_1 <= node_2)
            .count();
        let total_edge_weight = adjacency_matrix
            .view()
            .iter_rbr()
            .filter(|(_, (node_1, node_2))| node_1 <= node_2)
            .map(|(&weight, _)| weight.into())
            .sum::<usize>();
        let degrees = adjacency_matrix.degrees();
        let degrees_sum = degrees.iter().sum::<usize>();
        let norm = 1. / (degrees_sum * degrees_sum) as f64;
        Graph {
            adjacency_matrix,
            num_nodes,
            edge_entry_count,
            total_edge_weight,
            norm,
            degrees,
        }
    }

    pub fn get_adjacency_matrix(&self) -> &AdjMat<T> {
        &self.adjacency_matrix
    }

    pub fn get_num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of undirected adjacency entries in the graph.
    ///
    /// Each undirected edge contributes one entry (`src <= dest`), and self loops
    /// also contribute one entry.
    pub fn get_edge_entry_count(&self) -> usize {
        self.edge_entry_count
    }

    /// Returns the sum of all undirected edge weights in the graph.
    ///
    /// The sum is taken over the same set of undirected adjacency entries as
    /// [`Self::get_edge_entry_count`] (upper triangle `src <= dest`, including diagonal/self-loops).
    ///
    /// For unweighted graphs where every edge has weight `1`, this equals
    /// [`Self::get_edge_entry_count`].
    pub fn get_total_edge_weight(&self) -> usize {
        self.total_edge_weight
    }

    pub fn get_edge_weight(&self, src: NodeIdx, dst: NodeIdx) -> Option<T> {
        self.adjacency_matrix.get(src, dst).copied()
    }

    pub fn get_norm(&self) -> f64 {
        self.norm
    }

    pub fn get_degrees(&self) -> &Vec<usize> {
        &self.degrees
    }

    /// Returns a list of nodes connected to `node`. This does not include `node` even when it is
    /// connected to itself.
    pub fn neighbors(&self, node: NodeIdx) -> Vec<NodeIdx> {
        self.get_adjacency_matrix()
            .outer_view(node)
            .unwrap_or_else(|| panic!("Node index out of bounds: {}", node))
            .iter()
            .map(|(neighbor_idx, _)| neighbor_idx)
            .filter(|&neighbor_id| neighbor_id != node)
            .collect()
    }

    /// Returns an iterator over node indices connected to `node` (excluding `node` itself).
    ///
    /// This is the allocation-free counterpart of [`Self::neighbors`], intended for tight loops.
    pub fn neighbors_iter(&self, node: NodeIdx) -> impl Iterator<Item = NodeIdx> + '_ {
        if node >= self.num_nodes {
            panic!("Node index out of bounds: {}", node);
        }

        // We store the graph as CSR, so we can iterate a row without allocating intermediate Vecs.
        // The neighbors for `node` are the column indices in `indices[indptr[node]..indptr[node+1]]`.
        let indptr_base = self.get_adjacency_matrix().indptr();
        let indptr = indptr_base
            .as_slice()
            .expect("CSR indptr should expose a contiguous slice");
        let indices = self.get_adjacency_matrix().indices();
        let start = indptr[node];
        let end = indptr[node + 1];

        indices[start..end]
            .iter()
            .map(|&x| x as usize)
            .filter(move |&neighbor_id| neighbor_id != node)
    }

    /// Returns a list of edge connected to `node`, including self edges looping back
    pub fn get_edges_from(&self, node: NodeIdx) -> Vec<Edge<T>> {
        self.get_adjacency_matrix()
            .outer_view(node)
            .unwrap_or_else(|| panic!("Node index out of bounds: {}", node))
            .iter()
            .map(|(neighbor_id, &weight)| Edge::new(node, neighbor_id, Some(weight)))
            .collect()
    }

    /// Returns an iterator over edges incident to `node` (including self-loops).
    ///
    /// This is the allocation-free counterpart of [`Self::get_edges_from`], intended for tight loops.
    pub fn edges_from_iter(&self, node: NodeIdx) -> impl Iterator<Item = Edge<T>> + '_ {
        if node >= self.num_nodes {
            panic!("Node index out of bounds: {}", node);
        }

        // CSR: iterate row entries by matching `indices` and `data` in lockstep.
        let indptr_base = self.get_adjacency_matrix().indptr();
        let indptr = indptr_base
            .as_slice()
            .expect("CSR indptr should expose a contiguous slice");
        let indices = self.get_adjacency_matrix().indices();
        let data = self.get_adjacency_matrix().data();
        let start = indptr[node];
        let end = indptr[node + 1];

        indices[start..end]
            .iter()
            .map(|&x| x as usize)
            .zip(data[start..end].iter().copied())
            .map(move |(neighbor_id, weight)| Edge::new(node, neighbor_id, Some(weight)))
    }

    /// Iterate over all edges of the graph
    ///
    /// NB: this will iterate over edges in both directions
    pub fn get_edges_iter(&self) -> impl Iterator<Item = Edge<T>> {
        self.adjacency_matrix
            .view()
            .iter_rbr()
            .map(|(&weight, (node_1, node_2))| Edge::new(node_1 as usize, node_2 as usize, Some(weight)))
    }

    /// Get all edges where at least one of the nodes is in the selection
    /// Note that this will include edges that go outside the selection, and edges
    /// that loop back to the same node.
    /// Every egde will only be included once, i.e. (0,1) and (1,0) will not both be included
    /// as (0,1). The edges will be returned in arbitrary order.
    pub fn get_edges_in_selection(
        &self,
        selection: &HashSet<NodeIdx>,
    ) -> impl Iterator<Item = Edge<T>> {
        selection.iter().flat_map(|&node| {
            self.edges_from_iter(node)
                .filter(|Edge { src, dest, .. }| !(src > dest && selection.contains(dest)))
        })
    }

    /// Connected component id of every node, in node-index order.
    ///
    /// Nodes in the same component share an id. Ids start at 0. An isolated node
    /// is its own component.
    pub fn connected_components(&self) -> Vec<usize> {
        self.connected_components_by(|_, _| true)
    }

    /// [`Self::connected_components`] on the edges accepted by `filter`.
    ///
    /// `filter` must be transitive: `filter(a, b)` and `filter(b, c)` imply
    /// `filter(a, c)`. Each component starts at the lowest unvisited node index,
    /// and that start order is the component id order.
    pub fn connected_components_by<F>(&self, filter: F) -> Vec<usize>
    where
        F: Fn(NodeIdx, NodeIdx) -> bool,
    {
        let n = self.get_num_nodes();
        let mut labels = vec![0usize; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        let mut component_id = 0usize;

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let id = component_id;
            component_id += 1;
            visited[start] = true;
            queue.push_back(start);
            while let Some(node) = queue.pop_front() {
                labels[node] = id;
                for neighbor in self.neighbors_iter(node) {
                    if !visited[neighbor] && filter(node, neighbor) {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        labels
    }
}

/// Build a symmetric CSR adjacency matrix from an undirected edge iterator.
///
/// Input edges are stored once as `(min, max, weight)`, merged by summing duplicate
/// pairs (sprs COO semantics), then expanded into both CSR directions. Self-loops
/// are stored once. The undirected buffer is dropped before this function returns.
fn csr_from_undirected_edges<T, I>(edges: I, num_nodes: NodeIdx) -> AdjMat<T>
where
    T: EdgeWeight,
    I: Iterator<Item = Edge<T>>,
{
    let triplets: Vec<_> = edges
        .map(|Edge { src, dest, weight }| {
            if src <= dest {
                (src, dest, weight)
            } else {
                (dest, src, weight)
            }
        })
        .collect();
    csr_from_undirected_triplets(triplets, num_nodes)
}

/// Finish CSR construction from an undirected `(min, max, weight)` buffer.
///
/// Keeps the in-place sort + dedup + scatter path used by [`csr_from_undirected_edges`].
fn csr_from_undirected_triplets<T>(
    mut triplets: Vec<(NodeIdx, NodeIdx, T)>,
    num_nodes: NodeIdx,
) -> AdjMat<T>
where
    T: EdgeWeight,
{
    triplets.sort_unstable_by_key(|&(u, v, _)| (u, v));

    // Merge duplicate undirected pairs in place (adjacent after sort) by summing
    // weights. This avoids a second undirected buffer at peak construction memory.
    triplets.dedup_by(|later, earlier| {
        if later.0 == earlier.0 && later.1 == earlier.1 {
            earlier.2 = earlier.2 + later.2;
            true
        } else {
            false
        }
    });

    let mut row_nnz = vec![0usize; num_nodes];
    for &(u, v, _) in &triplets {
        row_nnz[u] += 1;
        if u != v {
            row_nnz[v] += 1;
        }
    }

    // Exclusive prefix sums become per-row write heads (`next`); copy them into
    // `indptr` so we do not keep a third n-vector.
    let mut current_offset = 0;
    for count in &mut row_nnz {
        let nnz = *count;
        *count = current_offset;
        current_offset += nnz;
    }
    let mut next = row_nnz;
    let nnz = current_offset;

    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.extend_from_slice(&next);
    indptr.push(nnz);
    assert!(num_nodes <= u32::MAX as usize, "node count exceeds u32");
    let mut indices = vec![0u32; nnz];
    let mut data = vec![T::zero(); nnz];

    for (u, v, weight) in triplets {
        let i = next[u];
        indices[i] = v as u32;
        data[i] = weight;
        next[u] += 1;
        if u != v {
            let j = next[v];
            indices[j] = u as u32;
            data[j] = weight;
            next[v] += 1;
        }
    }

    CsMatI::new((num_nodes, num_nodes), indptr, indices, data)
}

impl From<Graph<u8>> for Graph<usize> {
    fn from(graph: Graph<u8>) -> Graph<usize> {
        let shape = graph.adjacency_matrix.shape();
        let indptr = graph.adjacency_matrix.indptr();
        let indices = graph.adjacency_matrix.indices();
        let data = graph.adjacency_matrix.data();
        let adjacency_matrix = AdjMat::<usize>::new(
            shape,
            indptr.as_slice().unwrap().into(),
            indices.into(),
            data.iter().map(|&v| v as usize).collect(),
        );

        Graph::<usize>::from_adjacency_matrix(adjacency_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::types::{Edge, EdgeWeight, edges_from_tuples};

    #[test]
    fn test_graph_creation_and_stats() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        assert_eq!(graph.get_num_nodes(), 3); // 0, 1, 2
        assert_eq!(graph.get_edge_entry_count(), 3);
        assert_eq!(graph.get_total_edge_weight(), 3);
    }

    #[test]
    fn test_edge_metrics() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0), (0, 0), (0, 1)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        assert_eq!(graph.get_edge_entry_count(), 4);
        assert_eq!(graph.get_total_edge_weight(), 5);
    }

    #[test]
    fn test_graph_neighbors() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        let neighbors_0 = graph.neighbors(0);
        assert!(neighbors_0.contains(&1));
        assert!(neighbors_0.contains(&2));
        assert!(!neighbors_0.contains(&0));
    }

    #[test]
    #[should_panic(expected = "Node index out of bounds")]
    fn test_graph_neighbors_out_of_bounds() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        graph.neighbors(3); // out of bounds
    }

    #[test]
    fn test_graph_neighbors_complex() {
        //      0
        //     / \
        //    1   4
        //    |
        //    2
        //   / \
        //  3   5
        //
        //  6 (isolated)

        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        // Node 0: neighbors 1, 4
        let n0 = graph.neighbors(0);
        assert_eq!(n0.len(), 2);
        assert!(n0.contains(&1));
        assert!(n0.contains(&4));

        // Node 2: neighbors 1, 3, 5
        let n2 = graph.neighbors(2);
        assert_eq!(n2.len(), 3);
        assert!(n2.contains(&1));
        assert!(n2.contains(&3));
        assert!(n2.contains(&5));

        // Node 6: isolated
        let n6 = graph.neighbors(6);
        assert_eq!(n6.len(), 0);

        // Node 3: only neighbor is 2
        let n3 = graph.neighbors(3);
        assert_eq!(n3, vec![2]);
    }

    #[test]
    fn test_neighbors_iter_equivalent_graph_neighbors_complex() {
        // Same graph and expectations as `test_graph_neighbors_complex`,
        // but validating the allocation-free neighbor iterator.
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        let mut n0_alloc = graph.neighbors(0);
        n0_alloc.sort_unstable();
        let mut n0_iter: Vec<NodeIdx> = graph.neighbors_iter(0).collect();
        n0_iter.sort_unstable();
        assert_eq!(n0_alloc, n0_iter);

        let mut n2_alloc = graph.neighbors(2);
        n2_alloc.sort_unstable();
        let mut n2_iter: Vec<NodeIdx> = graph.neighbors_iter(2).collect();
        n2_iter.sort_unstable();
        assert_eq!(n2_alloc, n2_iter);

        assert!(graph.neighbors_iter(6).next().is_none());
        let n6_iter: Vec<NodeIdx> = graph.neighbors_iter(6).collect();
        assert_eq!(n6_iter, graph.neighbors(6));

        let n3_alloc = graph.neighbors(3);
        let n3_iter: Vec<NodeIdx> = graph.neighbors_iter(3).collect();
        assert_eq!(n3_iter, n3_alloc);
    }

    #[test]
    fn test_edges_from() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0), (0, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        let edges_from_0 = graph.get_edges_from(0);
        assert!(edges_from_0.contains(&Edge {
            src: 0,
            dest: 1,
            weight: 1
        }));
        assert!(edges_from_0.contains(&Edge {
            src: 0,
            dest: 2,
            weight: 1
        }));
        assert!(edges_from_0.contains(&Edge {
            src: 0,
            dest: 0,
            weight: 1
        }));
    }

    #[test]
    fn test_neighbors_iter_equivalent_neighbors() {
        // Include a self-loop to ensure `neighbors*` still excludes the node itself.
        let edges = vec![
            Edge::new(0, 1, Some(3_u8)),
            Edge::new(1, 2, Some(5_u8)),
            Edge::new(0, 0, Some(7_u8)),
        ];
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);

        let mut alloc_neighbors = graph.neighbors(0);
        let iter_neighbors: Vec<NodeIdx> = graph.neighbors_iter(0).collect();

        alloc_neighbors.sort_unstable();
        let mut iter_neighbors_sorted = iter_neighbors;
        iter_neighbors_sorted.sort_unstable();

        assert_eq!(alloc_neighbors, iter_neighbors_sorted);
    }

    #[test]
    fn test_edges_from_iter_equivalent_get_edges_from() {
        // Include a self-loop and varying weights to validate exact values.
        let edges = vec![
            Edge::new(0, 1, Some(3_u8)),
            Edge::new(1, 2, Some(5_u8)),
            Edge::new(0, 0, Some(7_u8)),
        ];
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);

        let mut alloc_edges = graph.get_edges_from(0);
        let mut iter_edges: Vec<Edge<u8>> = graph.edges_from_iter(0).collect();

        let mut alloc_tuples: Vec<(NodeIdx, NodeIdx, u8)> = alloc_edges
            .drain(..)
            .map(|e| (e.src, e.dest, e.weight))
            .collect();
        let mut iter_tuples: Vec<(NodeIdx, NodeIdx, u8)> = iter_edges
            .drain(..)
            .map(|e| (e.src, e.dest, e.weight))
            .collect();

        alloc_tuples.sort_unstable();
        iter_tuples.sort_unstable();

        assert_eq!(alloc_tuples, iter_tuples);
    }

    #[test]
    fn test_edges_in_selection() {
        //      ()
        //      0
        //     / \
        //    1   4
        //    |
        //    2
        //   / \
        //  3   5
        //
        //  6 (isolated)

        let edges = edges_from_tuples(vec![(0, 0), (0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        let selection: HashSet<NodeIdx> = vec![0, 1].into_iter().collect();
        let selected_edges: HashSet<Edge<u8>> = graph.get_edges_in_selection(&selection).collect();

        let expected_edges: HashSet<Edge<u8>> =
            edges_from_tuples(vec![(0, 0), (0, 1), (1, 2), (0, 4)]).collect();

        assert_eq!(selected_edges, expected_edges);
    }

    #[test]
    fn test_connected_components_isolated_nodes() {
        let graph = Graph::<u8>::from_edges(Vec::<Edge<u8>>::new().into_iter(), 4);
        assert_eq!(graph.connected_components(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_connected_components_simple() {
        // Graph: 0-1  2-3  4 (isolated)
        let edges = edges_from_tuples(vec![(0, 1), (2, 3)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 5);

        assert_eq!(graph.connected_components(), vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn test_connected_components_complex() {
        //      0
        //     / \
        //    1   4
        //    |
        //    2
        //   / \
        //  3   5
        //
        //  6 (isolated)
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        assert_eq!(graph.connected_components(), vec![0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_connected_components_self_loop() {
        let edges = vec![Edge::new(0, 0, Some(1_u8)), Edge::new(1, 2, Some(1_u8))];
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 4);

        assert_eq!(graph.connected_components(), vec![0, 1, 1, 2]);
    }

    #[test]
    fn test_connected_components_by() {
        // 0(-)1
        // |   |
        // 2(-)3
        //
        // Only consider components where all node indices have the same parity

        let parity = |src_node: NodeIdx, node: NodeIdx| src_node % 2 == node % 2;
        let tuples = vec![(0, 1), (0, 2), (1, 3), (2, 3)];

        let graph_u8 = Graph::<u8>::from_edges(edges_from_tuples(tuples.clone()), 4);
        let graph_usize = Graph::<usize>::from_edges(edges_from_tuples(tuples), 4);

        assert_eq!(graph_u8.connected_components_by(parity), vec![0, 1, 0, 1]);
        assert_eq!(
            graph_usize.connected_components_by(parity),
            vec![0, 1, 0, 1]
        );
    }

    /// Independent reference: union-find over the edges accepted by `filter`,
    /// with ids assigned in order of the lowest node index of each component.
    fn union_find_labels<T, F>(graph: &Graph<T>, filter: F) -> Vec<usize>
    where
        T: EdgeWeight,
        F: Fn(NodeIdx, NodeIdx) -> bool,
    {
        fn find(parents: &mut [usize], node: usize) -> usize {
            let mut root = node;
            while parents[root] != root {
                root = parents[root];
            }
            parents[node] = root;
            root
        }

        let n = graph.get_num_nodes();
        let mut parents: Vec<usize> = (0..n).collect();
        for edge in graph.get_edges_iter() {
            if filter(edge.src, edge.dest) {
                let (a, b) = (find(&mut parents, edge.src), find(&mut parents, edge.dest));
                parents[a.max(b)] = a.min(b);
            }
        }

        let mut ids = vec![usize::MAX; n];
        let mut next_id = 0;
        (0..n)
            .map(|node| {
                let root = find(&mut parents, node);
                if ids[root] == usize::MAX {
                    ids[root] = next_id;
                    next_id += 1;
                }
                ids[root]
            })
            .collect()
    }

    #[test]
    fn test_connected_components_match_union_find_on_random_graph() {
        use crate::common::test_utils::get_random_graph;
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let graph_u8 = get_random_graph::<u8>(200, 400, 0);
        let graph_usize = get_random_graph::<usize>(200, 400, 0);
        assert_eq!(
            graph_u8.connected_components(),
            union_find_labels(&graph_u8, |_, _| true)
        );
        assert_eq!(
            graph_usize.connected_components(),
            union_find_labels(&graph_usize, |_, _| true)
        );

        let mut rng = StdRng::seed_from_u64(0);
        let partition: Vec<usize> = (0..graph_u8.get_num_nodes())
            .map(|_| rng.random_range(0..5))
            .collect();
        let same_partition = |src: NodeIdx, node: NodeIdx| partition[src] == partition[node];
        assert_eq!(
            graph_u8.connected_components_by(same_partition),
            union_find_labels(&graph_u8, same_partition)
        );
        assert_eq!(
            graph_usize.connected_components_by(same_partition),
            union_find_labels(&graph_usize, same_partition)
        );
    }
}
