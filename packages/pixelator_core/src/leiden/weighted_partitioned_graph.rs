use crate::common::constants::MIN_PNA_COMPONENT_SIZE;
use crate::common::graph::{Graph, GraphProperties};
use crate::common::node_partitioning::{LeidenNodePartitioning, NodePartitioning};
use crate::common::types::{Edge, NodeIdx, PartitionId};
use crate::leiden::quality::QualityMetrics;
use is_close::is_close;
use itertools::{Itertools, izip};
use rand::Rng;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::fmt;
use std::mem;

pub enum AggregateOptions {
    OnlyWellConnected(f64),
    All,
}

#[derive(Clone)]
pub struct PartitionedGraphStatistics {
    /// Sum of edge weights in the current (possibly aggregated) graph.
    pub current_edge_weight_sum: usize,
    /// Number of original graph nodes (i.e. ancestor nodes).
    pub original_node_count: usize,
    /// Number of nodes in the largest partition
    pub fraction_nodes_in_largest_partition: f64,
    /// Number of nodes in partitions smaller than 8000 umis
    pub stranded_nodes: usize,

    /// Sum of edge weights crossing partition boundaries.
    pub crossing_edge_weight_sum: usize,
    /// Number of original nodes in each partition.
    pub partition_node_counts: Vec<usize>,
    /// Quality of the partitioning
    pub quality: f64,
}

impl fmt::Debug for PartitionedGraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sorted_partition_sizes = self.partition_node_counts.clone();
        sorted_partition_sizes.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        f.debug_struct("PartitionedGraphStatistics")
            .field("current_edge_weight_sum", &self.current_edge_weight_sum)
            .field("original_node_count", &self.original_node_count)
            .field(
                "fraction_nodes_in_largest_partition",
                &self.fraction_nodes_in_largest_partition,
            )
            .field("stranded_nodes", &self.stranded_nodes)
            .field("crossing_edge_weight_sum", &self.crossing_edge_weight_sum)
            .field(
                "partition_node_counts",
                &format!(
                    "[{:?},...; {}]",
                    &sorted_partition_sizes[..sorted_partition_sizes.len().min(5)],
                    sorted_partition_sizes.len()
                ),
            )
            .field("quality", &self.quality)
            .finish()
    }
}

impl From<PartitionedGraphStatistics> for GraphProperties {
    /// Convert PartitionedGraphStatistics into GraphProperties, assuming crossing edges are
    /// removed.
    fn from(partitioned_graph_stats: PartitionedGraphStatistics) -> Self {
        GraphProperties {
            node_count: partitioned_graph_stats.original_node_count,
            edge_weight_sum: (partitioned_graph_stats.current_edge_weight_sum
                - partitioned_graph_stats.crossing_edge_weight_sum),
            n_connected_components: partitioned_graph_stats.partition_node_counts.len(),
            fraction_in_largest_component: partitioned_graph_stats
                .fraction_nodes_in_largest_partition,
            stranded_nodes: partitioned_graph_stats.stranded_nodes,
            component_size_distribution: partitioned_graph_stats
                .partition_node_counts
                .iter()
                .copied()
                .counts()
                .into_iter()
                .collect(),
        }
    }
}

/// Per-community node-weight totals, indexed by partition id.
///
/// Ids are dense in `0..len`.
#[derive(Clone)]
struct PartitionWeights {
    weights: Vec<usize>,
}

impl PartitionWeights {
    /// Sum `node_weights` into a vector indexed by the node→partition map.
    ///
    /// Every partition id must be `< new_partition_id`.
    fn from_node_partitions(
        node_to_partition: &[PartitionId],
        node_weights: &[usize],
        new_partition_id: PartitionId,
    ) -> Self {
        let mut weights = vec![0; new_partition_id];
        for (node_id, &partition_id) in node_to_partition.iter().enumerate() {
            weights[partition_id] += node_weights[node_id];
        }
        PartitionWeights { weights }
    }

    fn singlet_from_node_weights(node_weights: &[usize]) -> Self {
        PartitionWeights {
            weights: node_weights.to_vec(),
        }
    }

    fn get(&self, partition_id: PartitionId) -> usize {
        self.weights.get(partition_id).copied().unwrap_or(0)
    }

    fn add(&mut self, partition_id: PartitionId, delta: usize) {
        if partition_id >= self.weights.len() {
            self.weights.resize(partition_id + 1, 0);
        }
        self.weights[partition_id] += delta;
    }

    fn sub(&mut self, partition_id: PartitionId, delta: usize) {
        self.weights[partition_id] -= delta;
    }

    /// Remap non-empty partition ids through `partition_index`.
    fn remap_dense(&self, partition_index: &HashMap<PartitionId, PartitionId>) -> Self {
        let mut weights = vec![0; partition_index.len()];
        for (&old_pid, &new_pid) in partition_index.iter() {
            weights[new_pid] = self.get(old_pid);
        }
        PartitionWeights { weights }
    }
}

/// A graph with weighted nodes and a partitioning, used for Leiden community detection.
///
/// In particular, it provides the `aggregate` method that makes it possible to aggregate nodes
/// belonging to the same partition as described in the Leiden algorithm.
///
/// As partitions are refined and nodes are aggregated, this structure also keeps track of the
/// mapping from the original nodes (called ancestor nodes) to their current partitions,
/// which will change as the graph is aggregated
#[derive(Clone)]
pub struct WeightedPartitionedGraph<Q: QualityMetrics> {
    graph: Graph<usize>,
    partitions: LeidenNodePartitioning,
    node_weights: Vec<usize>,
    partition_weights: PartitionWeights,
    quality_metrics: Q,
    /// maps from ancestor node ids to current node ids
    /// which will change as the graph is aggregated
    ancestor_map: Vec<NodeIdx>,
    rng: StdRng,
}

impl<Q: QualityMetrics> WeightedPartitionedGraph<Q> {
    /// Create a new WeightedPartitionedGraph
    ///
    /// # Arguments
    /// - `graph`: The underlying graph structure.
    /// - `partitions`: The current partitioning of the nodes. Ids that are not already
    ///   `0..k-1` are renumbered to that range, in sorted label order.
    /// - `quality_metrics`: The quality metrics used for community detection. Defines how to
    ///   compute node sizes and resolution.
    /// - `node_weights`: Optional weights for each node. If None, weights are computed using using
    ///   `quality_metrics.node_size`.
    /// - `seed`: Optional seed for random number generation. If None, a default seed of 0 is used.
    pub fn new(
        graph: Graph<usize>,
        partitions: LeidenNodePartitioning,
        quality_metrics: Q,
        node_weights: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> Self {
        let partitions = if partitions.is_normalized() {
            partitions
        } else {
            let dense_ids = partitions.get_dense_mapping();
            LeidenNodePartitioning::initialize_from_partitions(
                partitions
                    .get_node_to_partition_map()
                    .iter()
                    .map(|id| dense_ids[id])
                    .collect(),
            )
        };
        let node_weights = node_weights.unwrap_or_else(|| {
            (0..graph.get_num_nodes())
                .map(|node_id| quality_metrics.node_size(&graph, node_id))
                .collect::<Vec<usize>>()
        });

        let partition_weights = PartitionWeights::from_node_partitions(
            partitions.get_node_to_partition_map(),
            &node_weights,
            partitions.get_new_partition_id(),
        );

        let ancestor_map = (0..graph.get_num_nodes()).collect::<Vec<NodeIdx>>();
        let rng = StdRng::seed_from_u64(seed.unwrap_or(0));

        WeightedPartitionedGraph {
            graph,
            partitions,
            node_weights,
            partition_weights,
            quality_metrics,
            ancestor_map,
            rng,
        }
    }

    /// [pixelator-rs patch] Equivalent to `new(Graph::<usize>::from(graph), partitions, q, None,
    /// seed)` followed by `aggregate(AggregateOptions::All)`, but built straight from `graph`
    /// without the full-size `Graph<usize>` copy (32 B/edge). Returns `Err(graph)` when the quality
    /// metric cannot give node sizes from degrees (callers then use the old path).
    pub fn new_aggregated<T: crate::common::types::EdgeWeight>(
        graph: Graph<T>,
        partitions: LeidenNodePartitioning,
        quality_metrics: Q,
        seed: Option<u64>,
    ) -> Result<Self, (Graph<T>, LeidenNodePartitioning, Q)> {
        let n = graph.get_num_nodes();
        let node_weights = match quality_metrics.node_sizes_from_degrees(graph.get_degrees(), n) {
            Some(w) => w,
            None => return Err((graph, partitions, quality_metrics)),
        };
        // --- as in `new`
        let partitions = if partitions.is_normalized() {
            partitions
        } else {
            let dense_ids = partitions.get_dense_mapping();
            LeidenNodePartitioning::initialize_from_partitions(
                partitions.get_node_to_partition_map().iter().map(|id| dense_ids[id]).collect(),
            )
        };
        let partition_weights = PartitionWeights::from_node_partitions(
            partitions.get_node_to_partition_map(),
            &node_weights,
            partitions.get_new_partition_id(),
        );
        // --- as in `aggregate(AggregateOptions::All)`
        let mut normalized_partition_map = vec![0; partitions.get_new_partition_id()];
        for (new_pid, (&old_pid, _)) in partitions
            .get_partition_to_node_map_ref()
            .iter()
            .filter(|(_, nodes)| !nodes.is_empty())
            .enumerate()
        {
            normalized_partition_map[old_pid] = new_pid;
        }
        let sub_partition_index = partitions
            .get_node_to_partition_map()
            .iter()
            .map(|&p_id| normalized_partition_map[p_id])
            .collect::<Vec<NodeIdx>>();
        drop(normalized_partition_map);
        let n_partitions = partitions.num_partitions();
        // Pre-merge parallel edges between the same pair of partitions (integer sums, so the
        // resulting CSR is identical to collecting all edges first) to avoid a 24 B/edge buffer.
        let mut merged: HashMap<(NodeIdx, NodeIdx), usize> = HashMap::default();
        for Edge { src, dest, weight } in graph.get_edges_iter().filter(|Edge { src, dest, .. }| src <= dest) {
            let w: usize = weight.into();
            *merged.entry((sub_partition_index[src], sub_partition_index[dest])).or_insert(0) += w;
        }
        let aggregated_graph = Graph::<usize>::from_edges(
            merged.into_iter().map(|((src, dest), weight)| Edge { src, dest, weight }),
            n_partitions,
        );
        drop(graph);
        let mut new_partitions = vec![0; n_partitions];
        let mut new_node_weights = vec![0; n_partitions];
        for (&new_pid, &pid) in sub_partition_index.iter().zip(partitions.get_node_to_partition_map().iter()) {
            new_partitions[new_pid] = pid;
            new_node_weights[new_pid] = partition_weights.get(pid);
        }
        let new_partitions = LeidenNodePartitioning::initialize_from_partitions(new_partitions);
        drop(partitions);
        Ok(WeightedPartitionedGraph {
            graph: aggregated_graph,
            partitions: new_partitions,
            node_weights: new_node_weights,
            partition_weights,
            quality_metrics,
            ancestor_map: sub_partition_index,
            rng: StdRng::seed_from_u64(seed.unwrap_or(0)),
        })
    }

    pub fn get_graph(&self) -> &Graph<usize> {
        &self.graph
    }

    pub fn get_partitioning(&self) -> &LeidenNodePartitioning {
        &self.partitions
    }

    pub fn get_partitioning_mut(&mut self) -> &mut LeidenNodePartitioning {
        &mut self.partitions
    }

    pub fn get_statistics(&self) -> PartitionedGraphStatistics {
        let current_edge_weight_sum = self.graph.get_total_edge_weight();
        let original_node_count = self.ancestor_map.len();

        let partition_node_counts: Vec<usize> = self
            .ancestor_map
            .iter()
            .map(|&node_id| self.partitions.get_node_to_partition_map()[node_id])
            .counts()
            .values()
            .copied()
            .collect();

        let fraction_nodes_in_largest_partition =
            partition_node_counts.iter().max().copied().unwrap_or(0) as f64
                / original_node_count as f64;
        let stranded_nodes = partition_node_counts
            .iter()
            .filter(|&&n_nodes| n_nodes < MIN_PNA_COMPONENT_SIZE)
            .sum();

        let crossing_edge_weight_sum = self
            .graph
            .get_edges_iter()
            .filter(|Edge { src, dest, .. }| {
                src <= dest
                    && self.partitions.get_node_to_partition_map()[*src]
                        != self.partitions.get_node_to_partition_map()[*dest]
            })
            .map(
                |Edge {
                     src: _,
                     dest: _,
                     weight,
                 }| weight,
            )
            .sum();

        PartitionedGraphStatistics {
            current_edge_weight_sum,
            original_node_count,
            fraction_nodes_in_largest_partition,
            stranded_nodes,
            crossing_edge_weight_sum,
            partition_node_counts,
            quality: self.quality(),
        }
    }

    /// Update the partition of a given node
    ///
    /// # Arguments
    /// - `node`: The node index whose partition is to be updated.
    /// - `new_partition`: The new partition id to assign to the node.
    pub fn update_partition_slow(&mut self, node: NodeIdx, new_partition: PartitionId) {
        let current_partition = self.partitions.get_partition_for_node(&node);
        let node_weight = self.node_weights[node];
        self.partition_weights.sub(current_partition, node_weight);
        self.partition_weights.add(new_partition, node_weight);
        self.partitions.update_partition(node, new_partition);
    }

    /// Update the partition of a given node without maintaining the expensive
    /// `partition_to_node_map` cache.
    ///
    /// This is intended to be called during the local moving phase, where
    /// we need to update the partition of a node quickly, without the overhead of rebuilding the
    /// `partition_to_node_map` cache.
    ///
    /// The cache will need to be rebuilt once before it is needed again (e.g. aggregation/refinement).
    pub fn update_partition_fast(&mut self, node: NodeIdx, new_partition: PartitionId) {
        let current_partition = self.partitions.get_partition_for_node(&node);
        let node_weight = self.node_weights[node];
        self.partition_weights.sub(current_partition, node_weight);
        self.partition_weights.add(new_partition, node_weight);
        self.partitions.update_partition_fast(node, new_partition);
    }

    /// Get the weights of all nodes
    pub fn get_node_weights(&self) -> &Vec<usize> {
        &self.node_weights
    }

    /// Get the total node weight of a partition
    pub fn get_partition_weight(&self, partition_id: PartitionId) -> usize {
        self.partition_weights.get(partition_id)
    }

    /// Get the mapping from ancestor nodes to their current partitions
    pub fn get_ancestor_to_partition_map(&self) -> Vec<PartitionId> {
        self.ancestor_map
            .iter()
            .map(|&node_id| self.partitions.get_node_to_partition_map()[node_id])
            .collect()
    }

    /// Normalize partitions so that all ids from 0 to |P| - 1 are used.
    pub fn normalize_partition(&mut self) {
        let partition_index = self.partitions.get_dense_mapping();

        self.partitions = LeidenNodePartitioning::initialize_from_partitions(
            self.partitions
                .get_node_to_partition_map()
                .iter()
                .map(|pid| partition_index[pid])
                .collect(),
        );

        // After remapping, ids are dense in 0..|P|; always store as a Vec.
        self.partition_weights = self.partition_weights.remap_dense(&partition_index);
    }

    /// Aggregate nodes sharing the same partition
    ///
    /// If `aggregate_opts` is `AggregateOptions::OnlyWellConnected`, partitions will be refined
    /// so that only well connected nodes are aggregated
    pub fn aggregate(&mut self, aggregate_opts: AggregateOptions) {
        let (super_partitions, super_partition_weights) = match aggregate_opts {
            AggregateOptions::OnlyWellConnected(randomness) => {
                let (super_partitions, super_partition_weights) =
                    self.refine_partitions(randomness);
                (Some(super_partitions), Some(super_partition_weights))
            }
            AggregateOptions::All => (None, None),
        };

        // filter out empty partitions, and normalize the partition ids from 0..n
        // since this is necessary for building the aggregated graph
        let mut normalized_partition_map = vec![0; self.partitions.get_new_partition_id()];
        for (new_pid, (&old_pid, _)) in self
            .partitions
            .get_partition_to_node_map_ref()
            .iter()
            .filter(|(_, nodes)| !nodes.is_empty())
            .enumerate()
        {
            normalized_partition_map[old_pid] = new_pid;
        }
        // Map node ids between the current graph and the aggregated graph
        let sub_partition_index = self
            .partitions
            .get_node_to_partition_map()
            .iter()
            .map(|&p_id| normalized_partition_map[p_id])
            .collect::<Vec<NodeIdx>>();

        let aggregated_graph = self.build_aggregated_graph(&sub_partition_index);

        let n_partitions = self.partitions.num_partitions();
        let (new_partitions, new_node_weights) =
            self.build_new_partitions(&sub_partition_index, super_partitions, n_partitions);

        self.ancestor_map
            .iter_mut()
            .for_each(|node_id| *node_id = sub_partition_index[*node_id]);

        self.graph = aggregated_graph;
        self.partitions = new_partitions;
        self.node_weights = new_node_weights;
        if let Some(super_partition_weights) = super_partition_weights {
            self.partition_weights = super_partition_weights;
        }
    }

    /// Refine current partitions to separate poorly connected nodes and clusters into different
    /// communities.
    ///
    /// The unrefined partitioning becomes super partitioning and is returned by this function,
    /// together with the weights of the super partitions.
    fn refine_partitions(&mut self, randomness: f64) -> (LeidenNodePartitioning, PartitionWeights) {
        // Reset current partitions and weights to singlet partitions
        let mut super_partitions =
            LeidenNodePartitioning::initialize_with_singlet_partitions(self.graph.get_num_nodes());
        mem::swap(&mut self.partitions, &mut super_partitions);

        // Singlet: partition id equals node id, so weights are a dense copy of node_weights.
        let mut super_partition_weights =
            PartitionWeights::singlet_from_node_weights(&self.node_weights);
        mem::swap(&mut self.partition_weights, &mut super_partition_weights);

        // Group well connected nodes belonging to the same super partition
        for (id, nodes) in super_partitions.get_partition_to_node_map_ref().iter() {
            self.merge_node_subset(nodes, super_partition_weights.get(*id), randomness);
        }

        (super_partitions, super_partition_weights)
    }

    /// Build new graph, aggregating nodes belonging to the same partitions into nodes
    /// in a new graph.
    ///
    /// `partition_index` maps each node id in the original graph to the id of its aggregated
    /// node (partition) in the new graph. The values are expected to be in the range
    /// `0..num_aggregated_nodes` so they can be used directly as node indices.
    fn build_aggregated_graph(&self, partition_index: &[NodeIdx]) -> Graph<usize> {
        let num_aggregated_nodes = self.partitions.num_partitions();

        Graph::from_edges(
            self.graph
                .get_edges_iter()
                .filter(|Edge { src, dest, .. }| src <= dest)
                .map(|Edge { src, dest, weight }| Edge {
                    src: partition_index[src],
                    dest: partition_index[dest],
                    weight,
                }),
            num_aggregated_nodes,
        )
    }

    /// Generate new partition table and partition weights, based on the new node ids.
    ///
    /// If the partitions have been refined, `super_partitions` can be used to specify the current
    /// partitioning. Otherwise, singlet partitions will be assumed.
    fn build_new_partitions(
        &self,
        sub_partition_index: &[PartitionId],
        super_partitions: Option<LeidenNodePartitioning>,
        n_partitions: usize,
    ) -> (LeidenNodePartitioning, Vec<usize>) {
        let mut new_partitions = vec![0; n_partitions];
        let mut new_node_weights = vec![0; n_partitions];
        izip!(
            sub_partition_index.iter(),
            self.partitions.get_node_to_partition_map().iter(),
            super_partitions
                .as_ref()
                .unwrap_or(&self.partitions)
                .get_node_to_partition_map()
                .iter(),
        )
        .for_each(|(&new_pid, refined_pid, &super_pid)| {
            new_partitions[new_pid] = super_pid;
            new_node_weights[new_pid] = self.partition_weights.get(*refined_pid);
        });

        (
            LeidenNodePartitioning::initialize_from_partitions(new_partitions),
            new_node_weights,
        )
    }

    /// Sort by partition and aggregate edges outgoing from `node`
    ///
    /// This function is typically called millions of time in a loop. To avoid repeatedly
    /// allocating a new vector each time, the return vector `aggregated_edge_weights` should be
    /// allocated outside the function and passed as a mutable reference.
    ///
    /// Given that the nodes we deal with typically have only a few edges (median is 3, 99th
    /// percentile is 22). The overhead of using a HashMap is prohibitive in this case.
    ///
    /// # Arguments
    /// - `node`: node whose edges are being aggregated
    /// - `aggregate_edge_weights`: variables where the aggregated edges will be stored.
    fn aggregate_edge_weights(
        &self,
        node: NodeIdx,
        aggregated_edge_weights: &mut Vec<(PartitionId, usize)>,
    ) {
        aggregated_edge_weights.clear();

        for Edge { src, dest, weight } in self.graph.edges_from_iter(node) {
            if src != dest {
                let dst_partition = self.partitions.get_partition_for_node(&dest);

                // Find the partition or push a new one.
                // For small degrees (e.g. < 50), this destroys a HashMap.
                match aggregated_edge_weights
                    .iter_mut()
                    .find(|(p, _)| *p == dst_partition)
                {
                    Some((_, total_weight)) => *total_weight += weight,
                    None => aggregated_edge_weights.push((dst_partition, weight)),
                }
            }
        }
    }

    /// Merge well connected nodes from the given subset
    ///
    /// This follows the implementation from the original Leiden description.
    fn merge_node_subset(
        &mut self,
        node_subset: &HashSet<NodeIdx>,
        node_subset_size: usize,
        randomness: f64,
    ) {
        // First consider only nodes that are well connected within the subset
        let mut well_connected_nodes = node_subset
            .iter()
            .filter(|&&node_id| self.node_is_well_connected(node_id, node_subset, node_subset_size))
            .cloned()
            .collect::<Vec<NodeIdx>>();

        // Visit them in random order
        well_connected_nodes.shuffle(&mut self.rng);

        let mut aggregated_edge_weights: Vec<(PartitionId, usize)> = Vec::new();
        let mut community_deltas: Vec<(PartitionId, f64)> = Vec::new();
        for node in well_connected_nodes.iter() {
            // Only consider moving nodes that are currently in singlet partitions
            // as we are calling this from the  refinement step, all nodes will
            // start in singlet partitions, but as we update this they will potentially
            // end up in larger partitions.
            if self
                .partitions
                .is_singlet_partition(&self.partitions.get_partition_for_node(node))
            {
                community_deltas.clear();
                let node_size = self.node_weights[*node];

                // Find the communities of the neighbors of node, since we can only transfer
                // to those communities
                self.aggregate_edge_weights(*node, &mut aggregated_edge_weights);

                let src_partition = self.partitions.get_partition_for_node(node);
                let src_size = self.partition_weights.get(src_partition);
                let src_weight = aggregated_edge_weights
                    .iter()
                    .find(|(p, _)| *p == src_partition)
                    .map(|(_, w)| *w)
                    .unwrap_or(0);
                let loss_src = self.loss_src(node_size, src_weight, src_size);

                community_deltas.extend(
                    aggregated_edge_weights
                        .iter()
                        .filter(|(community_id, _)| {
                            self.community_is_well_connected(
                                community_id,
                                node_subset,
                                node_subset_size,
                            )
                        })
                        .map(|&(community_id, total_edge_weight)| {
                            (
                                community_id,
                                loss_src
                                    + self.contrib_dest(
                                        node_size,
                                        total_edge_weight,
                                        self.partition_weights.get(community_id),
                                    ),
                            )
                        })
                        .filter(|&(_, delta)| delta > 0.),
                );

                // If none of the deltas are positive, we will not move anything
                if community_deltas.is_empty() {
                    continue;
                }
                // And then pick a random one from the weighted distribution of those that
                // have a positive modularity delta
                let partition_idx = pick_random_community_from_weighted_options(
                    &community_deltas,
                    randomness,
                    &mut self.rng,
                );
                self.update_partition_slow(*node, partition_idx);
            }
        }
    }

    /// Select the community that gives the best quality increase for the given node.
    ///
    /// This function is typically called millions of time in a loop. To avoid repeatedly
    /// allocating a new vector each time, the return vector `aggregated_edge_weights` should be
    /// allocated outside the function and passed as a mutable reference.
    pub fn best_community_for_node(
        &self,
        node: NodeIdx,
        aggregated_edge_weights: &mut Vec<(PartitionId, usize)>,
    ) -> Option<(PartitionId, f64)> {
        self.aggregate_edge_weights(node, aggregated_edge_weights);

        // If node has no neighbors
        if aggregated_edge_weights.is_empty() {
            return None;
        }

        let node_size = self.node_weights[node];
        let src_partition = self.partitions.get_partition_for_node(&node);

        aggregated_edge_weights
            .iter()
            .filter(|&&(partition, _)| partition != src_partition)
            .filter_map(|&(partition, total_edge_weight)| {
                let contrib = self.contrib_dest(
                    node_size,
                    total_edge_weight,
                    self.partition_weights.get(partition),
                );
                if contrib > 0.0 {
                    Some((partition, contrib))
                } else {
                    None
                }
            })
            .max_by(
                |(community_id_a, contrib_dest_a), (community_id_b, contrib_dest_b)| {
                    if is_close!(*contrib_dest_a, *contrib_dest_b) {
                        community_id_a.cmp(community_id_b)
                    } else {
                        contrib_dest_a
                            .partial_cmp(contrib_dest_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                },
            )
            // A new community will always have 0 contribution since no neighbor will be a member
            // of this community, and its size will be 0.
            .or(Some((self.partitions.get_new_partition_id(), 0.0)))
            .map(|(id, contrib_dest)| {
                let src_size = self.partition_weights.get(src_partition);
                let src_weight = aggregated_edge_weights
                    .iter()
                    .find(|(p, _)| *p == src_partition)
                    .map(|(_, w)| *w)
                    .unwrap_or(0);
                let loss_src = self.loss_src(node_size, src_weight, src_size);
                (id, contrib_dest + loss_src)
            })
    }

    /// Given a graph and a partitioning, computes the quality of this partitioning, based on the
    /// implementation of the resolution and node size functions.
    pub fn quality(&self) -> f64 {
        self.partitions
            .get_partition_to_node_map_ref()
            .iter()
            .map(|(&partition_id, nodes_in_partition)| {
                let subset_size = self.partition_weights.get(partition_id);
                let edge_count_in_partition = self
                    .graph
                    .get_edges_in_selection(nodes_in_partition)
                    .filter(|Edge { src, dest, .. }| {
                        nodes_in_partition.contains(src) && nodes_in_partition.contains(dest)
                    })
                    .map(
                        |Edge {
                             src: _,
                             dest: _,
                             weight,
                         }| weight,
                    )
                    .sum::<usize>();

                let possible_pairs_in_community =
                    (subset_size as f64 * (subset_size as f64 - 1.0)) / 2.0;
                edge_count_in_partition as f64
                    - self.quality_metrics.resolution() * possible_pairs_in_community
            })
            .sum()
    }

    #[allow(clippy::doc_overindented_list_items)]
    #[allow(dead_code)]
    /// Compute the quality delta induced by a given atomic change, i.e. one node moving from
    /// one partition to another.
    ///
    /// The formula for quality is:
    /// H(G, P) = ΣE(C,C) - γ|C|(|C| - 1)/2
    ///
    /// To compute the delta caused by moving a node from `src` to `dest`:
    /// 1. Remove edges from target v to src, add edges from v to dest
    ///    src  \|/   dest
    ///         -v
    ///         /|\
    ///
    /// 2. Add the changes from community sizes. The formula reads as follow: add back the
    ///    contributions for both source and destination communities and then remove the
    ///    contributions of the communities as they would be after target node v is moved from src
    ///    to dest
    ///
    ///    In raw form this is:
    ///    γ*(
    ///           |src|*(|src| - 1)/2
    ///         + |dst|*(|dst| - 1)/2
    ///         - (|src| - |v|)*(|src| - |v| - 1)/2
    ///         - (|dst| + |v|)*(|dst| + |v| - 1)/2
    ///       )
    ///
    ///    Surprisingly this simplifies into
    ///         γ * |v| * (|src| - |v| - |dst|)
    ///
    ///  When several destinations needs to be compared, steps 1. and 2. can be split further
    ///  into the loss from src and the contribution from dest, see `loss_src` and `contrib_dest`
    ///  below.
    ///
    ///  NB: calling this function repeatedly for all possible destination communities is quite
    ///  inefficient, use `contrib_dest` and `loss_src` instead.
    pub fn delta(&self, node_id: NodeIdx, dest_partition: PartitionId) -> f64 {
        let node_size = self.node_weights[node_id];

        let node_to_partition = self.partitions.get_node_to_partition_map();
        let dest_edge_total_weight: usize = self
            .graph
            .edges_from_iter(node_id)
            .filter(|Edge { src: _, dest, .. }| node_to_partition[*dest] == dest_partition)
            .map(
                |Edge {
                     src: _,
                     dest: _,
                     weight,
                 }| weight,
            )
            .sum();
        let dest_size = self.get_partition_weight(dest_partition);

        let src_partition = self.partitions.get_node_to_partition_map()[node_id];
        let src_edge_total_weight: usize = self
            .graph
            .edges_from_iter(node_id)
            .filter(|Edge { src: _, dest, .. }| node_to_partition[*dest] == src_partition)
            .map(
                |Edge {
                     src: _,
                     dest: _,
                     weight,
                 }| weight,
            )
            .sum();
        let src_size = self.get_partition_weight(src_partition);
        self.contrib_dest(node_size, dest_edge_total_weight, dest_size)
            + self.loss_src(node_size, src_edge_total_weight, src_size)
    }

    fn loss_src(&self, node_size: usize, src_edge_total_weight: usize, src_size: usize) -> f64 {
        -(src_edge_total_weight as f64)
            + self.quality_metrics.resolution()
                * (node_size as f64 * (src_size as f64 - node_size as f64))
    }

    fn contrib_dest(
        &self,
        node_size: usize,
        dest_edge_total_weight: usize,
        dest_size: usize,
    ) -> f64 {
        dest_edge_total_weight as f64
            - self.quality_metrics.resolution() * node_size as f64 * dest_size as f64
    }

    /// Returns well connected nodes among the given subset.
    ///
    /// Well connected nodes satisfy the following formula (from A.2, l.34 in supplementary
    /// information of the Leiden article <https://doi.org/10.1038/s41598-019-41695-z>):
    ///
    /// v ∈ S && E(v, S − v) ≥ γ ‖v‖ ⋅ (‖S‖ − ‖v‖)
    pub fn node_is_well_connected(
        &self,
        node_id: NodeIdx,
        node_subset: &HashSet<NodeIdx>,
        subset_size: usize,
    ) -> bool {
        let node_size = self.node_weights[node_id];

        let expected_connections_in_subset = self.quality_metrics.resolution()
            * (node_size as f64 * (subset_size as f64 - node_size as f64));
        // Count edges from `node_id` to nodes in `node_subset` excluding `node_id` itself.
        let actual_edges_in_subset: f64 = self
            .graph
            .edges_from_iter(node_id)
            .filter(|Edge { src: _, dest, .. }| *dest != node_id && node_subset.contains(dest))
            .map(
                |Edge {
                     src: _,
                     dest: _,
                     weight,
                 }| weight,
            )
            .sum::<usize>() as f64;

        actual_edges_in_subset >= expected_connections_in_subset
    }

    /// Returns communities that are included and well connected to the supercommunity S.
    ///
    /// Such communities satisfy the following formula (from A.2, l.37 in supplementary information
    /// of the Leiden article <https://doi.org/10.1038/s41598-019-41695-z>):
    ///
    /// C ∈ P && C ⊆ S && E(C, S − C) ≥ γ ‖C‖ ⋅ (‖S‖ − ‖C‖)
    pub fn community_is_well_connected(
        &self,
        community_id: &PartitionId,
        supercommunity: &HashSet<NodeIdx>,
        supercommunity_size: usize,
    ) -> bool {
        let community_size = self.partition_weights.get(*community_id);
        let nodes_in_community = &self.partitions.get_partition_to_node_map_ref()[community_id];

        // Check that all nodes in the community are inside the supercommunity
        // If they are not, do not consider this community
        if !nodes_in_community
            .iter()
            .all(|n| supercommunity.contains(n))
        {
            return false;
        }

        let actual_edge_count_in_community = self
            .graph
            .get_edges_in_selection(nodes_in_community)
            .filter(|Edge { src, dest, .. }| {
                (supercommunity.contains(src) && supercommunity.contains(dest))
                    && (!nodes_in_community.contains(src) || !nodes_in_community.contains(dest))
            })
            .map(
                |Edge {
                     src: _,
                     dest: _,
                     weight,
                 }| weight,
            )
            .sum::<usize>() as f64;
        let expected_edges = (self.quality_metrics.resolution() * community_size as f64)
            * (supercommunity_size as f64 - community_size as f64);

        actual_edge_count_in_community >= expected_edges
    }
}

fn pick_random_community_from_weighted_options<R: Rng>(
    community_options: &[(PartitionId, f64)],
    randomness: f64,
    rng: &mut R,
) -> PartitionId {
    let total_weights = community_options.iter().map(|(_, w)| *w).sum::<f64>();
    let weights = community_options
        .iter()
        .map(|(_, w)| (w / total_weights).powf(randomness))
        .collect::<Vec<f64>>();
    let dist = WeightedIndex::new(&weights).unwrap();

    community_options[dist.sample(rng)].0
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::test_utils::get_example_graph;
    use crate::common::types::edges_from_tuples;
    use crate::leiden::quality::cpm::ConstantPottsModel;
    use crate::leiden::quality::modularity::Modularity;

    #[test]
    fn test_merge_node_subset() {
        let graph = get_example_graph::<usize>();
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let partitioning =
            LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);

        let subset: HashSet<usize> = vec![2, 3, 4, 5, 6, 7, 8].into_iter().collect();
        let subset_size = subset
            .iter()
            .map(|&id| wp_graph.get_node_weights()[id])
            .sum::<usize>();

        wp_graph.merge_node_subset(&subset, subset_size, 1.0);

        // Nota bene: The expected result here is based on the current implementation
        // and we should make sure to verify them. For now I just wanted a based-line to
        // start from - looking at the example graph it does seem to be a reasonable result,
        // partitioning the nodes that have a high connectivity inside the subset into
        // one partition.
        let partitioning = wp_graph.get_partitioning();
        assert_eq!(partitioning.num_partitions(), 7);

        // This test is not deterministic, but these solutions seem to be the options
        assert!(
            partitioning.get_node_to_partition_map() == vec![0, 1, 2, 3, 5, 5, 6, 5, 5]
                || partitioning.get_node_to_partition_map() == vec![0, 1, 2, 3, 4, 5, 6, 5, 4]
                || partitioning.get_node_to_partition_map() == vec![0, 1, 2, 3, 4, 5, 6, 4, 5],
        );
    }

    #[test]
    fn test_best_community_for_node_can_create_new_communities() {
        let edges = edges_from_tuples(vec![(0, 1)]);
        let graph = Graph::<usize>::from_edges(edges, 2);
        // Choose a very high resolution to force the nodes apart.
        let cpm = ConstantPottsModel { resolution: 10. };
        // Place the two nodes in the same community
        let partitioning = LeidenNodePartitioning::initialize_with_null_partition(2);

        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);

        let mut out_edge_weights = Vec::new();
        let result = wp_graph
            .best_community_for_node(0, &mut out_edge_weights)
            .unwrap();
        assert_eq!(result.0, 1);
    }

    #[test]
    fn test_best_community_for_node_tie_selects_smallest_id() {
        // Create a simple graph: 0 -- 1, 0 -- 2
        let edges = edges_from_tuples(vec![(0, 1), (0, 2)]);
        let graph = Graph::<usize>::from_edges(edges, 3);

        // Partition: node 0 in community 0, node 1 in community 1, node 2 in community 2
        let partitioning = LeidenNodePartitioning::initialize_from_partitions(vec![0, 1, 2]);

        // Quality function: returns constant modularity for any partition (simulate tie)
        struct DummyQuality;
        impl QualityMetrics for DummyQuality {
            fn resolution(&self) -> f64 {
                1.
            }

            fn node_size(&self, _graph: &Graph<usize>, _node_id: NodeIdx) -> usize {
                1
            }
        }

        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, DummyQuality, None, None);

        let mut out_edge_weights = Vec::new();
        // Node 0 has neighbors 1 and 2, both communities will have same modularity
        let result = wp_graph.best_community_for_node(0, &mut out_edge_weights);
        // Should select the largest community id, in this case, this is the id for a new partition
        assert_eq!(result, Some((3, 0.0)));
    }

    #[test]
    fn test_aggregate_graph_basic() {
        // Graph: 0-1, 1-2, 2-0 (triangle)
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<usize>::from_edges(edges, 3);

        // Partition: nodes 0,1 in group 0; node 2 in group 1
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 1]);

        let cpm = ConstantPottsModel { resolution: 0.5 };
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitions, cpm, None, None);

        wp_graph.aggregate(AggregateOptions::All);
        let agg_graph = wp_graph.get_graph();

        // Should be a 2x2 matrix
        assert_eq!(agg_graph.get_num_nodes(), 2);

        // Check sum of edge weights (each edge between partitions contributes weight)
        // Edges: (0,1), (1,0), (0,0) for intra-group, etc.
        assert!(agg_graph.get_total_edge_weight() > 0);

        // Check neighbors for partition 0
        let neighbors_0 = agg_graph.neighbors(0);
        // Should have at least partition 1 as neighbor
        assert!(neighbors_0.contains(&1));
    }

    #[test]
    fn test_aggregate_graph_all_singletons() {
        // Each node in its own partition
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<usize>::from_edges(edges, 3);
        let partitions = LeidenNodePartitioning::initialize_with_singlet_partitions(3);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitions, cpm, None, None);

        wp_graph.aggregate(AggregateOptions::All);
        let agg_graph = wp_graph.get_graph();

        // Should be a 3x3 matrix
        assert_eq!(agg_graph.get_num_nodes(), 3);
        // Each edge should be between different partitions
        assert_eq!(agg_graph.get_total_edge_weight(), 3);
    }

    #[test]
    fn test_aggregate_graph_all_in_one_partition() {
        // All nodes in one partition
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<usize>::from_edges(edges, 3);
        let partitions = LeidenNodePartitioning::initialize_with_null_partition(3);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitions, cpm, None, None);

        wp_graph.aggregate(AggregateOptions::All);
        let agg_graph = wp_graph.get_graph();

        // Should be a 1x1 matrix
        assert_eq!(agg_graph.get_num_nodes(), 1);
        // All edges are intra-partition
        assert_eq!(agg_graph.get_edges_iter().count(), 1);
        assert_eq!(agg_graph.get_total_edge_weight(), 3);
        assert!(agg_graph.neighbors(0).is_empty());
    }

    #[test]
    fn test_aggregate_graph_complex() {
        // Graph:
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
        let graph = Graph::<usize>::from_edges(edges, 7);

        // Partition:
        // Nodes 0,1,4 in partition 0
        // Nodes 2,3,5 in partition 1
        // Node 6 in partition 2

        let partitions =
            LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 1, 1, 0, 1, 2]);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitions, cpm, None, None);
        wp_graph.aggregate(AggregateOptions::All);
        let graph = wp_graph.get_graph();

        assert_eq!(graph.get_num_nodes(), 3);

        let partition_to_node_map = wp_graph.get_partitioning().get_partition_to_node_map_ref();
        for &node in &partition_to_node_map[&0] {
            assert_eq!(graph.neighbors(node).len(), 1)
        }
        for &node in &partition_to_node_map[&1] {
            assert_eq!(graph.neighbors(node).len(), 1)
        }
        for &node in &partition_to_node_map[&2] {
            assert_eq!(graph.neighbors(node).len(), 0)
        }
    }

    #[test]
    fn test_aggregate_graph_multiple_edges_between_partitions() {
        // 1 - 0 - 3
        //  \ /   /
        //   2 - 4
        let edges = edges_from_tuples(vec![(0, 1), (0, 2), (1, 2), (3, 4), (0, 3), (2, 4)]);
        let graph = Graph::<usize>::from_edges(edges, 5);

        // Partition: nodes 0,1,2 in partition 0, node 3,4 in partition 1
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1]);
        let cpm = ConstantPottsModel { resolution: 0.5 };

        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitions, cpm, None, None);
        wp_graph.aggregate(AggregateOptions::All);
        let agg_graph = wp_graph.get_graph();

        // Should be a 2x2 matrix since we have 2 partitions
        assert_eq!(agg_graph.get_num_nodes(), 2);

        // Check sum of edge weights (including inside partitions)
        assert_eq!(agg_graph.get_total_edge_weight(), 6);

        // Check that the edge weight between partitions is 2 (edges (0,3) and (2,4))
        let weight_0_1 = agg_graph.get_edge_weight(0, 1).unwrap();
        assert_eq!(weight_0_1, 2);

        // Check neighbors for partition 0
        let neighbors_0 = agg_graph.neighbors(0);
        // Should have at least partition 1 as neighbor
        assert!(neighbors_0.contains(&1));
    }

    #[test]
    fn test_aggregate_graph_constant_total_weight() {
        let graph = get_example_graph::<usize>();
        let cpm = ConstantPottsModel { resolution: 0.5 };

        let p_refined =
            LeidenNodePartitioning::initialize_from_partitions(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);

        let mut wp_graph = WeightedPartitionedGraph::new(graph, p_refined, cpm, None, None);

        wp_graph.aggregate(AggregateOptions::All);

        let total_weight_1 = (0..=8)
            .map(|node| wp_graph.get_node_weights()[node])
            .sum::<usize>();

        let p_refined =
            LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
        let mut wp_graph = WeightedPartitionedGraph::new(
            get_example_graph::<usize>(),
            p_refined,
            ConstantPottsModel { resolution: 0.5 },
            None,
            None,
        );
        wp_graph.aggregate(AggregateOptions::All);
        let total_weight_2 = (0..=2)
            .map(|node| wp_graph.get_node_weights()[node])
            .sum::<usize>();

        assert_eq!(total_weight_1, total_weight_2);
    }

    #[test]
    fn test_aggregate_graph_weighted_edges() {
        let edges = vec![
            Edge::<usize> {
                src: 0,
                dest: 1,
                weight: 3,
            },
            Edge::<usize> {
                src: 0,
                dest: 2,
                weight: 2,
            },
        ];
        let graph = Graph::<usize>::from_edges(edges.into_iter(), 3);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let p_refined = LeidenNodePartitioning::initialize_from_partitions(vec![0, 1, 1]);

        let mut wp_graph = WeightedPartitionedGraph::new(graph, p_refined, cpm, None, None);
        wp_graph.aggregate(AggregateOptions::All);

        let graph = wp_graph.get_graph();

        assert_eq!(graph.get_edge_weight(0, 1), Some(5));
    }

    #[test]
    fn test_aggregate_refine() {
        //    1−4
        //   /
        //  0−2−5
        //   \
        //    3−6
        //
        //  γ threshold below which partition 0 is aggregated is 1/14

        let edges = edges_from_tuples(vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)]);
        let graph = Graph::<usize>::from_edges(edges, 7);
        let partitions =
            LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 0, 1, 1, 1]);

        let mut wp_graph = WeightedPartitionedGraph::new(
            graph.clone(),
            partitions.clone(),
            Modularity {
                scaled_resolution: 1. / 12.,
            },
            None,
            None,
        );
        wp_graph.aggregate(AggregateOptions::OnlyWellConnected(0.01));
        assert_eq!(wp_graph.get_graph().get_num_nodes(), 7);

        let mut wp_graph = WeightedPartitionedGraph::new(
            graph.clone(),
            partitions.clone(),
            Modularity {
                scaled_resolution: 1. / 16.,
            },
            None,
            None,
        );
        wp_graph.aggregate(AggregateOptions::OnlyWellConnected(0.1));
        assert_eq!(wp_graph.get_graph().get_num_nodes(), 4);
    }

    #[test]
    fn test_aggregate_self_loops() {
        //    1−4
        //   /
        //  0−2−5
        //   \
        //    3−6
        //
        //  γ threshold below which partition 0 is aggregated is 1/14

        let edges = edges_from_tuples(vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)]);
        let graph = Graph::<usize>::from_edges(edges, 7);
        let partitions =
            LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 0, 1, 1, 1]);

        let mut wp_graph = WeightedPartitionedGraph::new(
            graph.clone(),
            partitions.clone(),
            Modularity {
                scaled_resolution: 1. / 12.,
            },
            None,
            None,
        );
        wp_graph.aggregate(AggregateOptions::OnlyWellConnected(0.01));
        assert_eq!(wp_graph.get_graph().get_num_nodes(), 7);

        let mut wp_graph = WeightedPartitionedGraph::new(
            graph.clone(),
            partitions.clone(),
            Modularity {
                scaled_resolution: 1. / 16.,
            },
            None,
            None,
        );
        wp_graph.aggregate(AggregateOptions::OnlyWellConnected(0.1));
        assert_eq!(wp_graph.get_graph().get_num_nodes(), 4);
    }

    #[test]
    fn test_get_statistics() {
        let graph = get_example_graph::<usize>();
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let partitioning =
            LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);
        let stats = wp_graph.get_statistics();

        assert_eq!(stats.current_edge_weight_sum, 12);
        assert_eq!(stats.original_node_count, 9);
        assert_eq!(stats.fraction_nodes_in_largest_partition, 1. / 9.);
        assert_eq!(stats.stranded_nodes, 9);
        assert_eq!(stats.crossing_edge_weight_sum, 12);
        assert_eq!(stats.partition_node_counts, vec![1; 9]);
        assert_eq!(stats.quality, 0.);
    }

    #[test]
    fn test_get_statistics_cpm() {
        // 1 - 0 - 3
        //  \ /   /
        //   2 - 4
        let edges = edges_from_tuples(vec![(0, 1), (0, 2), (1, 2), (3, 4), (0, 3), (2, 4)]);
        let graph = Graph::<usize>::from_edges(edges, 5);

        // Partition: nodes 0,1,2 in partition 0, node 3,4 in partition 1
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1]);
        let cpm = ConstantPottsModel { resolution: 0.5 };

        let wp_graph = WeightedPartitionedGraph::new(graph, partitions, cpm, None, None);
        let statistics = wp_graph.get_statistics();

        let mut partition_sizes = statistics.partition_node_counts.clone();
        partition_sizes.sort_unstable();

        assert_eq!(statistics.current_edge_weight_sum, 6);
        assert_eq!(statistics.original_node_count, 5);
        assert_eq!(statistics.fraction_nodes_in_largest_partition, 3. / 5.);
        assert_eq!(statistics.stranded_nodes, 5);
        assert_eq!(statistics.crossing_edge_weight_sum, 2);
        assert_eq!(partition_sizes, vec![2, 3]);
        assert_eq!(statistics.quality, 2.);
    }

    #[test]
    fn test_get_statistics_modularity() {
        // 1 - 0 - 3
        //  \ /   /
        //   2 - 4
        let edges = edges_from_tuples(vec![(0, 1), (0, 2), (1, 2), (3, 4), (0, 3), (2, 4)]);
        let graph = Graph::<usize>::from_edges(edges, 5);

        // Partition: nodes 0,1,2 in partition 0, node 3,4 in partition 1
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1]);
        let modularity = Modularity::new(0.5, 6);

        let wp_graph = WeightedPartitionedGraph::new(graph, partitions, modularity, None, None);
        let statistics = wp_graph.get_statistics();

        let mut partition_sizes = statistics.partition_node_counts.clone();
        partition_sizes.sort_unstable();

        assert_eq!(statistics.current_edge_weight_sum, 6);
        assert_eq!(statistics.original_node_count, 5);
        assert_eq!(statistics.fraction_nodes_in_largest_partition, 3. / 5.);
        assert_eq!(statistics.stranded_nodes, 5);
        assert_eq!(statistics.crossing_edge_weight_sum, 2);
        assert_eq!(partition_sizes, vec![2, 3]);
        assert_eq!(statistics.quality, 31. / 12.);
    }

    #[test]
    fn test_sparse_input_partitions_are_normalized() {
        // Far-apart labels share communities and are renumbered in sorted order.
        let edges = edges_from_tuples(vec![(0, 1), (1, 2)]);
        let graph = Graph::<usize>::from_edges(edges, 3);
        let partitions =
            LeidenNodePartitioning::initialize_from_partitions(vec![50_000, 10_000, 10_000]);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let mut wp_graph =
            WeightedPartitionedGraph::new(graph, partitions, cpm, Some(vec![3, 4, 5]), None);

        assert_eq!(
            wp_graph.get_partitioning().get_node_to_partition_map(),
            &[1, 0, 0]
        );
        assert_eq!(wp_graph.get_partition_weight(0), 9);
        assert_eq!(wp_graph.get_partition_weight(1), 3);

        let new_id = wp_graph.get_partitioning().get_new_partition_id();
        assert_eq!(new_id, 2);
        wp_graph.update_partition_slow(2, new_id);
        assert_eq!(wp_graph.get_partition_weight(0), 4);
        assert_eq!(wp_graph.get_partition_weight(1), 3);
        assert_eq!(wp_graph.get_partition_weight(2), 5);
    }
}
