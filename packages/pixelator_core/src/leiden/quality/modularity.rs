use crate::common::graph::Graph;
use crate::common::types::NodeIdx;
use crate::leiden::quality::QualityMetrics;

#[derive(Clone)]
pub struct Modularity {
    pub scaled_resolution: f64,
}

impl Modularity {
    pub fn new(resolution: f64, total_edge_weight: usize) -> Self {
        let scaled_resolution = resolution / (2.0 * total_edge_weight as f64);
        Self { scaled_resolution }
    }
}

impl QualityMetrics for Modularity {
    fn resolution(&self) -> f64 {
        self.scaled_resolution
    }

    fn node_size(&self, graph: &Graph<usize>, node_id: NodeIdx) -> usize {
        graph.get_degrees()[node_id]
    }

    fn node_sizes_from_degrees(&self, degrees: &[usize], _num_nodes: usize) -> Option<Vec<usize>> {
        Some(degrees.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashSet as HashSet;

    use crate::common::graph::Graph;
    use crate::common::node_partitioning::{LeidenNodePartitioning, NodePartitioning};
    use crate::common::types::edges_from_tuples;

    use crate::common::test_utils::get_example_graph;

    use crate::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;

    #[test]
    fn test_partition_weight() {
        // 0   3
        //  \ /|
        //   2 |
        //  / \|
        // 1   4
        let edges = edges_from_tuples(vec![(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]);
        let graph = Graph::<usize>::from_edges(edges, 5);
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1]);
        let quality = Modularity::new(1.0, graph.get_total_edge_weight());

        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitions, quality, None, None);

        assert_eq!(wp_graph.get_partition_weight(0), 6);
        assert_eq!(wp_graph.get_partition_weight(1), 4);
        assert_eq!(wp_graph.delta(2, 1), -0.8);

        wp_graph.update_partition_slow(2, 1);

        assert_eq!(wp_graph.get_partition_weight(0), 2);
        assert_eq!(wp_graph.get_partition_weight(1), 8);
    }

    #[test]
    fn test_partition_weight_aggregated_graph() {
        // Same test but with high weight on node 2
        // 0   3
        //  \ /|
        //   2 |
        //  / \|
        // 1   4
        let edges = edges_from_tuples(vec![(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]);
        let graph = Graph::<usize>::from_edges(edges, 5);
        let node_weights: Vec<usize> = vec![1, 1, 8, 2, 2];
        let quality = Modularity::new(1.0, 7);
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1]);

        let mut wp_graph =
            WeightedPartitionedGraph::new(graph, partitions, quality, Some(node_weights), None);

        assert_eq!(wp_graph.get_partition_weight(0), 10);
        assert_eq!(wp_graph.get_partition_weight(1), 4);
        assert_eq!(wp_graph.delta(2, 1), -32. / 28.);

        wp_graph.update_partition_slow(2, 1);

        assert_eq!(wp_graph.get_partition_weight(0), 2);
        assert_eq!(wp_graph.get_partition_weight(1), 12);
    }

    #[test]
    fn test_modularity_intra_community_edges_increase_quality() {
        // Create a graph with two communities, each with three nodes and two edges within each community, plus some inter-community edges
        // Partition: [0,1,2] in community 0, [3,4,5] in community 1
        let edges = vec![
            (0, 1),
            (1, 2), // community 0
            (3, 4),
            (4, 5), // community 1
            (2, 3), // inter-community edge
            (0, 4), // more inter-community
        ];
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<usize>::from_edges(edge_iter, 6);
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1, 1]);
        let modularity = Modularity::new(1.0, graph.get_total_edge_weight());

        let wp_graph =
            WeightedPartitionedGraph::new(graph, partitions.clone(), modularity, None, None);

        let q_before = wp_graph.quality();

        // Add more intra-community edges: (0,2) and (3,5)
        let mut more_edges = edges.clone();
        more_edges.extend(vec![(0, 2), (3, 5)]); // Add one more intra edge to each community
        let edge_iter = edges_from_tuples(more_edges);
        let graph_more = Graph::<usize>::from_edges(edge_iter, 6);
        let modularity_more = Modularity::new(1.0, graph_more.get_total_edge_weight());
        let wp_graph_more =
            WeightedPartitionedGraph::new(graph_more, partitions, modularity_more, None, None);
        let q_after = wp_graph_more.quality();

        assert!(
            q_after > q_before,
            "Adding intra-community edges should increase modularity (before: {}, after: {})",
            q_before,
            q_after
        );
    }

    #[test]
    fn test_modularity_get_well_connected_nodes_in_subset() {
        let graph = get_example_graph::<usize>();
        let modularity = Modularity::new(1.0, graph.get_total_edge_weight());
        let partition =
            LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
        let wp_graph = WeightedPartitionedGraph::new(graph, partition, modularity, None, None);

        let subset: HashSet<usize> = vec![2, 3, 4, 5, 6, 7, 8].into_iter().collect();
        let subset_size = subset
            .iter()
            .map(|&id| wp_graph.get_node_weights()[id])
            .sum::<usize>();

        let mut well_connected_nodes = subset
            .iter()
            .filter(|&&node_id| wp_graph.node_is_well_connected(node_id, &subset, subset_size))
            .copied()
            .collect::<Vec<usize>>();
        well_connected_nodes.sort();

        assert_eq!(well_connected_nodes, vec![3, 4, 5, 6, 7, 8])
    }

    #[test]
    fn test_modularity_get_well_connected_communities() {
        let graph = get_example_graph::<usize>();
        let modularity = Modularity::new(1.0, graph.get_total_edge_weight());
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(9);
        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, modularity, None, None);

        let subset: HashSet<usize> = vec![2, 3, 4, 5, 6, 7, 8].into_iter().collect();
        let communities_to_evaluate: HashSet<usize> = vec![3, 4, 5, 7].into_iter().collect();
        let subset_size = subset
            .iter()
            .map(|&id| wp_graph.get_node_weights()[id])
            .sum::<usize>();

        let well_connected_communities = communities_to_evaluate
            .into_iter()
            .filter(|community_id| {
                wp_graph.community_is_well_connected(community_id, &subset, subset_size)
            })
            .collect::<HashSet<usize>>();

        let expected = HashSet::from_iter(vec![3, 4, 5, 7]);
        assert_eq!(well_connected_communities, expected);
    }
}
