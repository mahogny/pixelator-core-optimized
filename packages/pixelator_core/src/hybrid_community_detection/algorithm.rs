use crate::common::graph::Graph;
use crate::common::graph::GraphProperties;
use crate::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use crate::fast_label_propagation::algorithm::fast_label_propagation;
use crate::fast_label_propagation::strategies::{AssignmentStrategy, DefaultAssignmentStrategy};
use crate::leiden::algorithm::leiden;
use crate::leiden::quality::QualityMetrics;
use crate::leiden::weighted_partitioned_graph::{
    AggregateOptions, PartitionedGraphStatistics, WeightedPartitionedGraph,
};
use log::debug;

/// Runs a hybrid community-detection pipeline: Fast Label Propagation (FLP) on the input graph,
/// aggregation into a [`WeightedPartitionedGraph`], then optionally Leiden refinement
/// (“multiplet recovery”).
///
/// # Arguments
///
/// * `graph` — Edge-weighted graph (`u8` per-edge weights), typically the raw adjacency before
///   aggregation.
/// * `quality_function` — Quality metric (e.g. modularity) used for the weighted partitioned
///   graph after FLP; drives node sizes and resolution during aggregation and Leiden.
/// * `randomness` - Randomness of node transitions. Low values will favor moves maximizing the
///   quality, while higher values will allow suboptimal moves, making it easier to avoid local
///   minima at the cost of convergence speed. Default value is 0.1
/// * `seed` — RNG seed for [`WeightedPartitionedGraph::new`]. If `None`, the graph uses its
///   default seed (0).
/// * `max_iteration` — Upper bound on Leiden iterations when `multiplet_recovery` is true; the
///   algorithm may stop earlier when converged.
/// * `flp_epochs` — Number of full FLP passes over `graph`.
/// * `refine_partitions` — If true, aggregate with [`AggregateOptions::OnlyWellConnected`]
///   using `randomness` between FLP and Leiden. If false, aggregate with [`AggregateOptions::All`]
///   instead (each partition becomes a single node).
/// * `multiplet_recovery` — If true, run [`leiden`] on the aggregated graph. If false, skip
///   Leiden; `post_leiden_statistics` then reflects the aggregated state without a Leiden pass.
///
/// # Returns
///
/// * Final node-to-partition map, maps node from the original graph to partitions
/// * Statistics of the weighted partitioned graph immediately after aggregation (post-FLP).
/// * Statistics after Leiden when `multiplet_recovery` is true, or the same aggregated statistics
///   when Leiden was skipped.
#[allow(clippy::too_many_arguments)]
fn memlog(tag: &str) {
    if std::env::var_os("PXR_MEMLOG").is_none() {
        return;
    }
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        let get = |k: &str| s.lines().find(|l| l.starts_with(k)).map(|l| l.split_whitespace().nth(1).unwrap_or("0").parse::<f64>().unwrap_or(0.0) / 1e6).unwrap_or(0.0);
        eprintln!("[mem] {tag:30} rss {:6.2} GB  peak {:6.2} GB", get("VmRSS:"), get("VmHWM:"));
    }
}

pub fn hybrid_community_detection(
    graph: Graph<u8>,
    quality_function: impl QualityMetrics,
    randomness: f64,
    seed: Option<u64>,
    max_iteration: Option<usize>,
    flp_epochs: u64,
    refine_partitions: bool,
    multiplet_recovery: bool,
) -> (
    LeidenNodePartitioning,
    PartitionedGraphStatistics,
    PartitionedGraphStatistics,
) {
    // [pixelator-rs patch] make FLP reproducible when a seed is given
    crate::common::utils::seed_flp_rng(seed);
    let graph_properties = GraphProperties::new(&graph);
    memlog("core: properties");
    debug!("Graph properties pre-aggregation: {:?}", graph_properties);

    let node_partition =
        FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let assignment_strategy: &dyn AssignmentStrategy<_> = &DefaultAssignmentStrategy;

    let node_partition = fast_label_propagation::<FastNodePartitioning>(
        &graph,
        flp_epochs,
        assignment_strategy,
        node_partition,
    );

    debug!(
        "Completed FLP. {} partitions found.",
        node_partition.num_partitions(),
    );

    memlog("core: FLP done");
    let partition = NodePartitioning::into_node_partitioning(node_partition);
    memlog("core: into leiden partitioning");

    // [pixelator-rs patch] aggregate directly from Graph<u8> (no full Graph<usize> copy)
    let direct = if refine_partitions {
        Err((graph, partition, quality_function))
    } else {
        WeightedPartitionedGraph::new_aggregated(graph, partition, quality_function, seed)
    };
    let mut wp_graph = match direct {
        Ok(wp) => wp,
        Err((graph, partition, quality_function)) => {
            let graph = Graph::<usize>::from(graph);
            let mut wp_graph =
                WeightedPartitionedGraph::new(graph, partition, quality_function, None, seed);
            wp_graph.aggregate(if refine_partitions {
                AggregateOptions::OnlyWellConnected(randomness)
            } else {
                AggregateOptions::All
            });
            wp_graph
        }
    };

    memlog("core: aggregated");
    let post_flp_statistics = wp_graph.get_statistics();

    if multiplet_recovery {
        leiden(&mut wp_graph, randomness, max_iteration, None);
    }
    let post_leiden_statistics = wp_graph.get_statistics();

    let final_partitioning = LeidenNodePartitioning::initialize_from_partitions(
        wp_graph.get_ancestor_to_partition_map(),
    );
    debug!(
        "Completed Leiden. {} partitions found. Final quality: {}.\n Graph Statistics: {:?}",
        final_partitioning.num_partitions(),
        post_leiden_statistics.quality,
        post_leiden_statistics,
    );

    (
        final_partitioning,
        post_flp_statistics,
        post_leiden_statistics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::graph::Graph;
    use crate::common::types::edges_from_tuples;
    use crate::leiden::quality::modularity::Modularity;

    use crate::common::test_utils::{get_2_communities_graph, get_example_graph, get_random_graph};

    #[test]
    fn test_hybrid_community_detection() {
        let graph = get_example_graph::<u8>();
        let num_nodes = graph.get_num_nodes();
        let quality = Modularity::new(0.1, graph.get_total_edge_weight());
        let randomness = 1.;
        let seed = None;

        let (result_partition, _, _) =
            hybrid_community_detection(graph, quality, randomness, seed, Some(100), 1, false, true);

        assert_eq!(
            result_partition.get_node_to_partition_map(),
            vec![0; num_nodes]
        );
    }

    #[test]
    fn test_hybrid_community_detection_empty_graph() {
        let graph = Graph::<u8>::from_edges(edges_from_tuples(vec![]), 0);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());

        let (result_partition, _, stats) =
            hybrid_community_detection(graph, quality, 0.1, None, Some(100), 1, false, true);

        assert!(result_partition.get_node_to_partition_map().is_empty());
        assert!(stats.partition_node_counts.is_empty());
        // Debug output is only built when debug logging is on; make sure it doesn't panic.
        let _ = format!("{:?}", stats);
    }

    #[test]
    fn test_hybrid_community_detection_disconnected_graph() {
        //    2      5
        //   / \    / \
        //  0 - 1  3 - 4
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let num_nodes = 6;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<u8>::from_edges(edge_iter, num_nodes);

        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let randomness = 1.;
        let seed = None;

        let (result_partition, _, _) =
            hybrid_community_detection(graph, quality, randomness, seed, Some(100), 1, false, true);
        let result_partition_map = result_partition.get_node_to_partition_map();

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);

        assert_eq!(result_partition_map[3], result_partition_map[4]);
        assert_eq!(result_partition_map[3], result_partition_map[5]);
    }

    #[test]
    fn test_hybrid_community_detection_2_communities() {
        let graph = get_2_communities_graph::<u8>();
        let num_nodes = graph.get_num_nodes();
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let randomness = 1.;
        let seed = None;

        let (result_partition, _, _) =
            hybrid_community_detection(graph, quality, randomness, seed, Some(100), 1, false, true);
        let result_partition_map = result_partition.get_node_to_partition_map();

        assert_eq!(result_partition_map.len(), num_nodes);

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);
        assert_eq!(result_partition_map[0], result_partition_map[3]);

        assert_eq!(result_partition_map[4], result_partition_map[5]);
        assert_eq!(result_partition_map[4], result_partition_map[6]);
        assert_eq!(result_partition_map[4], result_partition_map[7]);
    }

    #[test]
    fn test_hybrid_community_detection_random() {
        let n_nodes = 100;
        let n_edges = 2 * n_nodes;
        let seed = 0;
        let graph = get_random_graph(n_nodes, n_edges, seed);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let randomness = 1.;

        let (result_partition, post_flp_stats, post_leiden_stats) = hybrid_community_detection(
            graph.clone(),
            quality,
            randomness,
            Some(seed),
            Some(100),
            1,
            false,
            true,
        );

        assert_eq!(
            graph
                .connected_components_by(|node_1, node_2| result_partition
                    .get_node_to_partition_map()[node_1]
                    == result_partition.get_node_to_partition_map()[node_2])
                .into_iter()
                .max()
                .unwrap()
                + 1,
            result_partition.num_partitions()
        );

        assert_eq!(post_flp_stats.original_node_count, n_nodes);
        assert_eq!(post_flp_stats.current_edge_weight_sum, n_edges);
        assert_eq!(
            post_flp_stats.partition_node_counts.iter().sum::<usize>(),
            n_nodes
        );

        assert_eq!(post_leiden_stats.original_node_count, n_nodes);
        assert_eq!(post_leiden_stats.current_edge_weight_sum, n_edges);
        assert_eq!(
            post_leiden_stats
                .partition_node_counts
                .iter()
                .sum::<usize>(),
            n_nodes
        );
    }
}
