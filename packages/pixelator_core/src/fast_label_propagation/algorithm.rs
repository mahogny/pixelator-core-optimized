use indicatif::ProgressBar;
use log::info;
use rand::seq::SliceRandom;
use std::time::Duration;
use std::time::Instant;

use crate::common::graph::Graph;
use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::NodeIdx;
use crate::common::utils::NodeQueue;
use crate::fast_label_propagation::strategies::AssignmentStrategy;

pub fn fast_label_propagation<P>(
    graph: &Graph<u8>,
    epochs: u64,
    assignment_strategy: &dyn AssignmentStrategy<P>,
    mut node_partition: P,
) -> P
where
    P: NodePartitioning,
{
    for epoch in 0..epochs {
        info!("Starting epoch {}...", epoch + 1);
        node_partition = label_propagation_epoch::<P>(graph, node_partition, assignment_strategy);
    }
    node_partition
}

pub fn label_propagation_epoch<P>(
    graph: &Graph<u8>,
    mut node_partition: P,
    assignment_strategy: &dyn AssignmentStrategy<P>,
) -> P
where
    P: NodePartitioning,
{
    let mut nodes: Vec<NodeIdx> = (0..graph.get_num_nodes()).collect();
    crate::common::utils::with_flp_rng(|rng| nodes.shuffle(rng));
    let mut node_queue = NodeQueue::from(nodes);

    info!("Starting label propagation...");
    let bar = ProgressBar::new_spinner();
    bar.enable_steady_tick(Duration::from_millis(100));
    let mut iteration = 0usize;
    let mut last_report = Instant::now();
    let mut last_iteration = 0usize;

    while let Some(node_i) = node_queue.pop_front() {
        iteration += 1;
        if iteration.is_multiple_of(1_000_000) {
            update_progress_bar(
                &node_queue,
                &bar,
                iteration,
                &mut last_report,
                &mut last_iteration,
            );
        }
        assignment_strategy.assign(graph, &mut node_partition, &mut node_queue, node_i);
    }
    bar.finish_and_clear();

    info!("Label propagation finished after {} iterations.", iteration);
    clean_partitions(graph, node_partition)
}

fn update_progress_bar(
    node_queue: &NodeQueue,
    bar: &ProgressBar,
    iteration: usize,
    last_report: &mut Instant,
    last_iteration: &mut usize,
) {
    let now = Instant::now();
    let elapsed = now.duration_since(*last_report).as_secs_f64();
    let iters = (iteration - *last_iteration) as f64;
    let rate_millions = if elapsed > 0.0 {
        iters / elapsed / 1_000_000.0
    } else {
        0.0
    };
    bar.set_message(format!(
        "Processing nodes. Iteration: {}, queue size: {}, {:.2} M nodes/s",
        iteration,
        node_queue.len(),
        rate_millions
    ));
    *last_report = now;
    *last_iteration = iteration;
}

/// Split disconnected partitions into their individual components
pub fn clean_partitions<P>(graph: &Graph<u8>, node_partition: P) -> P
where
    P: NodePartitioning,
{
    info!("Cleaning partitions by splitting disconnected components...");
    let labels = graph.connected_components_by(|src_node, node| {
        node_partition.get_partition_for_node(&src_node)
            == node_partition.get_partition_for_node(&node)
    });
    P::initialize_from_partitions(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::graph::Graph;
    use crate::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
    use crate::common::types::edges_from_tuples;
    use crate::fast_label_propagation::strategies::DefaultAssignmentStrategy;

    use crate::common::test_utils::{get_2_communities_edges, get_example_graph, get_random_graph};

    #[test]
    fn test_flp_epoch() {
        //   1   3
        //   |   |
        // 0-2   5-4
        //    \ /
        //     6
        //    /|\
        //   7-8-9
        let edges = edges_from_tuples(vec![
            (0, 2),
            (1, 2),
            (3, 5),
            (4, 5),
            (2, 6),
            (5, 6),
            (6, 7),
            (6, 8),
            (6, 9),
        ]);

        let graph = Graph::<u8>::from_edges(edges.into_iter(), 10);
        let mut partitioning = FastNodePartitioning::initialize_from_partitions(vec![
            //nodes:
            //  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        ]);

        partitioning = label_propagation_epoch(&graph, partitioning, &DefaultAssignmentStrategy);
        partitioning.normalize();

        let partition_map = partitioning.node_partition_map;

        assert_eq!(partition_map[0], partition_map[1]);
        assert_eq!(partition_map[0], partition_map[2]);

        assert_eq!(partition_map[3], partition_map[4]);
        assert_eq!(partition_map[3], partition_map[5]);

        assert_eq!(partition_map[6], partition_map[7]);
        assert_eq!(partition_map[6], partition_map[8]);
        assert_eq!(partition_map[6], partition_map[9]);

        assert_ne!(partition_map[0], partition_map[3]);
        assert_ne!(partition_map[0], partition_map[6]);
    }

    #[test]
    fn test_flp() {
        let graph = get_example_graph();
        let num_nodes = graph.get_num_nodes();
        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);

        assert_eq!(result_partition.node_partition_map, vec![0; num_nodes]);
    }

    #[test]
    fn test_flp_disconnected_graph() {
        //    2      5
        //   / \    / \
        //  0 - 1  3 - 4
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let num_nodes = 6;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<u8>::from_edges(edge_iter, num_nodes);

        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);
        let result_partition_map = result_partition.node_partition_map;

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);

        assert_eq!(result_partition_map[3], result_partition_map[4]);
        assert_eq!(result_partition_map[3], result_partition_map[5]);
    }

    #[test]
    fn test_flp_2_communities() {
        let graph = Graph::<u8>::from_edges(edges_from_tuples(get_2_communities_edges()), 8);
        let num_nodes = graph.get_num_nodes();
        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);
        let result_partition_map = result_partition.node_partition_map;

        assert_eq!(result_partition_map.len(), num_nodes);

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);
        assert_eq!(result_partition_map[0], result_partition_map[3]);

        // NOTE: in a few cases (<5%), this test will put all nodes in the same community and the
        // assertion below will fail.
        //assert_ne!(result_partition_map[0], result_partition_map[4]);

        assert_eq!(result_partition_map[4], result_partition_map[5]);
        assert_eq!(result_partition_map[4], result_partition_map[6]);
        assert_eq!(result_partition_map[4], result_partition_map[7]);
    }

    #[test]
    fn test_flp_no_edges() {
        let edges = vec![];
        let num_nodes = 3;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::from_edges(edge_iter, num_nodes);

        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);
        let result_partition_map = result_partition.get_node_to_partition_map();

        assert_ne!(result_partition_map[0], result_partition_map[1]);
        assert_ne!(result_partition_map[0], result_partition_map[2]);
    }

    #[test]
    fn test_flp_random() {
        let n_nodes = 1000;
        let n_edges = 2 * n_nodes;
        let seed = 0;
        let graph = get_random_graph(n_nodes, n_edges, seed);

        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(n_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);

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
    }

    #[test]
    fn test_clean_partitions_splits_disconnected_partitions() {
        // Path 0-1-2-3. Nodes 0, 1 and 3 share partition 5, but 3 is not connected to 0-1.
        let graph = Graph::<u8>::from_edges(edges_from_tuples(vec![(0, 1), (1, 2), (2, 3)]), 4);
        let partitioning = FastNodePartitioning::initialize_from_partitions(vec![5, 5, 6, 5]);

        let cleaned = clean_partitions(&graph, partitioning);

        assert_eq!(cleaned.get_node_to_partition_map(), [0, 0, 1, 2].as_slice());
    }
}
