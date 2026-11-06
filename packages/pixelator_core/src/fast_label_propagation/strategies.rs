use smallvec::SmallVec;

use crate::common::graph::Graph;
use crate::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
use crate::common::types::NodeIdx;
use crate::common::utils::{NodeQueue, mode_via_sort};

pub trait AssignmentStrategy<P> {
    fn assign(
        &self,
        graph: &Graph<u8>,
        node_partition: &mut P,
        node_queue: &mut NodeQueue,
        current_node: NodeIdx,
    );
}

pub struct DefaultAssignmentStrategy;

impl AssignmentStrategy<FastNodePartitioning> for DefaultAssignmentStrategy {
    fn assign(
        &self,
        graph: &Graph<u8>,
        node_partition: &mut FastNodePartitioning,
        node_queue: &mut NodeQueue,
        current_node: NodeIdx,
    ) {
        // Not using the built in neighbors method to avoid allocation of a Vec
        let neighbors_view = graph
            .get_adjacency_matrix()
            .outer_view(current_node)
            .unwrap();
        if neighbors_view.nnz() == 0 {
            return;
        }
        let neighbors = neighbors_view.indices();

        // Using small vec to avoid heap allocation for small number of neighbors
        // which should be the case for most of the nodes in the graph
        let mut partitions = SmallVec::<[usize; 8]>::new();
        partitions.extend(
            neighbors
                .iter()
                .map(|&x| node_partition.get_partition_for_node(&(x as usize))),
        );

        // Finding the most common partition among neighbors by sorting and picking the mode
        // from the vector is more efficient than using a hash map to count occurrences
        // for thse small vectors we expect here.
        let most_common_neighbors_partition = mode_via_sort(partitions.as_mut_slice());
        let node_i_current_partition = node_partition.get_partition_for_node(&current_node);

        if most_common_neighbors_partition != node_i_current_partition {
            node_partition.update_partition(current_node, most_common_neighbors_partition);
            for neighbor_to_update in neighbors
                .iter()
                .map(|&x| x as usize)
                .filter(|neighbor| {
                    node_partition.get_partition_for_node(neighbor)
                        != most_common_neighbors_partition
                })
            {
                node_queue.push_back(neighbor_to_update);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::edges_from_tuples;
    use crate::common::utils::NodeQueue;

    use crate::common::graph::Graph;
    use crate::common::node_partitioning::FastNodePartitioning;

    #[test]
    fn test_assign_fastnodepartitioning_partition_update() {
        // Create a simple graph: 0-1, 0-2, 1-2
        let edges = edges_from_tuples(vec![(0, 1), (0, 2), (1, 2)]);
        let graph = Graph::from_edges(edges.into_iter(), 3);

        // Initial partition: node 0 in 0, node 1 in 1, node 2 in 2
        let mut partition = FastNodePartitioning::initialize_with_singlet_partitions(3);
        let mut queue = NodeQueue::from(vec![0, 1, 2]);
        let node_i = 0;

        // Assign should move node 0 to the most common neighbor partition (which is a tie, but will pick one)
        let strategy = DefaultAssignmentStrategy;
        strategy.assign(&graph, &mut partition, &mut queue, node_i);

        // After assignment, node 0 should be in the same partition as either node 1 or node 2
        let p0 = partition.get_partition_for_node(&0);
        let p1 = partition.get_partition_for_node(&1);
        let p2 = partition.get_partition_for_node(&2);
        assert!(p0 == p1 || p0 == p2);

        assert!(!queue.is_empty());
    }
}
