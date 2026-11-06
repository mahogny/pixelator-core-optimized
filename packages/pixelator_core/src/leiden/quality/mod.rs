use crate::common::graph::Graph;
use crate::common::types::NodeIdx;

pub trait QualityMetrics {
    /// Resolution paramater used to compute quality and well connected nodes and communities. In
    /// the Leiden paper, this is referenced as γ.
    fn resolution(&self) -> f64;

    /// Metric to determine the node size i.e. ||.||
    ///
    /// In the CPM, all nodes have a constant size of 1. In the context of modularity, the size of
    /// a node is the number of edges it has (namely, its degree).
    fn node_size(&self, graph: &Graph<usize>, node_id: NodeIdx) -> usize;

    /// [pixelator-rs patch] Node sizes computed directly from the unweighted degree vector, when
    /// the metric only depends on it (modularity). Lets callers avoid materializing a
    /// `Graph<usize>` copy. `None` = not supported for this metric.
    fn node_sizes_from_degrees(&self, _degrees: &[usize], _num_nodes: usize) -> Option<Vec<usize>> {
        None
    }
}

pub mod cpm;
pub mod modularity;
