use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;

use itertools::Itertools;
use rand::prelude::*;

use crate::common::types::NodeIdx;

std::thread_local! {
    static FLP_RNG: std::cell::RefCell<Option<StdRng>> = const { std::cell::RefCell::new(None) };
}

/// Seed (Some) or unseed (None, the default: OS-seeded `rand::rng()`) the RNG used by fast label
/// propagation on this thread (node shuffle and mode tie-breaking). [pixelator-rs patch]
pub fn seed_flp_rng(seed: Option<u64>) {
    FLP_RNG.with(|r| *r.borrow_mut() = seed.map(StdRng::seed_from_u64));
}

/// Run `f` with the FLP RNG: the seeded one if set, else `rand::rng()`.
pub fn with_flp_rng<T>(f: impl FnOnce(&mut dyn RngCore) -> T) -> T {
    FLP_RNG.with(|r| match r.borrow_mut().as_mut() {
        Some(rng) => f(rng),
        None => f(&mut rand::rng()),
    })
}

/// Returns the mode (most common value) of a mutable slice of usize, breaking ties randomly.
///
/// The function sorts the input slice in-place, then finds the value that appears most frequently.
/// If there are multiple values with the same highest frequency, one is chosen at random.
/// This is efficient for small slices and avoids heap allocation for counting.
///
pub fn mode_via_sort(numbers: &mut [usize]) -> usize {
    if numbers.is_empty() {
        panic!("No numbers to choose from, `numbers` is empty");
    }

    numbers.sort_unstable();

    with_flp_rng(|rng| mode_with_rng(numbers, rng))
}

fn mode_with_rng(numbers: &[usize], rng: &mut dyn RngCore) -> usize {
    let ((_, best_n), _) = numbers.iter().dedup_with_count().fold(
        ((0, 0), 0),
        |((best_count, best_n), ties), (c, &n)| match c.cmp(&best_count) {
            Ordering::Less => ((best_count, best_n), ties),
            Ordering::Greater => ((c, n), 1),
            Ordering::Equal => {
                let ties = ties + 1;
                if rng.random_bool(1. / ties as f64) {
                    ((c, n), ties)
                } else {
                    ((best_count, best_n), ties)
                }
            }
        },
    );
    best_n
}

pub struct NodeQueue {
    queue: VecDeque<NodeIdx>,
    present: Vec<bool>,
}

impl fmt::Debug for NodeQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.queue.len() > 20 {
            write!(
                f,
                "NodeQueue(len: {}, front: {:?}, ... , back: {:?})",
                self.queue.len(),
                self.queue.iter().take(3).collect::<Vec<&NodeIdx>>(),
                self.queue.iter().rev().take(3).collect::<Vec<&NodeIdx>>()
            )
        } else {
            write!(
                f,
                "NodeQueue(len: {}, elements: {:?})",
                self.queue.len(),
                self.queue
            )
        }
    }
}
/// The node queue can be used to iterate nodes in order. It has a number of specific requirements:
/// - It assumes all nodes are consecutive integers starting from 0
/// - It assumes all nodes are present in the queue at the time of initialization.
/// - No new nodes will show up during processing, i.e. all indexes 0..n must be known at
///   initialization.
impl NodeQueue {
    pub fn from(vec: Vec<NodeIdx>) -> Self {
        let present = vec![true; vec.len()];

        NodeQueue {
            queue: VecDeque::from(vec),
            present,
        }
    }

    pub fn pop_front(&mut self) -> Option<NodeIdx> {
        let node = self.queue.pop_front();
        if let Some(n) = node {
            self.present[n] = false;
        }
        node
    }

    pub fn push_back(&mut self, node: NodeIdx) {
        if !self.present[node] {
            self.queue.push_back(node);
            self.present[node] = true;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn contains(&self, node: NodeIdx) -> bool {
        self.present[node]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet as HashSet;

    #[test]
    fn test_mode_via_sort_basic() {
        let mut data = vec![1, 2, 2, 3];
        let mode = mode_via_sort(&mut data);
        assert_eq!(mode, 2);
    }

    #[test]
    fn test_mode_via_sort_tie() {
        // Both 1 and 2 appear twice, so either is valid
        let mut data = vec![1, 2, 1, 2, 3];
        let mode = mode_via_sort(&mut data);
        assert!(mode == 1 || mode == 2);
    }

    #[test]
    fn test_mode_via_sort_single_element() {
        let mut data = vec![42];
        let mode = mode_via_sort(&mut data);
        assert_eq!(mode, 42);
    }

    #[test]
    #[should_panic]
    fn test_mode_via_sort_empty_panics() {
        let mut data: Vec<usize> = vec![];
        let _ = mode_via_sort(&mut data);
    }

    #[test]
    fn test_nodequeue_basic_operations() {
        let mut nq = NodeQueue::from(vec![0, 1, 2]);
        assert_eq!(nq.len(), 3);
        assert!(!nq.is_empty());
        assert!(nq.contains(1));

        assert_eq!(nq.pop_front(), Some(0));
        assert!(!nq.contains(0));
        assert_eq!(nq.len(), 2);

        // Already present in queue: should not duplicate.
        nq.push_back(2);
        assert_eq!(nq.len(), 2);

        // Re-enqueueing a popped node should append it.
        nq.push_back(0);
        assert_eq!(nq.len(), 3);
        assert_eq!(nq.pop_front(), Some(1));
        assert_eq!(nq.pop_front(), Some(2));
        assert_eq!(nq.pop_front(), Some(0));
        assert!(nq.is_empty());
    }

    #[test]
    fn test_nodequeue_uniqueness() {
        let mut nq = NodeQueue::from(vec![0, 1]);
        nq.push_back(1); // should not add again
        nq.push_back(0); // should not add again
        assert_eq!(nq.len(), 2);
        let mut seen = HashSet::default();
        while let Some(n) = nq.pop_front() {
            assert!(!seen.contains(&n));
            seen.insert(n);
        }
        assert!(nq.is_empty());
    }

    #[test]
    fn test_nodequeue_from_preserves_order() {
        let mut nq = NodeQueue::from(vec![0, 1, 2, 3]);
        assert_eq!(nq.pop_front(), Some(0));
        assert_eq!(nq.pop_front(), Some(1));
        assert_eq!(nq.pop_front(), Some(2));
        assert_eq!(nq.pop_front(), Some(3));
        assert_eq!(nq.pop_front(), None);
    }

    #[test]
    fn test_nodequeue_reenqueue_after_pop() {
        let mut nq = NodeQueue::from(vec![0, 1]);
        assert_eq!(nq.pop_front(), Some(0));
        assert!(!nq.contains(0));

        nq.push_back(0);
        assert!(nq.contains(0));
        assert_eq!(nq.pop_front(), Some(1));
        assert_eq!(nq.pop_front(), Some(0));
    }

    #[test]
    #[should_panic]
    fn test_nodequeue_contains_out_of_range_panics() {
        let nq = NodeQueue::from(vec![0, 1, 2]);
        let _ = nq.contains(3);
    }

    #[test]
    fn test_nodequeue_empty_invariants() {
        let mut nq = NodeQueue::from(vec![]);
        assert!(nq.is_empty());
        assert_eq!(nq.len(), 0);
        assert_eq!(nq.pop_front(), None);
    }

    #[test]
    #[should_panic]
    fn test_nodequeue_push_back_out_of_range_panics() {
        let mut nq = NodeQueue::from(vec![0, 1, 2]);
        nq.push_back(3);
    }

    #[test]
    fn test_nodequeue_all_nodes_present_at_initialization() {
        let nq = NodeQueue::from(vec![0, 1, 2, 3]);
        assert!(nq.contains(0));
        assert!(nq.contains(1));
        assert!(nq.contains(2));
        assert!(nq.contains(3));
    }
}
