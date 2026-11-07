//! [pixelator-rs patch] Deterministic parallel label propagation for bipartite graphs.
//!
//! Same update rule as `DefaultAssignmentStrategy` (a node adopts the most frequent label among
//! its neighbours; neighbours whose label differs from the new one are re-activated), scheduled in
//! half-rounds: all active nodes of side 0 are updated in parallel from side-1 labels, then all
//! active nodes of side 1 from side-0 labels. In a bipartite graph a side has no internal edges,
//! so a half-round reads only labels that it does not write: the result depends on the seed only,
//! not on the thread count or scheduling. As in FLP, ties are broken uniformly at random (here by a
//! hash of (seed, node, round)) among all tied labels, the current one included; random tie-breaks
//! are what lets labels spread in sparse graphs. A node only changes (and re-activates neighbours)
//! when the pick differs from its label; `max_rounds` bounds the rare flip-flopping tail.

use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use crate::common::graph::Graph;
use crate::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
use crate::fast_label_propagation::algorithm::clean_partitions;

const PREFETCH_AHEAD: usize = 8;

#[inline(always)]
fn prefetch_read<T>(p: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const T as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

#[inline]
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// `epochs` passes (like FLP's epochs: each pass re-activates every node), each run to convergence
/// or `max_rounds` full rounds. `side[v]` = which side of the bipartition node v is on. Returns the
/// partition after splitting disconnected labels (`clean_partitions`).
pub fn bipartite_label_propagation(graph: &Graph<u8>, side: &[bool], seed: u64, epochs: usize, max_rounds: usize) -> FastNodePartitioning {
    let n = graph.get_num_nodes();
    let label = bipartite_label_propagation_raw(graph, side, seed, 0, epochs, max_rounds, (0..n as u32).collect());
    let p = FastNodePartitioning::initialize_from_partitions(label.into_iter().map(|x| x as usize).collect());
    clean_partitions(graph, p)
}

/// The label propagation itself: epochs `first_epoch .. first_epoch + epochs` (the epoch number
/// enters the tie-break hash) from the given initial labels; returns raw labels (not split into
/// connected parts). Lets a caller renumber nodes between epochs.
pub fn bipartite_label_propagation_raw(
    graph: &Graph<u8>,
    side: &[bool],
    seed: u64,
    first_epoch: usize,
    epochs: usize,
    max_rounds: usize,
    init: Vec<u32>,
) -> Vec<u32> {
    let n = graph.get_num_nodes();
    assert_eq!(side.len(), n);
    assert_eq!(init.len(), n);
    let adj = graph.get_adjacency_matrix();
    let nbrs = |v: usize| -> &[u32] { adj.outer_view(v).map(|r| r.into_raw_storage().0).unwrap_or(&[]) };
    let mut label: Vec<u32> = init;
    let prefetch = std::env::var_os("PXR_PREFETCH").is_some();
    let queued: Vec<AtomicBool> = (0..n).into_par_iter().map(|_| AtomicBool::new(true)).collect();
    let mut active: [Vec<u32>; 2] = [
        (0..n as u32).into_par_iter().filter(|&v| !side[v as usize]).collect(),
        (0..n as u32).into_par_iter().filter(|&v| side[v as usize]).collect(),
    ];
    let mut updates = 0usize;
    for epoch in first_epoch as u64..(first_epoch + epochs) as u64 {
    if epoch > first_epoch as u64 {
        queued.par_iter().for_each(|q| q.store(true, Relaxed));
        active = [
            (0..n as u32).into_par_iter().filter(|&v| !side[v as usize]).collect(),
            (0..n as u32).into_par_iter().filter(|&v| side[v as usize]).collect(),
        ];
    }
    for r in 0..max_rounds as u64 {
        let round = epoch * max_rounds as u64 + r;
        if active[0].is_empty() && active[1].is_empty() {
            break;
        }
        for s in 0..2 {
            let act = std::mem::take(&mut active[s]);
            if act.is_empty() {
                continue;
            }
            act.par_iter().for_each(|&v| queued[v as usize].store(false, Relaxed));
            let lab = &label;
            let changes: Vec<(u32, u32)> = act
                .par_iter()
                .enumerate()
                .filter_map(|(k, &v)| {
                    if prefetch {
                        // issue the neighbour-label loads of a node a few positions ahead
                        if let Some(&w) = act.get(k + PREFETCH_AHEAD) {
                            for &u in nbrs(w as usize) {
                                prefetch_read(&lab[u as usize]);
                            }
                        }
                    }
                    let nb = nbrs(v as usize);
                    if nb.is_empty() {
                        return None;
                    }
                    let mut ls: SmallVec<[u32; 16]> = nb.iter().map(|&u| lab[u as usize]).collect();
                    ls.sort_unstable();
                    let cur = lab[v as usize];
                    // mode with ties
                    let mut best = 0usize;
                    let mut tied: SmallVec<[u32; 8]> = SmallVec::new();
                    let mut i = 0;
                    while i < ls.len() {
                        let mut j = i + 1;
                        while j < ls.len() && ls[j] == ls[i] {
                            j += 1;
                        }
                        let c = j - i;
                        if c > best {
                            best = c;
                            tied.clear();
                            tied.push(ls[i]);
                        } else if c == best {
                            tied.push(ls[i]);
                        }
                        i = j;
                    }
                    let pick = tied[(mix(seed ^ mix((v as u64) << 8 ^ round)) % tied.len() as u64) as usize];
                    if pick == cur { None } else { Some((v, pick)) }
                })
                .collect();
            updates += changes.len();
            for &(v, l) in &changes {
                label[v as usize] = l;
            }
            let lab = &label;
            let q = &queued;
            let next: Vec<u32> = changes
                .par_iter()
                .flat_map_iter(|&(v, l)| {
                    nbrs(v as usize).iter().copied().filter(move |&u| lab[u as usize] != l && !q[u as usize].swap(true, Relaxed))
                })
                .collect();
            active[1 - s].extend(next);
        }
        if std::env::var_os("PXR_MEMLOG").is_some() {
            eprintln!("bipartite LP epoch {} round {}: {} updates so far, active {} + {}", epoch + 1, r + 1, updates, active[0].len(), active[1].len());
        }
    }
    }
    label
}
