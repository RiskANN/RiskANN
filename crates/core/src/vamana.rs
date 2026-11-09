//! Vamana-inspired graph search and pruning.
//!
//! # Ordering
//!
//! [`Candidate`] has a **total order**: ascending distance (via
//! [`f32::total_cmp`]), then ascending node id. Equal distances are therefore
//! broken by the lower internal id, which makes every result list
//! deterministic for an unchanged index.
//!
//! # Beam search ([`beam_search`])
//!
//! * **Frontier** — a min-heap (`BinaryHeap<Reverse<Candidate>>`); the
//!   nearest not-yet-expanded candidate is expanded first.
//! * **Result set** — a max-heap (`BinaryHeap<Candidate>`) bounded to `ef`
//!   entries; its top is the *worst* retained candidate, so admission and
//!   eviction are `O(log ef)`.
//! * A newly reached neighbour is admitted only if the result set is not yet
//!   full or the neighbour is better than the current worst result. Admitted
//!   neighbours are pushed to both heaps.
//! * **Stopping rule** — stop when the frontier is empty, or when the result
//!   set is full and the nearest frontier candidate is worse than the worst
//!   retained result (it, and everything after it, can no longer improve the
//!   result set). This is equivalent to Vamana's GreedySearch with list size
//!   `L = ef`: the search ends once every candidate in the list has been
//!   expanded.
//! * Every node is distance-evaluated at most once (bitset visited set), so
//!   the search terminates on cyclic graphs and never returns duplicates.
//!   Neighbour ids `>= num_nodes` are skipped and counted, never accessed.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fmt;

/// A node id paired with its distance to the current query.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub id: u32,
    pub distance: f32,
}

impl Candidate {
    pub fn new(id: u32, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

/// Errors from graph search or pruning.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchError {
    /// Search or pruning parameters are invalid (e.g. `k == 0`, `ef < k`).
    InvalidParams(String),
    /// The entry point is not a valid node id.
    EntryOutOfRange { entry: u32, num_nodes: usize },
    /// A distance callback produced NaN or an infinity.
    NonFiniteDistance { id: u32 },
    /// A storage callback failed for this node.
    Access { id: u32, message: String },
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::InvalidParams(m) => write!(f, "invalid search parameters: {m}"),
            SearchError::EntryOutOfRange { entry, num_nodes } => {
                write!(f, "entry point {entry} is out of range ({num_nodes} nodes)")
            }
            SearchError::NonFiniteDistance { id } => {
                write!(f, "non-finite distance computed for node {id}")
            }
            SearchError::Access { id, message } => {
                write!(f, "failed to access node {id}: {message}")
            }
        }
    }
}

impl std::error::Error for SearchError {}

/// Per-search counters. Collected with plain integer increments, so they do
/// not materially affect latency measurements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Distinct nodes reached (marked visited), including the entry point.
    pub visited_nodes: usize,
    /// Nodes whose adjacency list was read.
    pub expanded_nodes: usize,
    /// Query-to-node distance evaluations. Equal to `visited_nodes` because
    /// each visited node is evaluated exactly once.
    pub distance_computations: usize,
    /// Neighbour ids `>= num_nodes` that were ignored.
    pub skipped_invalid_edges: usize,
}

/// Configuration for [`beam_search`].
#[derive(Debug, Clone, Copy)]
pub struct BeamSearchConfig {
    /// Result-list size (Vamana's `L`, HNSW's `ef`). Must be `>= 1`.
    pub ef: usize,
    /// Valid node ids are `0..num_nodes`.
    pub num_nodes: usize,
    /// Record expanded nodes (needed by graph construction, not by queries).
    pub collect_expanded: bool,
}

/// Output of [`beam_search`] / [`vamana_search`].
#[derive(Debug, Clone, Default)]
pub struct BeamSearchOutput {
    /// Unique candidates in ascending `(distance, id)` order.
    pub results: Vec<Candidate>,
    /// Expanded nodes in expansion order (empty unless `collect_expanded`).
    pub expanded: Vec<Candidate>,
    pub stats: SearchStats,
}

/// Query-level parameters: `k` final results from an exploration budget of
/// `ef_search` candidates.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub k: usize,
    pub ef_search: usize,
}

struct VisitedSet {
    bits: Vec<u64>,
}

impl VisitedSet {
    fn new(num_nodes: usize) -> Self {
        Self {
            bits: vec![0u64; num_nodes.div_ceil(64)],
        }
    }

    /// Marks `id` visited. Returns `true` if it was not visited before.
    /// Ids beyond the set are reported as already visited (never followed);
    /// callers check bounds first, so this is purely defensive.
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        let i = id as usize;
        let mask = 1u64 << (i % 64);
        match self.bits.get_mut(i / 64) {
            Some(word) => {
                let fresh = *word & mask == 0;
                *word |= mask;
                fresh
            }
            None => false,
        }
    }
}

#[inline]
fn finite(id: u32, d: f32) -> Result<f32, SearchError> {
    if d.is_finite() {
        Ok(d)
    } else {
        Err(SearchError::NonFiniteDistance { id })
    }
}

/// Bounded best-first beam search over a graph (see module docs).
///
/// The search starts from every node in `entries` (duplicates are ignored;
/// at most `ef` of them are kept in the beam). Several entry points spread
/// across the data keep regions of the graph reachable that a single entry
/// point might not reach.
///
/// * `distance_to_query(id)` returns the distance from the query to node `id`.
/// * `neighbors(id, out)` appends the adjacency list of `id` to `out`
///   (`out` is cleared by the caller before every call).
///
/// Returns at most `config.ef` results. An empty graph (`num_nodes == 0`)
/// yields an empty result.
pub fn beam_search<FD, FN>(
    entries: &[u32],
    config: &BeamSearchConfig,
    mut distance_to_query: FD,
    mut neighbors: FN,
) -> Result<BeamSearchOutput, SearchError>
where
    FD: FnMut(u32) -> Result<f32, SearchError>,
    FN: FnMut(u32, &mut Vec<u32>) -> Result<(), SearchError>,
{
    if config.ef == 0 {
        return Err(SearchError::InvalidParams("ef must be at least 1".into()));
    }
    let mut out = BeamSearchOutput::default();
    if config.num_nodes == 0 {
        return Ok(out);
    }
    if entries.is_empty() {
        return Err(SearchError::InvalidParams(
            "at least one entry point is required".into(),
        ));
    }
    if let Some(&entry) = entries.iter().find(|&&e| (e as usize) >= config.num_nodes) {
        return Err(SearchError::EntryOutOfRange {
            entry,
            num_nodes: config.num_nodes,
        });
    }

    let mut visited = VisitedSet::new(config.num_nodes);
    let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(config.ef + 1);
    let mut buf: Vec<u32> = Vec::new();

    for &entry in entries {
        if !visited.insert(entry) {
            continue;
        }
        let d = finite(entry, distance_to_query(entry)?)?;
        out.stats.visited_nodes += 1;
        out.stats.distance_computations += 1;
        let cand = Candidate::new(entry, d);
        if results.len() < config.ef || results.peek().is_some_and(|w| cand < *w) {
            frontier.push(Reverse(cand));
            results.push(cand);
            if results.len() > config.ef {
                results.pop();
            }
        }
    }

    while let Some(Reverse(current)) = frontier.pop() {
        if results.len() >= config.ef && results.peek().is_some_and(|worst| current > *worst) {
            break;
        }

        out.stats.expanded_nodes += 1;
        if config.collect_expanded {
            out.expanded.push(current);
        }

        buf.clear();
        neighbors(current.id, &mut buf)?;
        for &n in &buf {
            if (n as usize) >= config.num_nodes {
                out.stats.skipped_invalid_edges += 1;
                continue;
            }
            if !visited.insert(n) {
                continue;
            }
            let d = finite(n, distance_to_query(n)?)?;
            out.stats.visited_nodes += 1;
            out.stats.distance_computations += 1;
            let cand = Candidate::new(n, d);
            let admit = results.len() < config.ef || results.peek().is_some_and(|w| cand < *w);
            if admit {
                frontier.push(Reverse(cand));
                results.push(cand);
                if results.len() > config.ef {
                    results.pop();
                }
            }
        }
    }

    out.results = results.into_sorted_vec();
    Ok(out)
}

/// Approximate k-nearest-neighbour query: validates `1 <= k <= ef_search`,
/// runs [`beam_search`] with `ef = ef_search`, and keeps the best `k`.
pub fn vamana_search<FD, FN>(
    entries: &[u32],
    params: SearchParams,
    num_nodes: usize,
    distance_to_query: FD,
    neighbors: FN,
) -> Result<BeamSearchOutput, SearchError>
where
    FD: FnMut(u32) -> Result<f32, SearchError>,
    FN: FnMut(u32, &mut Vec<u32>) -> Result<(), SearchError>,
{
    if params.k == 0 {
        return Err(SearchError::InvalidParams("k must be at least 1".into()));
    }
    if params.ef_search < params.k {
        return Err(SearchError::InvalidParams(format!(
            "ef_search ({}) must be >= k ({})",
            params.ef_search, params.k
        )));
    }
    let config = BeamSearchConfig {
        ef: params.ef_search,
        num_nodes,
        collect_expanded: false,
    };
    let mut out = beam_search(entries, &config, distance_to_query, neighbors)?;
    out.results.truncate(params.k);
    Ok(out)
}

/// Vamana RobustPrune.
///
/// `candidates` carry their distance to `node`. Candidates are processed in
/// ascending `(distance, id)` order; candidate `p` is kept iff for every
/// already-selected `s`: `alpha * d(s, p) > d(node, p)`. Selection stops at
/// `r` neighbours. `node` itself, duplicate ids and non-finite candidate
/// distances are discarded, so the result never contains a self-edge or a
/// duplicate edge and has at most `r` entries.
pub fn robust_prune<FP>(
    node: u32,
    candidates: &[Candidate],
    r: usize,
    alpha: f32,
    mut dist_between: FP,
) -> Result<Vec<u32>, SearchError>
where
    FP: FnMut(u32, u32) -> Result<f32, SearchError>,
{
    if r == 0 {
        return Err(SearchError::InvalidParams("r must be at least 1".into()));
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(SearchError::InvalidParams(
            "alpha must be a positive finite number".into(),
        ));
    }

    let mut pool: Vec<Candidate> = candidates
        .iter()
        .copied()
        .filter(|c| c.id != node && c.distance.is_finite())
        .collect();
    pool.sort();
    let mut seen: HashSet<u32> = HashSet::with_capacity(pool.len());
    pool.retain(|c| seen.insert(c.id));

    let mut selected: Vec<u32> = Vec::with_capacity(r);
    for p in pool {
        if selected.len() >= r {
            break;
        }
        let mut keep = true;
        for &s in &selected {
            let d = finite(p.id, dist_between(s, p.id)?)?;
            if alpha * d <= p.distance {
                keep = false;
                break;
            }
        }
        if keep {
            selected.push(p.id);
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distances::l2_distance;

    fn search(
        points: &[Vec<f32>],
        adj: &[Vec<u32>],
        entry: u32,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<BeamSearchOutput, SearchError> {
        vamana_search(
            &[entry],
            SearchParams { k, ef_search: ef },
            points.len(),
            |id| Ok(l2_distance(&points[id as usize], query)),
            |id, out| {
                out.extend_from_slice(&adj[id as usize]);
                Ok(())
            },
        )
    }

    fn ids(out: &BeamSearchOutput) -> Vec<u32> {
        out.results.iter().map(|c| c.id).collect()
    }

    fn assert_sorted_unique(out: &BeamSearchOutput) {
        for w in out.results.windows(2) {
            assert!(
                w[0] < w[1],
                "results not strictly ascending: {:?}",
                out.results
            );
        }
        let mut s: Vec<u32> = ids(out);
        s.sort();
        s.dedup();
        assert_eq!(s.len(), out.results.len(), "duplicate ids");
    }

    fn line(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
        let points = (0..n).map(|i| vec![i as f32]).collect();
        let adj = (0..n)
            .map(|i| {
                let mut v = Vec::new();
                if i > 0 {
                    v.push((i - 1) as u32);
                }
                if i + 1 < n {
                    v.push((i + 1) as u32);
                }
                v
            })
            .collect();
        (points, adj)
    }

    /// Deterministic LCG so tests need no external crates.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    #[test]
    fn candidate_ordering_distance_then_id() {
        let mut v = [
            Candidate::new(5, 1.0),
            Candidate::new(2, 0.5),
            Candidate::new(1, 1.0),
            Candidate::new(9, 0.0),
        ];
        v.sort();
        let got: Vec<u32> = v.iter().map(|c| c.id).collect();
        assert_eq!(got, vec![9, 2, 1, 5]);
        assert!(Candidate::new(1, 1.0) < Candidate::new(2, 1.0));
        assert_eq!(Candidate::new(3, 1.0), Candidate::new(3, 1.0));
    }

    #[test]
    fn candidate_ordering_is_total_with_nan() {
        // Sorting must never panic even if a NaN slipped in.
        let mut v = [
            Candidate::new(0, f32::NAN),
            Candidate::new(1, 1.0),
            Candidate::new(2, f32::INFINITY),
        ];
        v.sort();
        assert_eq!(v[0].id, 1);
    }

    #[test]
    fn non_finite_distance_is_an_error() {
        let points = [vec![0.0], vec![1.0]];
        let adj = [vec![1], vec![0]];
        let r = vamana_search(
            &[0],
            SearchParams { k: 1, ef_search: 2 },
            2,
            |id| {
                Ok(if id == 1 {
                    f32::NAN
                } else {
                    l2_distance(&points[id as usize], &[0.0])
                })
            },
            |id, out| {
                out.extend_from_slice(&adj[id as usize]);
                Ok(())
            },
        );
        assert_eq!(r.unwrap_err(), SearchError::NonFiniteDistance { id: 1 });
    }

    #[test]
    fn single_node_with_empty_neighbors() {
        let points = vec![vec![1.0, 1.0]];
        let adj = vec![vec![]];
        let out = search(&points, &adj, 0, &[0.0, 0.0], 3, 8).unwrap();
        assert_eq!(ids(&out), vec![0]);
        assert_eq!(out.stats.visited_nodes, 1);
        assert_eq!(out.stats.expanded_nodes, 1);
    }

    #[test]
    fn empty_graph_returns_empty() {
        let out = vamana_search(
            &[0],
            SearchParams { k: 1, ef_search: 1 },
            0,
            |_| Ok(0.0),
            |_, _| Ok(()),
        )
        .unwrap();
        assert!(out.results.is_empty());
    }

    #[test]
    fn entry_out_of_range_is_an_error() {
        let points = vec![vec![0.0]];
        let adj = vec![vec![]];
        let e = search(&points, &adj, 3, &[0.0], 1, 1).unwrap_err();
        assert_eq!(
            e,
            SearchError::EntryOutOfRange {
                entry: 3,
                num_nodes: 1
            }
        );
    }

    #[test]
    fn linear_graph_greedy_walk_with_ef_one() {
        let (points, adj) = line(10);
        let out = search(&points, &adj, 0, &[8.9], 1, 1).unwrap();
        assert_eq!(ids(&out), vec![9]);
    }

    #[test]
    fn linear_graph_top_k() {
        let (points, adj) = line(10);
        let out = search(&points, &adj, 0, &[4.2], 3, 3).unwrap();
        assert_eq!(ids(&out), vec![4, 5, 3]);
        assert_sorted_unique(&out);
    }

    #[test]
    fn cyclic_graph_terminates_and_visits_each_node_once() {
        let n = 12;
        let points: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                vec![a.cos(), a.sin()]
            })
            .collect();
        // Ring plus back-edges: every node points to both neighbours and itself.
        let adj: Vec<Vec<u32>> = (0..n)
            .map(|i| vec![((i + 1) % n) as u32, ((i + n - 1) % n) as u32, i as u32])
            .collect();
        let out = search(&points, &adj, 0, &points[6], 2, n).unwrap();
        assert_eq!(out.results[0].id, 6);
        assert!(out.stats.visited_nodes <= n);
        assert_eq!(out.stats.visited_nodes, out.stats.distance_computations);
        assert_sorted_unique(&out);
    }

    #[test]
    fn disconnected_graph_only_returns_reachable_nodes() {
        let points = vec![vec![0.0], vec![1.0], vec![100.0], vec![101.0]];
        let adj = vec![vec![1], vec![0], vec![3], vec![2]];
        let out = search(&points, &adj, 0, &[100.0], 4, 4).unwrap();
        assert_eq!(ids(&out), vec![1, 0]);
    }

    #[test]
    fn duplicate_neighbors_evaluated_once() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let adj = vec![vec![1, 1, 1, 2, 2], vec![0, 0], vec![0]];
        let out = search(&points, &adj, 0, &[0.0], 3, 3).unwrap();
        assert_eq!(out.stats.distance_computations, 3);
        assert_eq!(ids(&out), vec![0, 1, 2]);
    }

    #[test]
    fn k_one_and_k_larger_than_graph() {
        let (points, adj) = line(3);
        let one = search(&points, &adj, 0, &[2.0], 1, 4).unwrap();
        assert_eq!(ids(&one), vec![2]);
        let all = search(&points, &adj, 0, &[2.0], 5, 5).unwrap();
        assert_eq!(ids(&all), vec![2, 1, 0]);
    }

    #[test]
    fn ef_equal_to_k() {
        let (points, adj) = line(20);
        let out = search(&points, &adj, 0, &[10.0], 4, 4).unwrap();
        assert_eq!(out.results.len(), 4);
        assert_eq!(out.results[0].id, 10);
        assert_sorted_unique(&out);
    }

    #[test]
    fn invalid_parameters_rejected() {
        let (points, adj) = line(3);
        assert!(matches!(
            search(&points, &adj, 0, &[0.0], 0, 4),
            Err(SearchError::InvalidParams(_))
        ));
        assert!(matches!(
            search(&points, &adj, 0, &[0.0], 5, 4),
            Err(SearchError::InvalidParams(_))
        ));
        let cfg = BeamSearchConfig {
            ef: 0,
            num_nodes: 3,
            collect_expanded: false,
        };
        assert!(matches!(
            beam_search(&[0], &cfg, |_| Ok(0.0), |_, _| Ok(())),
            Err(SearchError::InvalidParams(_))
        ));
    }

    #[test]
    fn equal_distances_tie_break_by_lower_id_and_are_stable() {
        // Five points at distance 1 from the origin query, plus the centre.
        let points = vec![
            vec![0.5, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, -1.0],
            vec![-1.0, 0.0],
        ];
        // Star graph through node 0, neighbours listed in a scrambled order.
        let adj = vec![vec![4, 2, 3, 1], vec![0], vec![0], vec![0], vec![0]];
        let q = [0.0, 0.0];
        let first = search(&points, &adj, 0, &q, 5, 5).unwrap();
        assert_eq!(ids(&first), vec![0, 1, 2, 3, 4]);
        for _ in 0..10 {
            let again = search(&points, &adj, 0, &q, 5, 5).unwrap();
            assert_eq!(ids(&again), ids(&first));
        }
        let top2 = search(&points, &adj, 0, &q, 3, 5).unwrap();
        assert_eq!(ids(&top2), vec![0, 1, 2]);
    }

    #[test]
    fn out_of_range_neighbors_are_skipped_not_followed() {
        let points = vec![vec![0.0], vec![1.0]];
        let adj = vec![vec![1, 7, u32::MAX], vec![0]];
        let out = search(&points, &adj, 0, &[1.0], 2, 2).unwrap();
        assert_eq!(ids(&out), vec![1, 0]);
        assert_eq!(out.stats.skipped_invalid_edges, 2);
    }

    #[test]
    fn complete_graph_with_full_beam_matches_brute_force() {
        let mut rng = Lcg(7);
        let n = 60;
        let points: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..4).map(|_| rng.next_f32()).collect())
            .collect();
        let adj: Vec<Vec<u32>> = (0..n)
            .map(|i| (0..n as u32).filter(|&j| j != i as u32).collect())
            .collect();
        for qi in 0..10 {
            let q: Vec<f32> = (0..4).map(|_| rng.next_f32()).collect();
            let out = search(&points, &adj, (qi % n) as u32, &q, 10, n).unwrap();
            let mut exact: Vec<Candidate> = (0..n)
                .map(|i| Candidate::new(i as u32, l2_distance(&points[i], &q)))
                .collect();
            exact.sort();
            exact.truncate(10);
            assert_eq!(out.results, exact);
        }
    }

    #[test]
    fn collect_expanded_records_expansion_order() {
        let (points, adj) = line(5);
        let cfg = BeamSearchConfig {
            ef: 2,
            num_nodes: 5,
            collect_expanded: true,
        };
        let out = beam_search(
            &[0],
            &cfg,
            |id| Ok(l2_distance(&points[id as usize], &[4.0])),
            |id, o| {
                o.extend_from_slice(&adj[id as usize]);
                Ok(())
            },
        )
        .unwrap();
        let exp: Vec<u32> = out.expanded.iter().map(|c| c.id).collect();
        assert_eq!(exp, vec![0, 1, 2, 3, 4]);
        assert_eq!(out.stats.expanded_nodes, 5);
    }

    #[test]
    fn callback_errors_propagate() {
        let r = beam_search(
            &[0],
            &BeamSearchConfig {
                ef: 2,
                num_nodes: 2,
                collect_expanded: false,
            },
            |_| Ok(1.0),
            |id, _| {
                Err(SearchError::Access {
                    id,
                    message: "boom".into(),
                })
            },
        );
        assert!(matches!(r, Err(SearchError::Access { id: 0, .. })));
    }

    #[test]
    fn multiple_entry_points_reach_disconnected_components() {
        let points = [vec![0.0], vec![1.0], vec![100.0], vec![101.0]];
        let adj = [vec![1], vec![0], vec![3], vec![2]];
        let run = |entries: &[u32]| {
            vamana_search(
                entries,
                SearchParams { k: 2, ef_search: 4 },
                4,
                |id| Ok(l2_distance(&points[id as usize], &[100.0])),
                |id, out| {
                    out.extend_from_slice(&adj[id as usize]);
                    Ok(())
                },
            )
            .unwrap()
        };
        // From node 0 alone the far component is unreachable.
        assert_eq!(ids(&run(&[0])), vec![1, 0]);
        // With an entry point in each component the true neighbours are found;
        // duplicate entries are evaluated once.
        let both = run(&[0, 2, 2, 0]);
        assert_eq!(ids(&both), vec![2, 3]);
        assert_eq!(both.stats.visited_nodes, 4);
    }

    #[test]
    fn entry_point_list_is_validated() {
        let cfg = BeamSearchConfig {
            ef: 2,
            num_nodes: 3,
            collect_expanded: false,
        };
        assert!(matches!(
            beam_search(&[], &cfg, |_| Ok(0.0), |_, _| Ok(())),
            Err(SearchError::InvalidParams(_))
        ));
        assert_eq!(
            beam_search(&[0, 5], &cfg, |_| Ok(0.0), |_, _| Ok(())).unwrap_err(),
            SearchError::EntryOutOfRange {
                entry: 5,
                num_nodes: 3
            }
        );
    }

    fn prune_l2(points: &[Vec<f32>], node: u32, ids: &[u32], r: usize, alpha: f32) -> Vec<u32> {
        let cands: Vec<Candidate> = ids
            .iter()
            .map(|&i| Candidate::new(i, l2_distance(&points[node as usize], &points[i as usize])))
            .collect();
        robust_prune(node, &cands, r, alpha, |a, b| {
            Ok(l2_distance(&points[a as usize], &points[b as usize]))
        })
        .unwrap()
    }

    #[test]
    fn prune_removes_occluded_collinear_points() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(prune_l2(&points, 0, &[3, 2, 1], 3, 1.0), vec![1]);
    }

    #[test]
    fn prune_keeps_diverse_directions() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ];
        assert_eq!(prune_l2(&points, 0, &[1, 2, 3], 3, 1.2), vec![1, 2, 3]);
    }

    #[test]
    fn prune_excludes_self_and_duplicates_and_respects_r() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ];
        let out = prune_l2(&points, 0, &[0, 1, 1, 2, 2, 3, 0], 2, 1.2);
        assert_eq!(out, vec![1, 2]);
        assert!(!out.contains(&0));
    }

    #[test]
    fn prune_with_larger_alpha_keeps_more_edges() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        // alpha = 2.5: d(1,2)=1 -> 2.5 > d(0,2)=2, kept; d(1,3)=2 -> 5 > 3 kept,
        // d(2,3)=1 -> 2.5 <= 3 pruned.
        assert_eq!(prune_l2(&points, 0, &[1, 2, 3], 3, 2.5), vec![1, 2]);
    }

    #[test]
    fn prune_rejects_invalid_parameters() {
        let c = [Candidate::new(1, 1.0)];
        assert!(robust_prune(0, &c, 0, 1.2, |_, _| Ok(1.0)).is_err());
        assert!(robust_prune(0, &c, 2, f32::NAN, |_, _| Ok(1.0)).is_err());
        assert!(robust_prune(0, &c, 2, 0.0, |_, _| Ok(1.0)).is_err());
    }
}
