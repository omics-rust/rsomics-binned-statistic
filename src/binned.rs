//! Per-bin statistics, value-exact to `scipy.stats.binned_statistic` (scipy
//! 1.17.1).
//!
//! scipy reduces each statistic through `np.bincount` (mean/std/sum/count) or a
//! per-bin sort (median/min/max). `np.bincount` accumulates in input order with
//! an ordinary sequential add — not numpy's pairwise tree — so a single
//! in-order pass over the values reproduces it to the last bit. Empty bins are
//! NaN for mean/std/median/min/max and 0 for count/sum, matching scipy.

use std::cmp::Ordering;

use crate::edges::{bin_edges, bin_numbers, data_range};

/// numpy's sort order: every NaN sorts after every real number regardless of its
/// sign bit, matching `np.argsort` / `np.lexsort`. This is what lets the per-bin
/// median/min/max reductions agree with scipy on a bin that contains a NaN —
/// scipy sorts the values and picks positionally, so a NaN in the bin lands at
/// the high end rather than deciding the result. `total_cmp` alone would order a
/// negative NaN below `-inf`; numpy keeps all NaN together at the top.
fn numpy_cmp(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.total_cmp(&b),
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

/// The statistic to compute per bin.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Statistic {
    Mean,
    Std,
    Median,
    Count,
    Sum,
    Min,
    Max,
}

impl Statistic {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Statistic::Mean => "mean",
            Statistic::Std => "std",
            Statistic::Median => "median",
            Statistic::Count => "count",
            Statistic::Sum => "sum",
            Statistic::Min => "min",
            Statistic::Max => "max",
        }
    }
}

/// The full result of one `binned_statistic` call.
pub struct Binned {
    /// One value per real bin (`nbins` entries; outlier bins are stripped).
    pub statistic: Vec<f64>,
    /// The `nbins + 1` bin edges.
    pub bin_edges: Vec<f64>,
    /// The 1-based bin number for each input value (0 = below, `nbins+1` = above).
    pub binnumber: Vec<usize>,
}

/// Compute a per-bin statistic of `values`, binned by `x` into `nbins` bins.
///
/// `range` is `(lo, hi)`; when `None` the data range `(x.min(), x.max())` is
/// used. `values` is ignored for `count`. Both slices must be the same length
/// (caller-enforced) except when `stat == Count`.
pub fn binned_statistic(
    x: &[f64],
    values: &[f64],
    stat: Statistic,
    nbins: usize,
    range: Option<(f64, f64)>,
) -> Binned {
    let (smin, smax) = range.unwrap_or_else(|| data_range(x));
    let edges = bin_edges(smin, smax, nbins);
    let binnumber = bin_numbers(x, &edges);

    // Flat layout includes the two outlier bins: index 0 and nbins+1.
    let flat = nbins + 2;
    let statistic = match stat {
        Statistic::Count => count(&binnumber, flat, nbins),
        Statistic::Sum => sum(values, &binnumber, flat, nbins),
        Statistic::Mean => mean(values, &binnumber, flat, nbins),
        Statistic::Std => std(values, &binnumber, flat, nbins),
        Statistic::Median => median(values, &binnumber, nbins),
        Statistic::Min => extreme(values, &binnumber, nbins, Extreme::Min),
        Statistic::Max => extreme(values, &binnumber, nbins, Extreme::Max),
    };

    Binned {
        statistic,
        bin_edges: edges,
        binnumber,
    }
}

/// `np.bincount(binnumber)` — sequential per-bin counts over the flat layout.
fn bincount(binnumber: &[usize], flat: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; flat];
    for &b in binnumber {
        c[b] += 1.0;
    }
    c
}

/// `np.bincount(binnumber, weights)` — sequential per-bin weighted sums.
fn bincount_weighted(values: &[f64], binnumber: &[usize], flat: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; flat];
    for (&b, &v) in binnumber.iter().zip(values) {
        s[b] += v;
    }
    s
}

/// Strip the two outlier bins (flat indices 0 and `nbins+1`).
fn core(flat_result: &[f64], nbins: usize) -> Vec<f64> {
    flat_result[1..=nbins].to_vec()
}

fn count(binnumber: &[usize], flat: usize, nbins: usize) -> Vec<f64> {
    core(&bincount(binnumber, flat), nbins)
}

fn sum(values: &[f64], binnumber: &[usize], flat: usize, nbins: usize) -> Vec<f64> {
    core(&bincount_weighted(values, binnumber, flat), nbins)
}

fn mean(values: &[f64], binnumber: &[usize], flat: usize, nbins: usize) -> Vec<f64> {
    let cnt = bincount(binnumber, flat);
    let s = bincount_weighted(values, binnumber, flat);
    let mut out = vec![f64::NAN; flat];
    for b in 0..flat {
        if cnt[b] != 0.0 {
            out[b] = s[b] / cnt[b];
        }
    }
    core(&out, nbins)
}

/// scipy's population std: per-element `delta = v - binmean`, then
/// `sqrt(bincount(delta^2) / count)`.
fn std(values: &[f64], binnumber: &[usize], flat: usize, nbins: usize) -> Vec<f64> {
    let cnt = bincount(binnumber, flat);
    let s = bincount_weighted(values, binnumber, flat);
    let deltas: Vec<f64> = binnumber
        .iter()
        .zip(values)
        .map(|(&b, &v)| {
            let d = v - s[b] / cnt[b];
            d * d
        })
        .collect();
    let sq = bincount_weighted(&deltas, binnumber, flat);
    let mut out = vec![f64::NAN; flat];
    for b in 0..flat {
        if cnt[b] != 0.0 {
            out[b] = (sq[b] / cnt[b]).sqrt();
        }
    }
    core(&out, nbins)
}

/// scipy's median: lexsort by (bin, value), then per bin take the mean of the
/// floor/ceil of `(count-1)/2` — `np.median`'s two-middle average for even n.
fn median(values: &[f64], binnumber: &[usize], nbins: usize) -> Vec<f64> {
    let flat = nbins + 2;
    let mut order: Vec<usize> = (0..values.len()).collect();
    // Stable sort by bin then value, matching np.lexsort((values, binnumber)).
    order.sort_by(|&a, &b| {
        binnumber[a]
            .cmp(&binnumber[b])
            .then_with(|| numpy_cmp(values[a], values[b]))
    });

    let mut out = vec![f64::NAN; flat];
    let mut i = 0;
    while i < order.len() {
        let bin = binnumber[order[i]];
        let start = i;
        while i < order.len() && binnumber[order[i]] == bin {
            i += 1;
        }
        let n = i - start;
        let mid = (n - 1) as f64 / 2.0;
        let lo = mid.floor() as usize;
        let hi = mid.ceil() as usize;
        let a = values[order[start + lo]];
        let b = values[order[start + hi]];
        out[bin] = (a + b) / 2.0;
    }
    core(&out, nbins)
}

enum Extreme {
    Min,
    Max,
}

/// scipy's min/max: argsort `values` (reversed for min), scatter into bins so
/// the last write per bin is the extreme; empty bins stay NaN.
fn extreme(values: &[f64], binnumber: &[usize], nbins: usize, which: Extreme) -> Vec<f64> {
    let flat = nbins + 2;
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| numpy_cmp(values[a], values[b]));
    if let Extreme::Min = which {
        order.reverse();
    }
    let mut out = vec![f64::NAN; flat];
    for &idx in &order {
        out[binnumber[idx]] = values[idx];
    }
    core(&out, nbins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_doc_example() {
        let x = [1.0, 1.0, 2.0, 5.0, 7.0];
        let v = [1.0, 1.0, 2.0, 1.5, 3.0];
        let r = binned_statistic(&x, &v, Statistic::Sum, 2, None);
        assert_eq!(r.statistic, vec![4.0, 4.5]);
        assert_eq!(r.bin_edges, vec![1.0, 4.0, 7.0]);
        assert_eq!(r.binnumber, vec![1, 1, 1, 2, 2]);
    }

    #[test]
    fn mean_doc_example() {
        let x = [1.0, 2.0, 1.0, 2.0, 4.0];
        let v = [0.0, 1.0, 2.0, 3.0, 4.0];
        let r = binned_statistic(&x, &v, Statistic::Mean, 3, None);
        assert_eq!(r.statistic, vec![1.0, 2.0, 4.0]);
        assert_eq!(r.binnumber, vec![1, 2, 1, 2, 3]);
    }

    #[test]
    fn empty_bins_are_nan_for_mean_and_zero_for_count() {
        let x = [0.0, 0.1, 9.9, 10.0];
        let v = [1.0, 2.0, 3.0, 4.0];
        let m = binned_statistic(&x, &v, Statistic::Mean, 5, Some((0.0, 10.0)));
        assert_eq!(m.statistic[0], 1.5);
        assert!(m.statistic[1].is_nan());
        assert!(m.statistic[2].is_nan());
        assert!(m.statistic[3].is_nan());
        assert_eq!(m.statistic[4], 3.5);

        let c = binned_statistic(&x, &v, Statistic::Count, 5, Some((0.0, 10.0)));
        assert_eq!(c.statistic, vec![2.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn std_is_population() {
        let x = [0.0, 0.1, 9.9, 10.0];
        let v = [1.0, 2.0, 3.0, 4.0];
        let r = binned_statistic(&x, &v, Statistic::Std, 5, Some((0.0, 10.0)));
        assert_eq!(r.statistic[0], 0.5);
        assert_eq!(r.statistic[4], 0.5);
    }

    #[test]
    fn median_even_count_averages_two_middles() {
        let x = [0.0, 0.0, 0.0, 0.0];
        let v = [1.0, 2.0, 3.0, 4.0];
        let r = binned_statistic(&x, &v, Statistic::Median, 1, Some((0.0, 1.0)));
        assert_eq!(r.statistic, vec![2.5]);
    }

    #[test]
    fn min_max_per_bin() {
        let x = [0.0, 0.0, 9.0, 9.0];
        let v = [5.0, 1.0, 7.0, 2.0];
        let mn = binned_statistic(&x, &v, Statistic::Min, 2, Some((0.0, 10.0)));
        assert_eq!(mn.statistic, vec![1.0, 2.0]);
        let mx = binned_statistic(&x, &v, Statistic::Max, 2, Some((0.0, 10.0)));
        assert_eq!(mx.statistic, vec![5.0, 7.0]);
    }

    /// A NaN inside a bin must not panic; scipy sorts it to the high end, so
    /// median/min pick the surviving values while mean/std/sum/max propagate NaN.
    /// Oracle: `binned_statistic([1,1,1,2],[10,nan,30,5], stat, bins=2)`,
    /// scipy 1.17.1 — bin1 = {10, nan, 30}, bin2 = {5}.
    #[test]
    fn nan_in_bin_matches_scipy() {
        let x = [1.0, 1.0, 1.0, 2.0];
        let v = [10.0, f64::NAN, 30.0, 5.0];

        let median = binned_statistic(&x, &v, Statistic::Median, 2, None);
        assert_eq!(median.statistic, vec![30.0, 5.0]);

        let min = binned_statistic(&x, &v, Statistic::Min, 2, None);
        assert_eq!(min.statistic, vec![10.0, 5.0]);

        let max = binned_statistic(&x, &v, Statistic::Max, 2, None);
        assert!(max.statistic[0].is_nan());
        assert_eq!(max.statistic[1], 5.0);

        let count = binned_statistic(&x, &v, Statistic::Count, 2, None);
        assert_eq!(count.statistic, vec![3.0, 1.0]);

        // A NaN reaches every additive reduction, so bin1 is NaN; bin2 is the
        // single value 5 (std of one element is 0). Oracle bin2: mean/sum 5, std 0.
        for (stat, bin2) in [
            (Statistic::Mean, 5.0),
            (Statistic::Sum, 5.0),
            (Statistic::Std, 0.0),
        ] {
            let r = binned_statistic(&x, &v, stat, 2, None);
            assert!(
                r.statistic[0].is_nan(),
                "{} bin1 should be NaN",
                stat.name()
            );
            assert_eq!(r.statistic[1], bin2, "{} bin2", stat.name());
        }
    }

    /// When a NaN lands on a median middle slot the average is NaN, and `max`
    /// stays NaN whenever the bin holds a NaN. Oracle bin1 = {5, nan} (even),
    /// bin2 = {1, 2, 3, nan}.
    #[test]
    fn nan_at_median_middle_matches_scipy() {
        let x = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let v = [5.0, f64::NAN, 1.0, 2.0, 3.0, f64::NAN];

        let median = binned_statistic(&x, &v, Statistic::Median, 2, None);
        assert!(median.statistic[0].is_nan());
        assert_eq!(median.statistic[1], 2.5);

        let min = binned_statistic(&x, &v, Statistic::Min, 2, None);
        assert_eq!(min.statistic, vec![5.0, 1.0]);

        let max = binned_statistic(&x, &v, Statistic::Max, 2, None);
        assert!(max.statistic[0].is_nan());
        assert!(max.statistic[1].is_nan());
    }

    /// Empty bins: NaN for mean/std/median/min/max, 0 for count/sum.
    #[test]
    fn empty_bins_match_scipy_for_every_statistic() {
        let x = [0.0, 10.0];
        let v = [1.0, 2.0];
        for stat in [
            Statistic::Mean,
            Statistic::Std,
            Statistic::Median,
            Statistic::Min,
            Statistic::Max,
        ] {
            let r = binned_statistic(&x, &v, stat, 3, Some((0.0, 10.0)));
            assert!(r.statistic[1].is_nan(), "{} middle bin", stat.name());
        }
        let count = binned_statistic(&x, &v, Statistic::Count, 3, Some((0.0, 10.0)));
        assert_eq!(count.statistic[1], 0.0);
        let sum = binned_statistic(&x, &v, Statistic::Sum, 3, Some((0.0, 10.0)));
        assert_eq!(sum.statistic[1], 0.0);
    }
}
