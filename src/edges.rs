//! Bin edges and per-value bin numbers, value-exact to scipy's
//! `_binned_statistic._bin_edges` / `_bin_numbers` (numpy `linspace` +
//! `digitize`), scipy 1.17.1.
//!
//! scipy builds `edges = linspace(smin, smax, nbins + 1)` (the 1D path of
//! `_bin_edges`, where `nbin = nbins + 2` counts the two outlier bins). The
//! per-value bin number is `np.digitize(x, edges)` — 0 below the first edge,
//! `nbins + 1` above the last — with one correction: a value sitting on the
//! rightmost edge (to scipy's rounding precision) is pulled into the last real
//! bin instead of the upper outlier bin.

/// numpy's `linspace(start, stop, num, endpoint=True)`, scalar float path,
/// reproduced bit-for-bit so every edge comparison matches scipy's.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; num];
    let div = num as f64 - 1.0;
    let delta = stop - start;
    if div > 0.0 {
        let step = delta / div;
        if step == 0.0 {
            for (i, e) in y.iter_mut().enumerate() {
                *e = (i as f64 / div) * delta;
            }
        } else {
            for (i, e) in y.iter_mut().enumerate() {
                *e = i as f64 * step;
            }
        }
    } else {
        for (i, e) in y.iter_mut().enumerate() {
            *e = i as f64 * delta;
        }
    }
    for e in &mut y {
        *e += start;
    }
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// `_bin_edges`'s data range: `(x.min(), x.max())`, with scipy's degenerate-range
/// widening (`smin -= 0.5; smax += 0.5`) when every value is identical.
pub fn data_range(x: &[f64]) -> (f64, f64) {
    let mut smin = x[0];
    let mut smax = x[0];
    for &v in &x[1..] {
        if v < smin {
            smin = v;
        }
        if v > smax {
            smax = v;
        }
    }
    if smin == smax {
        smin -= 0.5;
        smax += 0.5;
    }
    (smin, smax)
}

/// The `nbins + 1` edges over `[smin, smax]`, plus scipy's degenerate widening
/// applied to an explicit equal-bounds range.
pub fn bin_edges(smin: f64, smax: f64, nbins: usize) -> Vec<f64> {
    let (lo, hi) = if smin == smax {
        (smin - 0.5, smax + 0.5)
    } else {
        (smin, smax)
    };
    linspace(lo, hi, nbins + 1)
}

/// scipy's rounding precision for the rightmost-edge correction:
/// `int(-log10(min_edge_spacing)) + 6`.
fn edge_decimals(edges: &[f64]) -> i32 {
    let mut dmin = f64::INFINITY;
    for w in edges.windows(2) {
        let d = w[1] - w[0];
        if d < dmin {
            dmin = d;
        }
    }
    (-dmin.log10()) as i32 + 6
}

/// numpy's `np.around(value, decimals)` (round-half-to-even) for the precision
/// the edge correction needs.
fn round_to(value: f64, decimals: i32) -> f64 {
    let scale = 10f64.powi(decimals);
    let scaled = value * scale;
    let rounded = scaled.round_ties_even();
    rounded / scale
}

/// `np.digitize(v, edges)` with `right=False`: the index `i` such that
/// `edges[i-1] <= v < edges[i]`, where edges are ascending. Returns 0 for
/// `v < edges[0]` and `edges.len()` for `v >= edges[last]`.
fn digitize(v: f64, edges: &[f64]) -> usize {
    edges.partition_point(|&e| e <= v)
}

/// The 1-based bin number for one value, matching scipy's `_bin_numbers`:
/// digitize, then pull a value on the rightmost edge into the last real bin.
///
/// 0 = below the range, `1..=nbins` = real bins, `nbins + 1` = above the range.
pub fn bin_number(v: f64, edges: &[f64], decimals: i32) -> usize {
    let mut b = digitize(v, edges);
    let last = *edges.last().unwrap();
    if v >= last && round_to(v, decimals) == round_to(last, decimals) {
        b -= 1;
    }
    b
}

/// Per-value bin numbers for the whole sample, sharing the edge-rounding
/// precision across the batch.
pub fn bin_numbers(x: &[f64], edges: &[f64]) -> Vec<usize> {
    let decimals = edge_decimals(edges);
    x.iter().map(|&v| bin_number(v, edges, decimals)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_integer_edges() {
        assert_eq!(
            linspace(0.0, 10.0, 11),
            (0..=10).map(f64::from).collect::<Vec<_>>()
        );
    }

    #[test]
    fn digitize_matches_numpy_semantics() {
        let edges = [0.0, 5.0, 10.0];
        let got: Vec<usize> = [-1.0, 0.0, 2.5, 5.0, 7.5, 10.0, 11.0]
            .iter()
            .map(|&v| digitize(v, &edges))
            .collect();
        assert_eq!(got, vec![0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn upper_edge_pulled_into_last_bin() {
        let edges = bin_edges(0.0, 10.0, 2);
        let decimals = edge_decimals(&edges);
        assert_eq!(bin_number(10.0, &edges, decimals), 2);
    }

    #[test]
    fn outliers_get_zero_and_top() {
        let edges = bin_edges(0.0, 10.0, 2);
        let nums = bin_numbers(&[-1.0, 0.0, 5.0, 11.0], &edges);
        assert_eq!(nums, vec![0, 1, 2, 3]);
    }

    #[test]
    fn degenerate_range_widens() {
        let (lo, hi) = data_range(&[3.0, 3.0, 3.0]);
        assert_eq!((lo, hi), (2.5, 3.5));
    }
}
