//! Value-exact compatibility with `scipy.stats.binned_statistic` 1.17.1.
//!
//! Golden values were produced once by running scipy on the committed fixtures
//! and on deterministic LCG-generated vectors (n=5000, n=1e6); this test
//! re-derives them with no scipy present. The 1e6 case exercises the per-bin
//! sequential accumulation that must match numpy's `np.bincount` to the last
//! bit. Goldens are IEEE-754 hex bit patterns because serde_json's float parser
//! is not correctly-rounded and would otherwise mask a true bit-exact match.

use rsomics_binned_statistic::{Statistic, binned_statistic, io::parse_table};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    dataset: String,
    stat: String,
    bins: usize,
    range: Option<Vec<f64>>,
    statistic_bits: Vec<String>,
    edges_bits: Vec<String>,
    binnumber: Option<Vec<usize>>,
    binnumber_sum: Option<u64>,
}

#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

fn from_bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

/// Deterministic LCG matching the Python golden generator bit-for-bit.
fn lcg_vec(n: usize, seed: u64, scale: f64, off: f64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            u * scale + off
        })
        .collect()
}

const SEED_X: u64 = 0x2545_F491_4F6C_DD1D;
const SEED_V: u64 = 0x1234_5678_9ABC_DEF0;

fn dataset(name: &str) -> (Vec<f64>, Vec<f64>) {
    match name {
        "small.tsv" => parse_table(include_bytes!("golden/small.tsv")).unwrap(),
        "empty.tsv" => parse_table(include_bytes!("golden/empty.tsv")).unwrap(),
        "lcg5000" => (
            lcg_vec(5000, SEED_X, 100.0, 0.0),
            lcg_vec(5000, SEED_V, 1000.0, -500.0),
        ),
        "lcg1000000" => (
            lcg_vec(1_000_000, SEED_X, 100.0, 0.0),
            lcg_vec(1_000_000, SEED_V, 1000.0, -500.0),
        ),
        other => panic!("unknown dataset {other}"),
    }
}

fn parse_stat(s: &str) -> Statistic {
    match s {
        "mean" => Statistic::Mean,
        "std" => Statistic::Std,
        "median" => Statistic::Median,
        "count" => Statistic::Count,
        "sum" => Statistic::Sum,
        "min" => Statistic::Min,
        "max" => Statistic::Max,
        other => panic!("unknown stat {other}"),
    }
}

fn ulp(a: f64, b: f64) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a == b {
        return 0;
    }
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    let ia = if ia < 0 { i64::MIN - ia } else { ia };
    let ib = if ib < 0 { i64::MIN - ib } else { ib };
    ia.abs_diff(ib)
}

#[test]
fn matches_scipy_golden() {
    let golden: Golden = serde_json::from_slice(include_bytes!("golden/expected.json")).unwrap();

    // sqrt-derived std may differ by up to 1 ULP across architectures; every
    // other statistic and all edges are required bit-exact (0 ULP).
    let std_tol = 1u64;

    let mut worst_stat = 0u64;
    let mut worst_edge = 0u64;
    for c in &golden.cases {
        let (x, v) = dataset(&c.dataset);
        let stat = parse_stat(&c.stat);
        let range = c.range.as_ref().map(|r| (r[0], r[1]));
        let got = binned_statistic(&x, &v, stat, c.bins, range);

        assert_eq!(
            got.statistic.len(),
            c.statistic_bits.len(),
            "{} {} bins={}: statistic length mismatch",
            c.dataset,
            c.stat,
            c.bins
        );
        let tol = if stat == Statistic::Std { std_tol } else { 0 };
        for (i, (g, want_bits)) in got.statistic.iter().zip(&c.statistic_bits).enumerate() {
            let want = from_bits(want_bits);
            let u = ulp(*g, want);
            worst_stat = worst_stat.max(u);
            assert!(
                u <= tol,
                "{} {} bins={} range={:?} bin[{i}]: got {g} want {want}, {u} ULP",
                c.dataset,
                c.stat,
                c.bins,
                c.range
            );
        }

        for (i, (g, want_bits)) in got.bin_edges.iter().zip(&c.edges_bits).enumerate() {
            let want = from_bits(want_bits);
            let u = ulp(*g, want);
            worst_edge = worst_edge.max(u);
            assert_eq!(
                u, 0,
                "{} bins={} range={:?} edge[{i}]: got {g} want {want}, {u} ULP",
                c.dataset, c.bins, c.range
            );
        }

        if let Some(want_bn) = &c.binnumber {
            assert_eq!(
                &got.binnumber, want_bn,
                "{} bins={} range={:?}: binnumber mismatch",
                c.dataset, c.bins, c.range
            );
        }
        if let Some(want_sum) = c.binnumber_sum {
            let sum: u64 = got.binnumber.iter().map(|&b| b as u64).sum();
            assert_eq!(
                sum, want_sum,
                "{} bins={} range={:?}: binnumber checksum mismatch",
                c.dataset, c.bins, c.range
            );
        }
    }
    assert_eq!(
        worst_edge, 0,
        "edges must be bit-exact, worst {worst_edge} ULP"
    );
    assert!(
        worst_stat <= std_tol,
        "worst statistic {worst_stat} ULP exceeds std tolerance"
    );
    eprintln!("worst stat ULP = {worst_stat} (std tol {std_tol}), worst edge ULP = {worst_edge}");
}
