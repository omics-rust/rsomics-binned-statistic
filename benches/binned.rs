use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_binned_statistic::{Statistic, binned_statistic};
use std::hint::black_box;

/// Deterministic LCG so the bench fixture is reproducible without rand.
fn sample(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f64> = (0..n).map(|_| next() * 100.0).collect();
    let v: Vec<f64> = (0..n).map(|_| next() * 1000.0 - 500.0).collect();
    (x, v)
}

fn bench(c: &mut Criterion) {
    let (x, v) = sample(1_000_000);
    let range = Some((0.0, 100.0));
    for stat in [
        Statistic::Mean,
        Statistic::Std,
        Statistic::Sum,
        Statistic::Count,
        Statistic::Median,
        Statistic::Min,
        Statistic::Max,
    ] {
        c.bench_function(&format!("{}_1e6", stat.name()), |b| {
            b.iter(|| binned_statistic(black_box(&x), black_box(&v), stat, 50, range))
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
