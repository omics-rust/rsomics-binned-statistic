# rsomics-binned-statistic

Per-bin statistic of a second variable — bin `x` into equal-width bins and
compute the mean, standard deviation, median, count, sum, min, or max of the
associated `values` within each bin. A value-exact Rust port of
`scipy.stats.binned_statistic` (scipy 1.17.1).

```text
rsomics-binned-statistic <data.tsv> --statistic mean   --bins 10
rsomics-binned-statistic <data.tsv> --statistic median --bins 50 --range 0,100
rsomics-binned-statistic x.tsv --values v.tsv --statistic std --bins 20
```

Input is a two-column `x<TAB>values` table (`-` reads stdin), or `x` from `DATA`
with `values` from a separate `--values FILE`. The output is `bin_index<TAB>statistic`
per line with the bin edges as a header (or the full result — statistic, bin
edges, and per-value bin numbers — in the `--json` envelope; `--binnumber` adds
the bin numbers to the text output).

## Statistics

- `--statistic mean` — mean of `values` per bin; empty bins are `NaN`.
- `--statistic std` — population standard deviation (`ddof=0`) per bin; empty
  bins `NaN`.
- `--statistic median` — median per bin (mean of the two middles for even
  counts, as `np.median`); empty bins `NaN`.
- `--statistic count` — number of points per bin (an unweighted histogram);
  empty bins `0`.
- `--statistic sum` — sum of `values` per bin (a weighted histogram); empty bins
  `0`.
- `--statistic min` / `max` — extreme of `values` per bin; empty bins `NaN`.

Binning follows scipy exactly: `bins` equal-width bins over `range` (default
`(x.min(), x.max())`); all but the last bin are half-open `[edge, next)`, the
last is closed `[edge, max]`. Values outside `range` are excluded from the
statistic. The reported `binnumber` is 1-based: `0` means below the range,
`1..=bins` a real bin, `bins+1` above the range.

## Value-exactness

scipy reduces each statistic through numpy, and reproducing it to the last bit
hinges on a few ordering details:

- **Bin edges.** `np.linspace(min, max, bins+1)` is reproduced with numpy's
  exact scalar arithmetic (the `i * step` form with the final element forced to
  `stop`), so every edge — and thus every comparison against it — is bit-exact.
- **Bin numbers.** `np.digitize` with scipy's rightmost-edge correction: a value
  on the last edge (to scipy's rounding precision,
  `int(-log10(min_spacing)) + 6`) is pulled into the last real bin instead of
  the upper-outlier bin.
- **Sequential accumulation.** mean/std/count/sum go through `np.bincount`, which
  accumulates in input order with an ordinary sequential add — *not* numpy's
  pairwise tree. A single in-order pass over the values reproduces it bit-for-bit
  even at n = 1e6 (verified), where a pairwise sum would diverge in the low bits.
- **std** is scipy's two-pass form: per-element `delta = v − binmean`, then
  `sqrt(bincount(delta²) / count)`.
- **median** replicates scipy's `lexsort((values, binnumber))` then
  `(values[floor(mid)] + values[ceil(mid)]) / 2` with `mid = (count−1)/2`.
- **min/max** replicate scipy's argsort-and-scatter (reversed for min) so the
  last write per bin is the extreme.

`tests/compat.rs` re-derives committed scipy goldens (the small fixtures plus
deterministic n=5000 and n=1e6 LCG vectors, all seven statistics across a grid of
bins and ranges including empty bins) with no scipy present. Statistic and edge
values are asserted **bit-exact (0 ULP)**; `std`, whose final `sqrt` may differ
by a ULP across architectures, is bounded at 1 ULP (it measured 0 ULP on the
build machine). `binnumber` is checked exactly (full array up to n=5000, a
checksum at n=1e6 to keep the committed golden small). Goldens are stored as
IEEE-754 hex bit patterns because serde_json's float parser is not
correctly-rounded and would otherwise mask a true bit-exact match.

## Performance

Single-thread, ours vs scipy single-thread on a 1e6-row fixture (mini_m2, M2;
scipy 1.17.1, `*_NUM_THREADS=1`), `--bins 50 --range 0,100`:

| stat   | end-to-end | compute-only |
|--------|-----------|--------------|
| mean   | 13.94×    | 9.15×        |
| std    | 13.21×    | 7.48×        |
| count  | 14.04×    | 9.74×        |
| sum    | 13.50×    | 9.87×        |
| median | 10.01×    | 5.40×        |
| min    | 10.69×    | 3.98×        |
| max    | 10.78×    | 4.24×        |

mean/std/count/sum are a single in-order pass; median/min/max sort within bins.
See `perf/` for full provenance. scipy pays the Python call plus several
intermediate numpy allocations on top of the same arithmetic.

## Origin

This crate is an independent Rust reimplementation of
`scipy.stats.binned_statistic` based on:

- The scipy 1.17.1 source (`scipy/stats/_binned_statistic.py`, BSD-3-Clause),
  which is permissively licensed and was read and cited: `_bin_edges` builds
  `linspace(min, max, bins+1)`; `_bin_numbers` is `np.digitize` with the
  rightmost-edge rounding correction; the per-statistic reductions are scipy's
  `_bincount` (mean/std/count/sum), `lexsort` median, and argsort-scatter
  min/max.
- The numpy 2.x reduction semantics — `np.bincount`'s sequential accumulation
  and `np.linspace`'s scalar arithmetic — reproduced so the result is value-exact
  at large N.

No GPL source was consulted. Test fixtures are independently generated by a
deterministic LCG.

License: MIT OR Apache-2.0.
Upstream credit: scipy https://github.com/scipy/scipy (BSD-3-Clause),
numpy https://github.com/numpy/numpy (BSD-3-Clause).
