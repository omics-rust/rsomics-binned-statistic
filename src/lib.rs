//! Per-bin statistic of a second variable — `scipy.stats.binned_statistic`,
//! value-exact to scipy 1.17.1.
//!
//! Bins `x` into equal-width bins and computes a statistic (mean, std, median,
//! count, sum, min, max) of the associated `values` within each bin. The bin
//! edges, the per-value bin numbers, and every per-bin statistic are reproduced
//! bit-for-bit; see `edges` for the numpy `linspace`/`digitize` semantics and
//! `binned` for the per-bin reductions.

pub mod binned;
pub mod edges;
pub mod io;

pub use binned::{Binned, Statistic, binned_statistic};
pub use edges::{bin_edges, bin_numbers, data_range};
pub use io::read_xy;
