use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};
use serde::Serialize;

use rsomics_binned_statistic::{Statistic, binned_statistic, read_xy};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Stat {
    Mean,
    Std,
    Median,
    Count,
    Sum,
    Min,
    Max,
}

impl From<Stat> for Statistic {
    fn from(s: Stat) -> Self {
        match s {
            Stat::Mean => Statistic::Mean,
            Stat::Std => Statistic::Std,
            Stat::Median => Statistic::Median,
            Stat::Count => Statistic::Count,
            Stat::Sum => Statistic::Sum,
            Stat::Min => Statistic::Min,
            Stat::Max => Statistic::Max,
        }
    }
}

/// Per-bin statistic of a second variable — value-exact to
/// `scipy.stats.binned_statistic`.
///
/// Bins `x` into `--bins` equal-width bins over `--range` (default the data
/// range) and computes `--statistic` of the associated `values` within each
/// bin. Input is a two-column `x<TAB>values` table (`-` reads stdin), or `x`
/// from DATA and `values` from `--values FILE`. Prints `bin_index<TAB>statistic`
/// per line with the bin edges as a header (or the full result, including
/// `binnumber`, in the `--json` envelope).
#[derive(Parser, Debug)]
#[command(name = "rsomics-binned-statistic", version, about, long_about = None)]
pub struct Cli {
    /// Two-column `x<TAB>values` table, or the `x` column when `--values` is set
    /// (`-` reads stdin).
    #[arg(value_name = "DATA", default_value = "-")]
    pub data: PathBuf,

    /// Statistic to compute per bin.
    #[arg(long, value_enum, default_value_t = Stat::Mean)]
    pub statistic: Stat,

    /// Number of equal-width bins.
    #[arg(long, value_name = "N", default_value_t = 10)]
    pub bins: usize,

    /// Bin range as `lo,hi`. Without this, the data range `(x.min(), x.max())`
    /// is used.
    #[arg(
        long,
        value_name = "LO,HI",
        value_delimiter = ',',
        allow_hyphen_values = true
    )]
    pub range: Vec<f64>,

    /// Read `values` from this file (one per line); `DATA` then holds only `x`.
    #[arg(long, value_name = "FILE")]
    pub values: Option<PathBuf>,

    /// Print the per-value bin numbers (1-based) after the statistic table.
    #[arg(long)]
    pub binnumber: bool,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
struct Output {
    statistic_name: &'static str,
    statistic: Vec<f64>,
    bin_edges: Vec<f64>,
    binnumber: Vec<usize>,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            if self.bins < 1 {
                return Err(RsomicsError::InvalidInput(
                    "`--bins` must be at least 1".into(),
                ));
            }

            let (x, values) = read_xy(&self.data, self.values.as_deref())?;
            if x.len() != values.len() {
                return Err(RsomicsError::InvalidInput(format!(
                    "x length {} does not match values length {}",
                    x.len(),
                    values.len()
                )));
            }

            let range = match self.range.as_slice() {
                [] => None,
                [lo, hi] => {
                    if hi < lo {
                        return Err(RsomicsError::InvalidInput(format!(
                            "in range, start must be <= stop: [{lo}, {hi}]"
                        )));
                    }
                    Some((*lo, *hi))
                }
                _ => {
                    return Err(RsomicsError::InvalidInput(
                        "`--range` takes exactly two values: lo,hi".into(),
                    ));
                }
            };

            let stat: Statistic = self.statistic.into();
            let r = binned_statistic(&x, &values, stat, self.bins, range);

            if !common.json {
                let edges: Vec<String> = r.bin_edges.iter().map(f64::to_string).collect();
                println!("# statistic={} bin_edges={}", stat.name(), edges.join(","));
                for (i, v) in r.statistic.iter().enumerate() {
                    println!("{i}\t{v}");
                }
                if self.binnumber {
                    println!("# binnumber");
                    for b in &r.binnumber {
                        println!("{b}");
                    }
                }
            }

            Ok(Output {
                statistic_name: stat.name(),
                statistic: r.statistic,
                bin_edges: r.bin_edges,
                binnumber: r.binnumber,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
