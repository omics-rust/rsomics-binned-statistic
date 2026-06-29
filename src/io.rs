//! Numeric input: a two-column `x<TAB>values` table, or two single-column files.
//!
//! The table reader splits on ASCII whitespace and parses each token with
//! `fast_float2`, taking the first two fields of every non-empty line. No
//! per-line `String` allocation.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use rsomics_common::{Result, RsomicsError};

fn slurp(path: &Path) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    if path.as_os_str() == "-" {
        std::io::stdin()
            .lock()
            .read_to_end(&mut buf)
            .map_err(RsomicsError::Io)?;
    } else {
        File::open(path)
            .map_err(RsomicsError::Io)?
            .read_to_end(&mut buf)
            .map_err(RsomicsError::Io)?;
    }
    Ok(buf)
}

fn parse_f64(tok: &[u8]) -> Result<f64> {
    fast_float2::parse(tok).map_err(|_| {
        RsomicsError::InvalidInput(format!(
            "value '{}' is not a number",
            String::from_utf8_lossy(tok)
        ))
    })
}

/// Parse a single whitespace-separated column of f64 values.
pub fn parse_column(buf: &[u8]) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &b) in buf.iter().enumerate() {
        if b.is_ascii_whitespace() {
            if let Some(s) = start.take() {
                out.push(parse_f64(&buf[s..i])?);
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push(parse_f64(&buf[s..])?);
    }
    if out.is_empty() {
        return Err(RsomicsError::InvalidInput("no values in input".into()));
    }
    Ok(out)
}

/// Parse a two-column table: each non-empty line yields `(x, value)` from its
/// first two whitespace-separated fields.
pub fn parse_table(buf: &[u8]) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut x = Vec::new();
    let mut v = Vec::new();
    for raw in buf.split(|&b| b == b'\n') {
        let line = raw.strip_suffix(b"\r").unwrap_or(raw);
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let mut fields = line
            .split(u8::is_ascii_whitespace)
            .filter(|f| !f.is_empty());
        let xt = fields
            .next()
            .ok_or_else(|| RsomicsError::InvalidInput("line has no x column".into()))?;
        let vt = fields.next().ok_or_else(|| {
            RsomicsError::InvalidInput(
                "line has only one column; need `x<TAB>values` or --values FILE".into(),
            )
        })?;
        x.push(parse_f64(xt)?);
        v.push(parse_f64(vt)?);
    }
    if x.is_empty() {
        return Err(RsomicsError::InvalidInput("no rows in input".into()));
    }
    Ok((x, v))
}

/// Read `x` and `values`. With `values_path`, read `x` as a single column from
/// `data_path` and `values` from `values_path`; otherwise read a two-column
/// table from `data_path`.
pub fn read_xy(data_path: &Path, values_path: Option<&Path>) -> Result<(Vec<f64>, Vec<f64>)> {
    match values_path {
        Some(vp) => {
            let x = parse_column(&slurp(data_path)?)?;
            let v = parse_column(&slurp(vp)?)?;
            Ok((x, v))
        }
        None => parse_table(&slurp(data_path)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_parses() {
        assert_eq!(parse_column(b"1\n2.5\n-3\n").unwrap(), vec![1.0, 2.5, -3.0]);
    }

    #[test]
    fn table_parses_two_columns() {
        let (x, v) = parse_table(b"1\t10\n2\t20\n3\t30\n").unwrap();
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
        assert_eq!(v, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn table_skips_blank_lines() {
        let (x, v) = parse_table(b"1 10\n\n2 20\n").unwrap();
        assert_eq!(x, vec![1.0, 2.0]);
        assert_eq!(v, vec![10.0, 20.0]);
    }

    #[test]
    fn table_rejects_single_column() {
        assert!(parse_table(b"1\n2\n").is_err());
    }
}
