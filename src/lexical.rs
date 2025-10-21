use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct MSTTRKwargs {
    n: usize,
}

#[polars_expr(output_type=Float64)]
fn py_msttr(inputs: &[Series], kwargs: MSTTRKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let n = kwargs.n;

    let len = series.len();

    // If series is shorter than n, return null (no complete segments)
    if len < n {
        return Ok(Series::new("msttr".into(), &[None::<f64>]));
    }

    let num_segments = len / n;

    let ttr_sum: f64 = (0..num_segments)
        .map(|i| {
            let start = i * n;
            let segment = series.slice(start as i64, n);
            let unique_count = segment.n_unique().unwrap();
            unique_count as f64 / n as f64
        })
        .sum();

    let msttr = ttr_sum / num_segments as f64;

    Ok(Series::new("msttr".into(), &[msttr]))
}
