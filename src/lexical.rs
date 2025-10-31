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
    let num_segments = series.len() / n;

    if num_segments == 0 {
        return Ok(Series::new("msttr".into(), &[None::<f64>]));
    }
    
    let ttr_sum: usize = (0..num_segments)
        .map(|i| {
            let start = i * n;
            let segment = series.slice(start as i64, n);
            segment.n_unique().unwrap()
        })
        .sum();

    let msttr = ttr_sum as f64 / (num_segments * n) as f64;

    Ok(Series::new("msttr".into(), &[msttr]))
}
