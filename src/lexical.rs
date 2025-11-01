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

/// Count factors from an iterator of optional string references
fn count_factors<'a, I>(tokens: I, threshold: f64) -> f64
where
    I: Iterator<Item = Option<&'a str>>,
{
    let mut factors = 0.0;
    let mut types: PlHashSet<&'a str> = PlHashSet::new();
    let mut token_count = 0;

    for token in tokens.flatten() {
        types.insert(token);
        token_count += 1;

        let ttr = types.len() as f64 / token_count as f64;

        if ttr <= threshold {
            factors += 1.0;
            types.clear();
            token_count = 0;
        }
    }

    // Add partial factor if there are remaining tokens
    if token_count > 0 {
        let ttr = types.len() as f64 / token_count as f64;
        let partial_factor = (1.0 - ttr) / (1.0 - threshold);
        factors += partial_factor;
    }

    factors
}

#[polars_expr(output_type=Float64)]
fn py_mtld(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let threshold = 0.720;

    // Count non-null tokens
    let n_tokens = ca.len() - ca.null_count();

    if n_tokens < 10 {
        return Ok(Series::new("mtld".into(), &[None::<f64>]));
    }

    // Count factors forward
    let forward_factors = count_factors(ca.iter(), threshold);

    // Count factors backward (iterate in reverse without collecting)
    let backward_factors = count_factors(ca.iter().rev(), threshold);

    // MTLD is the average of forward and backward
    let forward_mtld = n_tokens as f64 / forward_factors;
    let backward_mtld = n_tokens as f64 / backward_factors;
    let mtld = (forward_mtld + backward_mtld) / 2.0;

    Ok(Series::new("mtld".into(), &[mtld]))
}
