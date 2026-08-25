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
///
/// Returns `None` if no factor closed, leaving only a partial: tokens per
/// factor would then measure the sequence's TTR, not a factor length.
fn count_factors<'a, I>(tokens: I, threshold: f64) -> Option<f64>
where
    I: Iterator<Item = Option<&'a str>>,
{
    let mut factors: f64 = 0.0;
    let mut types: PlHashSet<Option<&'a str>> = PlHashSet::new();
    let mut token_count: usize = 0;

    for token in tokens {
        types.insert(token);
        token_count += 1;

        let ttr = types.len() as f64 / token_count as f64;
        if ttr <= threshold {
            factors += 1.0;
            types.clear();
            token_count = 0;
        }
    }

    if factors == 0.0 {
        return None;
    }

    // Tokens left over at the end contribute a fraction of a factor
    if token_count > 0 {
        let ttr = types.len() as f64 / token_count as f64;
        factors += (1.0 - ttr) / (1.0 - threshold);
    }

    Some(factors)
}

#[derive(Deserialize)]
struct MTLDKwargs {
    threshold: f64,
}

#[polars_expr(output_type=Float64)]
fn py_mtld(inputs: &[Series], kwargs: MTLDKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let threshold = kwargs.threshold;

    let n_tokens = ca.len();
    if n_tokens < 10 {
        return Ok(Series::new("mtld".into(), &[None::<f64>]));
    }

    let mtld = count_factors(ca.iter(), threshold)
        .zip(count_factors(ca.iter().rev(), threshold))
        .map(|(forward, backward)| (n_tokens as f64 / forward + n_tokens as f64 / backward) / 2.0);

    Ok(Series::new("mtld".into(), &[mtld]))
}
