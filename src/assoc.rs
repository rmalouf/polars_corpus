// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
fn loglik(inputs: &[Series]) -> PolarsResult<Series> {
    let f12_series = inputs[0].cast(&DataType::Float64)?;
    let f1_series = inputs[1].cast(&DataType::Float64)?;
    let f2_series = inputs[2].cast(&DataType::Float64)?;
    let n_series = inputs[3].cast(&DataType::Float64)?;

    let f12_ca = f12_series.f64()?;
    let f1_ca = f1_series.f64()?;
    let f2_ca = f2_series.f64()?;
    let n_ca = n_series.f64()?;

    let ll: Vec<f64> = izip!(f12_ca, f1_ca, f2_ca, n_ca)
        .map(|(f12, f1, f2, n)| loglik_element(f12.unwrap(), f1.unwrap(), f2.unwrap(), n.unwrap()))
        .collect();
    Ok(Series::new("LL".into(), ll))
}

fn loglik_element(f12: f64, f1: f64, f2: f64, n: f64) -> f64 {
    let o11: f64 = f12;
    let o12: f64 = f1 - f12;
    let o21: f64 = f2 - f12;
    let o22: f64 = n - f1 - f2 + f12;

    let mut ll: f64 = 0.0;
    if o11 != 0.0 {
        let e11 = f1 * f2 / n;
        ll += o11 * (o11 / e11).ln();
    };
    if o12 != 0.0 {
        let e12 = f1 * (n - f2) / n;
        ll += o12 * (o12 / e12).ln();
    };
    if o21 != 0.0 {
        let e21 = (n - f1) * f2 / n;
        ll += o21 * (o21 / e21).ln();
    };
    if o22 != 0.0 {
        let e22 = (n - f1) * (n - f2) / n;
        ll += o22 * (o22 / e22).ln();
    };
    ll * 2.0
}
