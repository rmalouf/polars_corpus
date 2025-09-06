// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use statrs::distribution::{ContinuousCDF, StudentsT};

#[polars_expr(output_type=Float64)]
fn loglik(inputs: &[Series]) -> PolarsResult<Series> {
    let f12_ca = inputs[0].u32()?;
    let f1_ca = inputs[1].u32()?;
    let f2_ca = inputs[2].u32()?;
    let n_ca = inputs[3].u32()?;

    let ll: Float64Chunked = izip!(f12_ca, f1_ca, f2_ca, n_ca)
        .map(|(f12, f1, f2, n)| match (f12, f1, f2, n) {
            (Some(f12), Some(f1), Some(f2), Some(n)) => Some(loglik_element(f12, f1, f2, n)),
            _ => None,
        })
        .collect();
    Ok(ll.into_series())
}

fn loglik_element(u_f12: u32, u_f1: u32, u_f2: u32, u_n: u32) -> f64 {
    let f12: f64 = u_f12.into();
    let f1: f64 = u_f1.into();
    let f2: f64 = u_f2.into();
    let n: f64 = u_n.into();
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

// Welch's t test for two samples with unequal variances

#[derive(Deserialize)]
struct AlternativeKwargs {
    alt: String,
}

pub fn t_test_output_schema(_: &[Field]) -> PolarsResult<Field> {
    let fields = DataType::Struct(vec![
        Field::new("stat".into(), DataType::Float64),
        Field::new("pval".into(), DataType::Float64),
        Field::new("df".into(), DataType::Float64),
    ]);

    Ok(Field::new("t_test".into(), fields))
}

#[polars_expr(output_type_func=t_test_output_schema)]
fn welchs_t(inputs: &[Series], kwargs: AlternativeKwargs) -> PolarsResult<Series> {
    let s1: &Series = &inputs[0];
    let s2: &Series = &inputs[1];

    let m1: Option<f64> = s1.mean();
    let v1: Option<f64> = s1.var(1);
    let n1: f64 = s1.len() as f64;

    let m2: Option<f64> = s2.mean();
    let v2: Option<f64> = s2.var(1);
    let n2: f64 = s2.len() as f64;

    let (t, p, df) = match (m1, m2, v1, v2) {
        (Some(m1), Some(m2), Some(v1), Some(v2))
            if v1 > 0.0 && v2 > 0.0 && n1 >= 2.0 && n2 >= 2.0 =>
        {
            let vn1 = v1 / n1;
            let vn2 = v2 / n2;
            let t = (m1 - m2) / (vn1 + vn2).sqrt();
            let df = (vn1 + vn2).powi(2) / (vn1.powi(2) / (n1 - 1.) + (vn2.powi(2) / (n2 - 1.)));
            let dist = StudentsT::new(0.0, 1.0, df).unwrap();
            let pval = if kwargs.alt == "less" {
                dist.sf(-t)
            } else if kwargs.alt == "greater" {
                dist.sf(t)
            } else {
                2.0 * dist.sf(t.abs())
            };
            (Some(t), Some(pval), Some(df))
        },
        _ => (None, None, None),
    };

    let t_s = Series::new("stat".into(), &[t]);
    let p_s = Series::new("pval".into(), &[p]);
    let df_s = Series::new("df".into(), &[df]);
    let ret_value =
        StructChunked::from_series("t_test".into(), 1, [&t_s, &p_s, &df_s].into_iter())?;

    Ok(ret_value.into_series())
}
