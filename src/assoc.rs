// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use itertools::izip;
use polars::datatypes::DataType::Float64;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use statrs::distribution::{ContinuousCDF, StudentsT};

// Dunning's log likelihood (G²)

#[polars_expr(output_type=Float64)]
fn py_loglik(inputs: &[Series]) -> PolarsResult<Series> {
    let f12_ca = get_f64_chunked_array(&inputs[0])?;
    let f1_ca = get_f64_chunked_array(&inputs[1])?;
    let f2_ca = get_f64_chunked_array(&inputs[2])?;
    let n_ca = get_f64_chunked_array(&inputs[3])?;

    let ll: Float64Chunked = izip!(f12_ca.iter(), f1_ca.iter(), f2_ca.iter(), n_ca.iter())
        .map(|(f12, f1, f2, n)| match (f12, f1, f2, n) {
            (Some(f12), Some(f1), Some(f2), Some(n)) => loglik_row(f12, f1, f2, n),
            _ => None,
        })
        .collect();
    Ok(ll.into_series())
}

fn loglik_row(f12: f64, f1: f64, f2: f64, n: f64) -> Option<f64> {
    if !f12.is_finite() || !f1.is_finite() || !f2.is_finite() || !n.is_finite() {
        return None;
    }
    if f12 < 0.0 || f1 <= 0.0 || f2 <= 0.0 || n <= 0.0 {
        return None;
    }

    let o11 = f12;
    let o12 = f1 - f12;
    let o21 = f2 - f12;
    let o22 = n - f1 - f2 + f12;

    let e11 = f1 * f2 / n;
    let e12 = f1 * (n - f2) / n;
    let e21 = (n - f1) * f2 / n;
    let e22 = (n - f1) * (n - f2) / n;

    fn term(o: f64, e: f64) -> f64 {
        if o == 0.0 { 0.0 } else { o * (o / e).ln() }
    }

    let ll = term(o11, e11) + term(o12, e12) + term(o21, e21) + term(o22, e22);
    if o11 < e11 {
        Some(-ll * 2.0)
    } else {
        Some(ll * 2.0)
    }
}

fn get_f64_chunked_array(series: &Series) -> PolarsResult<Float64Chunked> {
    use DataType::*;
    match series.dtype() {
        Float64 => Ok(series.f64()?.clone()),
        Float32 | UInt8 | UInt16 | UInt32 | UInt64 | Int8 | Int16 | Int32 | Int64 => {
            Ok(series.cast(&Float64)?.f64()?.clone())
        },
        dt => Err(PolarsError::InvalidOperation(
            format!("Unsupported dtype {:?}. Expected any integer or float.", dt).into(),
        )),
    }
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

fn py_welchs_t(inputs: &[Series], kwargs: AlternativeKwargs) -> PolarsResult<Series> {
    let s1: &Series = &inputs[0].cast(&Float64)?;
    let s2: &Series = &inputs[1].cast(&Float64)?;

    let sums1: f64 = s1.sum()?;
    let sumsqs1: f64 = (s1 * s1)?.sum()?;
    let n1: f64 = s1.len() as f64;
    let sums2: f64 = s2.sum()?;
    let sumsqs2: f64 = (s2 * s2)?.sum()?;
    let n2: f64 = s2.len() as f64;

    // dbg!(sums1, sumsqs1, n1, sums2, sumsqs2, n2);

    let (t, p, df) = welchs_t_from_stats(
        Some(sums1),
        Some(sumsqs1),
        n1,
        Some(sums2),
        Some(sumsqs2),
        n2,
        &kwargs,
    );

    let t_s = Series::new("stat".into(), &[t]);
    let p_s = Series::new("pval".into(), &[p]);
    let df_s = Series::new("df".into(), &[df]);
    let ret_value =
        StructChunked::from_series("t_test".into(), 1, [&t_s, &p_s, &df_s].into_iter())?;

    Ok(ret_value.into_series())
}
fn welchs_t_from_stats(
    sum1: Option<f64>,
    sumsq1: Option<f64>,
    n1: f64,
    sum2: Option<f64>,
    sumsq2: Option<f64>,
    n2: f64,
    kwargs: &AlternativeKwargs,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let (s1, ss1, s2, ss2) = match (sum1, sumsq1, sum2, sumsq2) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return (None, None, None),
    };

    if !(n1.is_finite() && n2.is_finite()) || n1 < 2.0 || n2 < 2.0 {
        return (None, None, None);
    }

    let m1 = s1 / n1;
    let v1 = (ss1 - s1 * s1 / n1) / (n1 - 1.0);
    let m2 = s2 / n2;
    let v2 = (ss2 - s2 * s2 / n2) / (n2 - 1.0);

    if !(v1.is_finite() && v2.is_finite()) || v1 <= 0.0 || v2 <= 0.0 {
        return (None, None, None);
    }

    let a = v1 / n1;
    let b = v2 / n2;
    let se = v1 / n1 + v2 / n2;
    let t = (m1 - m2) / se.sqrt();
    let ddof = se.powi(2) / (a.powi(2) / (n1 - 1.) + b.powi(2) / (n2 - 1.));
    let dist = StudentsT::new(0.0, 1.0, ddof).unwrap();
    let pval = if kwargs.alt == "less" {
        dist.sf(-t)
    } else if kwargs.alt == "greater" {
        dist.sf(t)
    } else {
        2.0 * dist.sf(t.abs())
    };
    //             (Some(t), Some(pval), Some(df))

    (Some(t), Some(pval), Some(ddof))
}

#[polars_expr(output_type_func=t_test_output_schema)]

fn py_welchs_t_from_stats(inputs: &[Series], kwargs: AlternativeKwargs) -> PolarsResult<Series> {
    let sums1_ca = get_f64_chunked_array(&inputs[0])?;
    let sumsqs1_ca = get_f64_chunked_array(&inputs[1])?;
    let n1_ca = get_f64_chunked_array(&inputs[2])?;
    let sums2_ca = get_f64_chunked_array(&inputs[3])?;
    let sumsqs2_ca = get_f64_chunked_array(&inputs[4])?;
    let n2_ca = get_f64_chunked_array(&inputs[5])?;

    let n = sums1_ca.len();

    let mut t_v = Vec::with_capacity(n);
    let mut p_v = Vec::with_capacity(n);
    let mut d_v = Vec::with_capacity(n);

    for (s1, ss1, n1, s2, ss2, n2) in izip!(
        sums1_ca.iter(),
        sumsqs1_ca.iter(),
        n1_ca.iter(),
        sums2_ca.iter(),
        sumsqs2_ca.iter(),
        n2_ca.iter()
    ) {
        let (t, pval, df) =
            welchs_t_from_stats(s1, ss1, n1.unwrap(), s2, ss2, n2.unwrap(), &kwargs);
        t_v.push(t);
        p_v.push(pval);
        d_v.push(df);
    }

    let t_s = Series::new("stat".into(), &t_v);
    let p_s = Series::new("pval".into(), &p_v);
    let df_s = Series::new("df".into(), &d_v);
    let ret_value =
        StructChunked::from_series("t_test".into(), n, [&t_s, &p_s, &df_s].into_iter())?;

    Ok(ret_value.into_series())
}
