// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

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

// fn pl_welch_t(inputs: &[Series]) -> PolarsResult<Series> {
//     let mean1 = inputs[0].f64()?;
//     let mean1 = mean1.get(0).unwrap_or(f64::NAN);
//     let mean2 = inputs[1].f64()?;
//     let mean2 = mean2.get(0).unwrap_or(f64::NAN);
//     let var1 = inputs[2].f64()?;
//     let var1 = var1.get(0).unwrap_or(f64::NAN);
//     let var2 = inputs[3].f64()?;
//     let var2 = var2.get(0).unwrap_or(f64::NAN);
//     let n1 = inputs[4].u64()?;
//     let n1 = n1.get(0).map_or(f64::NAN, |x| x as f64);
//     let n2 = inputs[5].u64()?;
//     let n2 = n2.get(0).map_or(f64::NAN, |x| x as f64);
//
//     let alt = inputs[6].str()?;
//     let alt = alt.get(0).unwrap();
//     let alt = stats::Alternative::from(alt);
//
//     // See comment above for why there is no finiteness or n > 0 checks
//
//     let (s, p) = welch_t(mean1, mean2, var1, var2, n1, n2, alt);
//     generic_stats_output(s, p)
// }
//
// #[inline]
// fn welch_t(
//     m1: f64,
//     m2: f64,
//     v1: f64,
//     v2: f64,
//     n1: f64,
//     n2: f64,
//     alt: Alternative,
// ) -> (f64, f64) {
//
//     let num = m1 - m2;
//     let vn1 = v1 / n1;
//     let vn2 = v2 / n2;
//     let denom = (vn1 + vn2).sqrt();
//     let t = num / denom;
//     let df = (vn1 + vn2).powi(2) / (vn1.powi(2) / (n1 - 1.) + (vn2.powi(2) / (n2 - 1.)));
//     let p = match alt {
//         // the distribution is approximately student t
//         Alternative::Less => beta::student_t_sf(-t, df).unwrap_or(f64::NAN),
//         Alternative::Greater => beta::student_t_sf(t, df).unwrap_or(f64::NAN),
//         Alternative::TwoSided => match beta::student_t_sf(t.abs(), df) {
//             Ok(p) => 2.0 * p,
//             Err(_) => f64::NAN,
//         },
//     };
//
//     (t, p)
//
// }
