// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use polars::chunked_array::builder::{ListBuilderTrait, get_list_builder};
use polars::prelude::*;
use pyo3::exceptions::{PyException, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::{PyErr, PyResult};
use pyo3_polars::{PyDataFrame, PySeries};

#[pyfunction]
pub fn _to_chunks(n: usize, spans: Vec<Span>) -> PyResult<PySeries> {
    let mut span_vec = vec!["O"; n];
    for span in spans {
        let start = span.start;
        let end = span.end;
        if (start > n) | (end > n) {
            return Err(PyValueError::new_err("index out of bounds"));
        } else {
            span_vec[start] = "B";
            if start + 1 < end {
                span_vec[start + 1..end].fill("I");
            }
        }
    }
    let result = Series::new("spans".into(), &span_vec);
    Ok(PySeries(result))
}

#[pyfunction]
pub fn py_concordance(
    py_df: PyDataFrame,
    matched_spans: Vec<Span>,
    window_size: Option<i32>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = py_df.0;
    let out_df = concordance_df(df, matched_spans, window_size)
        .map_err(|e: PolarsError| PyErr::new::<PyException, _>(e.to_string()))?;
    Ok(PyDataFrame(out_df))
}

fn concordance_df(
    df: DataFrame,
    matched_spans: Vec<Span>,
    window_size: Option<i32>,
) -> PolarsResult<DataFrame> {
    let mut result_columns: Vec<Column> = Vec::new();
    for column in df.get_columns() {
        let column_name = column.name();
        let series = column.as_materialized_series();

        match window_size {
            Some(window_size) => {
                let (left_context, node, right_context) =
                    kwic_concordance_series(series, &matched_spans, window_size)?;
                result_columns.push(
                    left_context
                        .with_name(format!("{column_name}_left_context").as_str().into())
                        .into(),
                );
                result_columns.push(
                    node.with_name(format!("{column_name}_node").as_str().into())
                        .into(),
                );
                result_columns.push(
                    right_context
                        .with_name(format!("{column_name}_right_context").as_str().into())
                        .into(),
                );
            },
            None => {
                let node = implode_series_by_spans(series, &matched_spans)?;
                result_columns.push(
                    node.with_name(format!("{column_name}_node").as_str().into())
                        .into(),
                );
            },
        }
    }
    DataFrame::new(result_columns)
}

fn kwic_concordance_series(
    series: &Series,
    matched_spans: &[Span],
    window_size: i32,
) -> PolarsResult<(Series, Series, Series)> {
    let left_spans = left_fixed_context_from_spans(matched_spans, window_size)?;
    let right_spans = right_fixed_context_from_spans(series.len(), matched_spans, window_size)?;

    let node = implode_series_by_spans(series, matched_spans)?;
    let left_context = implode_series_by_spans(series, &left_spans)?;
    let right_context = implode_series_by_spans(series, &right_spans)?;

    Ok((left_context, node, right_context))
}

fn left_fixed_context_from_spans(spans: &[Span], window_size: i32) -> PolarsResult<Vec<Span>> {
    let mut context_spans = Vec::with_capacity(spans.len());
    for &Span { start, end: _ } in spans {
        let left_edge = (start as i32) - window_size;
        if left_edge < 0 {
            context_spans.push(Span::new(0, start));
        } else {
            context_spans.push(Span::new(left_edge as usize, start));
        }
    }
    Ok(context_spans)
}

fn right_fixed_context_from_spans(
    df_height: usize,
    spans: &[Span],
    window_size: i32,
) -> PolarsResult<Vec<Span>> {
    let mut context_spans = Vec::with_capacity(spans.len());
    for &Span { start: _, end } in spans {
        let right_edge = end + window_size as usize;
        if right_edge > df_height {
            context_spans.push(Span::new(end, df_height));
        } else {
            context_spans.push(Span::new(end, right_edge));
        }
    }
    Ok(context_spans)
}

// consider batching: https://claude.ai/chat/5682209c-4114-49b0-9a1d-359ac330dcf8

fn implode_series_by_spans(s: &Series, spans: &[Span]) -> PolarsResult<Series> {
    let values_cap: usize = spans.iter().map(|sp| sp.end - sp.start).sum();
    let mut builder = get_list_builder(s.dtype(), spans.len(), values_cap, s.name().clone());

    for &Span { start, end } in spans {
        let slice = s.slice(start as i64, ((end as isize) - (start as isize)) as usize);
        builder.append_series(&slice)?;
    }

    let out_series = builder.finish().into_series();
    Ok(out_series)
}

// Span

#[pyclass(eq, module = "polars_corpus")]
#[derive(Clone, PartialEq)]
pub struct Span {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
}

#[pymethods]
impl Span {
    #[new]
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }

    fn __repr__(&self) -> String {
        format!("Span({}, {})", self.start, self.end)
    }

    fn __getitem__(&self, index: usize) -> PyResult<usize> {
        match index {
            0 => Ok(self.start),
            1 => Ok(self.end),
            _ => Err(PyIndexError::new_err("Index out of range")),
        }
    }

    fn __len__(&self) -> usize {
        2
    }
}
