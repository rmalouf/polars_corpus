// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use polars::chunked_array::builder::{ListBuilderTrait, get_list_builder};
use polars::prelude::*;
use pyo3::PyResult;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySeries};

use crate::matcher::Match;

#[pyfunction]
pub fn spans_to_chunks(spans: Vec<Span>, n: usize) -> PyResult<PySeries> {
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
    // polars <-> pyo3 shim
    py_df: PyDataFrame,
    matches: Vec<Match>,
    py_chunk_tag: PySeries,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = py_df.0;
    let chunk_tag = py_chunk_tag.0;
    let matched_spans: Vec<Span> = matches.into_iter().map(|m| m.span).collect();
    let out_df = concordance_df(&df, &matched_spans, &chunk_tag).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out_df))
}

fn concordance_df(
    df: &DataFrame,
    matched_spans: &[Span],
    chunk_tag: &Series,
) -> PolarsResult<DataFrame> {
    let mut result_columns: Vec<Column> = Vec::new();
    for column in df.columns() {
        let left_spans = Some(left_chunk_context_from_spans(matched_spans, chunk_tag)?);
        let right_spans = Some(right_chunk_context_from_spans(matched_spans, chunk_tag)?);
        add_columns(
            column,
            left_spans.as_deref(),
            matched_spans,
            right_spans.as_deref(),
            &mut result_columns,
        )?;
    }
    DataFrame::new_infer_height(result_columns)
}

fn add_columns(
    column: &Column,
    left_spans: Option<&[Span]>,
    matched_spans: &[Span],
    right_spans: Option<&[Span]>,
    result_columns: &mut Vec<Column>,
) -> PolarsResult<()> {
    let column_name = column.name();
    let series = column.as_materialized_series();

    if let Some(left_spans) = left_spans {
        let left_context: Series = implode_series_by_spans(series, left_spans)?;
        result_columns.push(
            left_context
                .with_name(format!("{column_name}_left_context").into())
                .into(),
        );
    };

    let node: Series = implode_series_by_spans(series, matched_spans)?;
    result_columns.push(node.with_name(format!("{column_name}").into()).into());

    if let Some(right_spans) = right_spans {
        let right_context: Series = implode_series_by_spans(series, right_spans)?;
        result_columns.push(
            right_context
                .with_name(format!("{column_name}_right_context").into())
                .into(),
        );
    }

    Ok(())
}

fn left_chunk_context_from_spans(spans: &[Span], chunk_tag: &Series) -> PolarsResult<Vec<Span>> {
    let mut context_spans = Vec::with_capacity(spans.len());
    for &Span { start, end: _ } in spans {
        let mut left_edge = (start as i32) - 1;
        while left_edge > 0 && chunk_tag.get(left_edge as usize)?.extract_str().unwrap() == "I" {
            left_edge -= 1;
        }
        context_spans.push(Span::new(left_edge as usize, start));
    }
    Ok(context_spans)
}

fn right_chunk_context_from_spans(spans: &[Span], chunk_tag: &Series) -> PolarsResult<Vec<Span>> {
    let mut context_spans = Vec::with_capacity(spans.len());
    let df_height = chunk_tag.len();
    for &Span { start: _, end } in spans {
        let mut right_edge = end as i32;
        while right_edge < df_height as i32
            && chunk_tag.get(right_edge as usize)?.extract_str().unwrap() == "I"
        {
            right_edge += 1;
        }
        context_spans.push(Span::new(end, right_edge as usize));
    }
    Ok(context_spans)
}

#[pyfunction]
pub fn py_kwic(
    // polars <-> pyo3 shim
    py_df: PyDataFrame,
    matches: Vec<Match>,
    left_window: i32,
    right_window: i32,
) -> PyResult<PyDataFrame> {
    let matched_spans: Vec<Span> = matches.into_iter().map(|m| m.span).collect();
    let df: DataFrame = py_df.0;
    let out_df =
        kwic_df(&df, &matched_spans, left_window, right_window).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out_df))
}

fn kwic_df(
    df: &DataFrame,
    matched_spans: &[Span],
    left_window: i32,
    right_window: i32,
) -> PolarsResult<DataFrame> {
    let mut result_columns: Vec<Column> = Vec::new();
    for column in df.columns() {
        let left_spans = if left_window > 0 {
            Some(left_fixed_context_from_spans(matched_spans, left_window)?)
        } else {
            None
        };
        let right_spans = if right_window > 0 {
            Some(right_fixed_context_from_spans(matched_spans, right_window)?)
        } else {
            None
        };
        add_columns(
            column,
            left_spans.as_deref(),
            matched_spans,
            right_spans.as_deref(),
            &mut result_columns,
        )?;
    }
    DataFrame::new_infer_height(result_columns)
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

fn right_fixed_context_from_spans(spans: &[Span], window_size: i32) -> PolarsResult<Vec<Span>> {
    let mut context_spans = Vec::with_capacity(spans.len());
    for &Span { start: _, end } in spans {
        let right_edge = end + window_size as usize;
        context_spans.push(Span::new(end, right_edge));
    }
    Ok(context_spans)
}

// consider batching: https://claude.ai/chat/5682209c-4114-49b0-9a1d-359ac330dcf8

fn implode_series_by_spans(s: &Series, spans: &[Span]) -> PolarsResult<Series> {
    let values_cap: usize = spans.iter().map(|sp| sp.end - sp.start).sum();
    let mut builder = get_list_builder(s.dtype(), spans.len(), values_cap, s.name().clone());

    for &Span { start, end } in spans {
        let clipped_end = end.min(s.len());
        let slice = s.slice(start as i64, clipped_end - start);
        builder.append_series(&slice)?;
    }

    let out_series = builder.finish().into_series();
    Ok(out_series)
}

// Span

#[pyclass(eq, module = "polars_corpus", from_py_object)]
#[derive(Clone, PartialEq, Debug)]
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
