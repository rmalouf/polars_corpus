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

use crate::matcher::{Match, run_ends};

/// The file `(start, end)` bounds holding each span, or None with no file ids.
/// Context windows are clipped to these so they never cross a file boundary.
fn span_file_bounds(
    spans: &[Span],
    file_ids: Option<&Series>,
) -> PolarsResult<Option<Vec<(usize, usize)>>> {
    let Some(ids) = file_ids else { return Ok(None) };
    let ends = run_ends(ids)?;
    let bounds = spans
        .iter()
        .map(|sp| {
            let i = ends.partition_point(|&e| e <= sp.start).min(ends.len() - 1);
            let start = if i == 0 { 0 } else { ends[i - 1] };
            (start, ends[i])
        })
        .collect();
    Ok(Some(bounds))
}

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
#[pyo3(signature = (py_df, matches, py_chunk_tag, metadata, bindings, file_ids=None))]
pub fn py_concordance(
    // polars <-> pyo3 shim
    py_df: PyDataFrame,
    matches: Vec<Match>,
    py_chunk_tag: PySeries,
    metadata: Option<PyDataFrame>,
    bindings: Vec<String>,
    file_ids: Option<PySeries>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = py_df.0;
    let chunk_tag = py_chunk_tag.0;
    let metadata_df = metadata.map(|m| m.0);
    let bound_spans = bound_spans(&matches, &bindings);
    let matched_spans: Vec<Span> = matches.into_iter().map(|m| m.span).collect();
    let bounds = span_file_bounds(&matched_spans, file_ids.as_ref().map(|s| s.as_ref()))
        .map_err(PyPolarsErr::from)?;
    let out_df = concordance_df(
        &df,
        &matched_spans,
        &chunk_tag,
        metadata_df.as_ref(),
        &bound_spans,
        bounds.as_deref(),
    )
    .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out_df))
}

fn concordance_df(
    df: &DataFrame,
    matched_spans: &[Span],
    chunk_tag: &Series,
    metadata: Option<&DataFrame>,
    bound_spans: &[(&str, Vec<Option<Span>>)],
    bounds: Option<&[(usize, usize)]>,
) -> PolarsResult<DataFrame> {
    // Every column takes its context from the same positions, so the spans
    // are built once and read by all of them.
    let left_spans = left_chunk_context_from_spans(matched_spans, chunk_tag, bounds)?;
    let right_spans = right_chunk_context_from_spans(matched_spans, chunk_tag, bounds)?;
    let mut result_columns: Vec<Column> = Vec::new();
    for column in df.columns() {
        add_columns(
            column,
            Some(&left_spans),
            matched_spans,
            Some(&right_spans),
            &mut result_columns,
        )?;
        add_binding_columns(column, bound_spans, &mut result_columns)?;
    }
    add_metadata_columns(metadata, matched_spans, &mut result_columns)?;
    DataFrame::new_infer_height(result_columns)
}

/// The span each match bound to each name, in the order the names are given.
/// A match that never bound a name gets `None`, and so a null in its column.
fn bound_spans<'a>(matches: &[Match], names: &'a [String]) -> Vec<(&'a str, Vec<Option<Span>>)> {
    names
        .iter()
        .map(|name| {
            let spans = matches
                .iter()
                .map(|m| m.bindings.get(name).cloned())
                .collect();
            (name.as_str(), spans)
        })
        .collect()
}

fn add_binding_columns(
    column: &Column,
    bound_spans: &[(&str, Vec<Option<Span>>)],
    result_columns: &mut Vec<Column>,
) -> PolarsResult<()> {
    let column_name = column.name();
    let series = column.as_materialized_series();
    for (name, spans) in bound_spans {
        let bound = implode_series_by_bound_spans(series, spans)?;
        result_columns.push(
            bound
                .with_name(format!("{column_name}_{name}").into())
                .into(),
        );
    }
    Ok(())
}

fn add_metadata_columns(
    metadata: Option<&DataFrame>,
    matched_spans: &[Span],
    result_columns: &mut Vec<Column>,
) -> PolarsResult<()> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    let idx: Vec<IdxSize> = matched_spans.iter().map(|sp| sp.start as IdxSize).collect();
    let idx_ca = IdxCa::from_vec(PlSmallStr::EMPTY, idx);
    let taken = metadata.take(&idx_ca)?;
    result_columns.extend(taken.into_columns());
    Ok(())
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
    result_columns.push(node.with_name(column_name.clone()).into());

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

// Both directions walk the tag column as a string array, and treat anything
// that isn't "I" -- including a null -- as the edge of the chunk. The file
// bounds, when given, are a harder edge still.
fn left_chunk_context_from_spans(
    spans: &[Span],
    chunk_tag: &Series,
    bounds: Option<&[(usize, usize)]>,
) -> PolarsResult<Vec<Span>> {
    let tags = chunk_tag.str()?;
    let mut context_spans = Vec::with_capacity(spans.len());
    for (i, &Span { start, end: _ }) in spans.iter().enumerate() {
        let floor = bounds.map_or(0, |b| b[i].0);
        // Walk back over the chunk's interior, then take in the tag that opens
        // it. A match at the start of the corpus has neither to its left, and
        // gets an empty context.
        let mut left_edge = start;
        while left_edge > floor && tags.get(left_edge - 1) == Some("I") {
            left_edge -= 1;
        }
        left_edge = left_edge.saturating_sub(1).max(floor);
        context_spans.push(Span::new(left_edge, start));
    }
    Ok(context_spans)
}

fn right_chunk_context_from_spans(
    spans: &[Span],
    chunk_tag: &Series,
    bounds: Option<&[(usize, usize)]>,
) -> PolarsResult<Vec<Span>> {
    let tags = chunk_tag.str()?;
    let mut context_spans = Vec::with_capacity(spans.len());
    let df_height = chunk_tag.len();
    for (i, &Span { start: _, end }) in spans.iter().enumerate() {
        let ceiling = bounds.map_or(df_height, |b| b[i].1);
        // The tag opening the next chunk is not part of this one, so the right
        // context ends before it.
        let mut right_edge = end;
        while right_edge < ceiling && tags.get(right_edge) == Some("I") {
            right_edge += 1;
        }
        context_spans.push(Span::new(end, right_edge));
    }
    Ok(context_spans)
}

#[pyfunction]
#[pyo3(signature = (py_df, matches, left_window, right_window, metadata, bindings, file_ids=None))]
pub fn py_kwic(
    // polars <-> pyo3 shim
    py_df: PyDataFrame,
    matches: Vec<Match>,
    left_window: i32,
    right_window: i32,
    metadata: Option<PyDataFrame>,
    bindings: Vec<String>,
    file_ids: Option<PySeries>,
) -> PyResult<PyDataFrame> {
    let bound_spans = bound_spans(&matches, &bindings);
    let matched_spans: Vec<Span> = matches.into_iter().map(|m| m.span).collect();
    let df: DataFrame = py_df.0;
    let metadata_df = metadata.map(|m| m.0);
    let bounds = span_file_bounds(&matched_spans, file_ids.as_ref().map(|s| s.as_ref()))
        .map_err(PyPolarsErr::from)?;
    let out_df = kwic_df(
        &df,
        &matched_spans,
        left_window,
        right_window,
        metadata_df.as_ref(),
        &bound_spans,
        bounds.as_deref(),
    )
    .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out_df))
}

fn kwic_df(
    df: &DataFrame,
    matched_spans: &[Span],
    left_window: i32,
    right_window: i32,
    metadata: Option<&DataFrame>,
    bound_spans: &[(&str, Vec<Option<Span>>)],
    bounds: Option<&[(usize, usize)]>,
) -> PolarsResult<DataFrame> {
    // Every column takes its context from the same positions, so the spans
    // are built once and read by all of them.
    let left_spans = if left_window > 0 {
        Some(left_fixed_context_from_spans(
            matched_spans,
            left_window,
            bounds,
        )?)
    } else {
        None
    };
    let right_spans = if right_window > 0 {
        Some(right_fixed_context_from_spans(
            matched_spans,
            right_window,
            bounds,
        )?)
    } else {
        None
    };
    let mut result_columns: Vec<Column> = Vec::new();
    for column in df.columns() {
        add_columns(
            column,
            left_spans.as_deref(),
            matched_spans,
            right_spans.as_deref(),
            &mut result_columns,
        )?;
        add_binding_columns(column, bound_spans, &mut result_columns)?;
    }
    add_metadata_columns(metadata, matched_spans, &mut result_columns)?;
    DataFrame::new_infer_height(result_columns)
}

fn left_fixed_context_from_spans(
    spans: &[Span],
    window_size: i32,
    bounds: Option<&[(usize, usize)]>,
) -> PolarsResult<Vec<Span>> {
    let mut context_spans = Vec::with_capacity(spans.len());
    for (i, &Span { start, end: _ }) in spans.iter().enumerate() {
        let floor = bounds.map_or(0, |b| b[i].0);
        let left_edge = start.saturating_sub(window_size as usize).max(floor);
        context_spans.push(Span::new(left_edge, start));
    }
    Ok(context_spans)
}

fn right_fixed_context_from_spans(
    spans: &[Span],
    window_size: i32,
    bounds: Option<&[(usize, usize)]>,
) -> PolarsResult<Vec<Span>> {
    let mut context_spans = Vec::with_capacity(spans.len());
    for (i, &Span { start: _, end }) in spans.iter().enumerate() {
        // Without bounds the edge is clipped to the corpus later, at slicing.
        let ceiling = bounds.map_or(usize::MAX, |b| b[i].1);
        let right_edge = (end + window_size as usize).min(ceiling);
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

fn implode_series_by_bound_spans(s: &Series, spans: &[Option<Span>]) -> PolarsResult<Series> {
    let values_cap: usize = spans.iter().flatten().map(|sp| sp.end - sp.start).sum();
    let mut builder = get_list_builder(s.dtype(), spans.len(), values_cap, s.name().clone());

    for span in spans {
        match span {
            // A bound span lies within the match, so it needs no clipping.
            Some(sp) => builder.append_series(&s.slice(sp.start as i64, sp.end - sp.start))?,
            // The match never bound this name: null, as against the empty list
            // a binding that matched no token gets.
            None => builder.append_null(),
        }
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
