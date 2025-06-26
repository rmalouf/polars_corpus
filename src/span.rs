#![allow(clippy::upper_case_acronyms)]
// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use polars::prelude::*;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::PySeries;

#[pyfunction]
pub fn _to_spans(n: usize, spans: Vec<Span>) -> PyResult<PySeries> {
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
pub fn _make_spans_mask(n: usize, spans: Vec<(usize, usize)>) -> PyResult<PySeries> {
    let mut mask = vec![false; n];
    for (start, end) in spans {
        for i in start..end {
            mask[i] = true;
        }
    }
    let result = Series::new("mask".into(), &mask);
    Ok(PySeries(result))
}

// Span

#[pyclass]
#[derive(Clone)]
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
