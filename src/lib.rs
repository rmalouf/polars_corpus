// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

mod expressions;
use pyo3::prelude::*;
// use pyo3::types::{PyList, PyTuple};
use polars::prelude::*;
use pyo3_polars::{PolarsAllocator, PySeries};

#[pyfunction]
fn _with_spans(n: usize,
    spans: Vec<(usize, usize)>
) -> PyResult<PySeries> {
    let mut span_idx = vec!["O"; n];
    for (start, end) in spans {
        span_idx[start] = "B";
        for i in start+1..end {
            span_idx[i] = "I";
        }
    }
    let result = Series::new("spans".into(), &span_idx);
    return Ok(PySeries(result));
}

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(_with_spans, m)?)?;
    Ok(())
}



#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
