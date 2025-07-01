// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

mod assoc;
mod matcher;
mod span;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "PanicException",
        <pyo3::panic::PanicException as pyo3::PyTypeInfo>::type_object(py),
    )?;
    m.add_class::<matcher::Opcode>()?;
    m.add_class::<matcher::OpcodeMatcher>()?;
    m.add_class::<span::Span>()?;
    m.add_function(wrap_pyfunction!(span::_to_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(span::py_concordance, m)?)?;
    // m.add_function(wrap_pyfunction!(assoc::loglik, m)?)?;
    Ok(())
}
