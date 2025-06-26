// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

mod matcher;
mod span;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

// #[pyfunction]
// fn _set_vars(n: usize, spans: Vec<(usize, usize)>, values: Vec<String>) -> PyResult<PySeries> {
//     let mut span_vec : Vec<Option<Vec<String>>> = vec![None; n];
//     for ((start, end), value) in spans.iter().zip(values) {
//         if (*start > n) || (*end > n) {
//             return Err(PyValueError::new_err("index out of bounds"));
//         } else {
//             for i in *start..*end {
//                 span_vec[i] = Some(value.clone());
//             }
//         }
//     }
//
//     let mut builder = ListStringChunkedBuilder::new("example".into(), n, n*10);
//
//     for item in span_vec {
//         match item {
//             Some(strings) => {
//                 builder.append_values_iter(strings.iter().map(|s| s.as_str()));
//             }
//             None => {
//                 builder.append_null();
//             }
//         }
//     }
//
//     Ok(PySeries(builder.finish().into_series()))
// }

#[pymodule]
fn _internal(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "PanicException",
        <pyo3::panic::PanicException as pyo3::PyTypeInfo>::type_object(py),
    )?;
    m.add_function(wrap_pyfunction!(span::_to_spans, m)?)?;
    // m.add_function(wrap_pyfunction!(_set_vars, m)?)?;
    m.add_function(wrap_pyfunction!(span::_make_spans_mask, m)?)?;
    m.add_class::<span::Span>()?;
    m.add_class::<matcher::Opcode>()?;
    m.add_class::<matcher::OpcodeMatcher>()?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
