// use polars::datatypes::BooleanChunked;
// use pyo3::exceptions::PyRuntimeError;
// use pyo3::prelude::*;
// use pyo3::types::PyList;
// use pyo3_polars::PySeries;
// use std::fs::File;
// use std::io::{BufReader, Lines};
// use std::io::prelude::*;
// use std::iter::Peekable;
// use std::path::Path;
// use std::str::SplitWhitespace;
//
// // #[pyclass(module = "polars_corpus")]
// pub struct TextCorpusReader {
//     file : File,
//     lines_iter : Lines<BufReader<&File>>,
//     token_iter: Peekable<SplitWhitespace<'static>>,
//     bos : bool,
// }
//
// // #[pymethods]
// impl TextCorpusReader {
//     // #[new]
//     fn new<'py>(filename: &str) -> PyResult<Self> {
//         let path = Path::new(filename);
//         let file = match File::open(&path) {
//             Err(why) => panic!("couldn't open {}: {}", display, why),
//             Ok(file) => file,
//         };
//         let lines_iter = BufReader::new(&file).lines();
//         let line_buffer
//         let bos = true;
//         Ok(TextCorpusReader { file, lines_iter, bos })
//     }
//
//     fn __iter__(&self) -> PyResult<&Self> {
//         Ok(self)
//     }
//
//     fn __next__(&mut self) -> PyResult<Option<String>> {
//         if self.token_iter.peek().is_none() {
//             match self.lines_iter.next() {
//                 Some(next_line) => {
//                     self.token_iter = next_line?.split_whitespace().peekable();
//                     self.bos = true;
//                 },
//                 None => { return Ok(None); }
//             };
//         };
//         let next_token = self.token_iter.next().unwrap();
//
//         self.bos = false;
//
//
//         Ok(result)
//     }
//
//     // for line in open(file, "rt"):
//     // if line != "\n":
//     //     bos = True
//     // tokens = line.strip().split()
//     // for token in tokens:
//     //     try:
//     //     tok, tag = token.rsplit("/", 1)
//     // except ValueError:
//     //     raise ValueError(f'Malformed token "{token}"')
//     //                      yield {"token": tok, "tag": tag, "sent": "B" if bos else "I"}
//     //                      bos = False
//
// }
//
// // let operations = crate::matcher::compile_operations(opcodes)?;
// //         let mut mask_vec = Vec::with_capacity(py_masks.len());
// //         for m in py_masks.iter() {
// //             let series: PySeries = m.extract()?;
// //             let data = series
// //                 .as_ref()
// //                 .bool()
// //                 .map_err(|_| PyRuntimeError::new_err("all masks must be boolean"))?;
// //             mask_vec.push(data.clone());
// //         }
// //         Ok(crate::matcher::OpcodeMatcher {
// //             operations,
// //             mask_vec,
// //         })
// //
// //
// //         // Return None to signal StopIteration
// //         fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<usize> {
// //             if slf.current < slf.stop {
// //                 let v = slf.current;
// //                 slf.current += 1;
// //                 Some(v)
// //             } else {
// //                 None
// //             }
// //         }
// //
// //     }
