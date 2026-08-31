use std::fs::File;
use std::io::{BufRead, BufReader};

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;

/// Keep the error kind, so a missing file still raises FileNotFoundError, and
/// name the file, which the OS error does not.
fn io_err(path: &str, e: std::io::Error) -> PyErr {
    std::io::Error::new(e.kind(), format!("{path}: {e}")).into()
}

/// An open COCA/COHA `.wlp` file, iterating over frames of `batch_size` tokens.
///
/// A `##` line names the text the tokens after it belong to; every other line
/// is one token as three tab-separated fields. Bytes that are not valid UTF-8
/// are replaced rather than raising, because real COCA files carry a few.
///
/// Only a batch is held at a time, so a query that wants the first rows of a
/// corpus never parses the rest of the file.
#[pyclass]
pub struct WlpFileReader {
    path: String,
    batch_size: usize,
    reader: BufReader<File>,
    /// The text id in force, carried across batches.
    text_id: String,
    line: Vec<u8>,
    lineno: usize,
}

#[pymethods]
impl WlpFileReader {
    #[new]
    fn new(path: &str, batch_size: usize) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| io_err(path, e))?;
        Ok(WlpFileReader {
            path: path.to_owned(),
            batch_size,
            reader: BufReader::new(file),
            text_id: String::new(),
            line: Vec::new(),
            lineno: 0,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next batch of tokens, or None once the file is spent.
    fn __next__(&mut self) -> PyResult<Option<PyDataFrame>> {
        let mut tokens = StringChunkedBuilder::new("token".into(), self.batch_size);
        let mut lemmas = StringChunkedBuilder::new("lemma".into(), self.batch_size);
        let mut tags = StringChunkedBuilder::new("pos".into(), self.batch_size);
        let mut ids = StringChunkedBuilder::new("file_id".into(), self.batch_size);

        let mut rows = 0;
        while rows < self.batch_size {
            self.line.clear();
            let read = self
                .reader
                .read_until(b'\n', &mut self.line)
                .map_err(|e| io_err(&self.path, e))?;
            if read == 0 {
                break;
            }
            self.lineno += 1;
            // Trimming drops the newline and, as in Python, the padding tabs a
            // header line ends with.
            let line = String::from_utf8_lossy(&self.line);
            let line = line.trim();
            if let Some(id) = line.strip_prefix("##") {
                self.text_id.clear();
                self.text_id.push_str(id.trim_start_matches('#'));
                continue;
            }
            let mut fields = line.split('\t');
            match (fields.next(), fields.next(), fields.next(), fields.next()) {
                (Some(token), Some(lemma), Some(pos), None) => {
                    tokens.append_value(token);
                    lemmas.append_value(lemma);
                    tags.append_value(pos);
                    ids.append_value(&self.text_id);
                },
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "{}:{}: expected three tab-separated fields, got \"{line}\"",
                        self.path, self.lineno
                    )));
                },
            }
            rows += 1;
        }

        if rows == 0 {
            return Ok(None);
        }
        let df = DataFrame::new_infer_height(vec![
            tokens.finish().into_column(),
            lemmas.finish().into_column(),
            tags.finish().into_column(),
            ids.finish().into_column(),
        ])
        .map_err(PyPolarsErr::from)?;
        Ok(Some(PyDataFrame(df)))
    }
}
