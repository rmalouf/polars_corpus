use std::fs::File;
use std::io::{BufRead, BufReader};

use flate2::read::MultiGzDecoder;
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

/// A compressed format a corpus file may be stored in.
enum Compression {
    None,
    Gzip,
    Zstd,
}

impl Compression {
    /// The format the bytes a file opens with identify, or `None` if they
    /// identify no format. Fewer bytes than a magic number simply match none.
    fn detect(magic: &[u8]) -> Self {
        if magic.starts_with(&[0x1f, 0x8b]) {
            Compression::Gzip
        } else if magic.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
            Compression::Zstd
        } else {
            Compression::None
        }
    }
}

/// Open `path`, decoding it if it is compressed.
///
/// What the file opens with decides, not what it is named, because corpora are
/// distributed under every convention and none: a gzipped file reads the same
/// whether it is called `.wlp.gz`, `.wlp`, or `w_acad_1990`. `fill_buf` only
/// peeks at those bytes, so an uncompressed file goes on to be read through
/// the very buffer they were seen in.
fn open_maybe_compressed(path: &str) -> std::io::Result<Box<dyn BufRead + Send + Sync>> {
    let mut file = BufReader::new(File::open(path)?);
    let compression = Compression::detect(file.fill_buf()?);
    // A decoder is only a Read, so it needs a buffer of its own for the
    // line-at-a-time reads to draw on.
    Ok(match compression {
        Compression::None => Box::new(file),
        // Multi, because a `.gz` is often several members concatenated.
        Compression::Gzip => Box::new(BufReader::new(MultiGzDecoder::new(file))),
        Compression::Zstd => Box::new(BufReader::new(zstd::Decoder::with_buffer(file)?)),
    })
}

/// An open COCA/COHA `.wlp` file, iterating over frames of `batch_size` tokens.
///
/// A `##` line naming a text id starts the tokens belonging to it; every other
/// line is one token as three tab-separated fields. Bytes that are not valid
/// UTF-8 are replaced rather than raising, because real COCA files carry a few.
///
/// A gzip- or zstd-compressed file is decoded as it is read, recognized by its
/// leading bytes rather than its name.
///
/// Only a batch is held at a time, so a query that wants the first rows of a
/// corpus never parses the rest of the file.
#[pyclass]
pub struct WlpFileReader {
    path: String,
    batch_size: usize,
    reader: Box<dyn BufRead + Send + Sync>,
    /// The text id in force, carried across batches.
    text_id: String,
    line: Vec<u8>,
    lineno: usize,
}

#[pymethods]
impl WlpFileReader {
    #[new]
    fn new(path: &str, batch_size: usize) -> PyResult<Self> {
        let reader = open_maybe_compressed(path).map_err(|e| io_err(path, e))?;
        Ok(WlpFileReader {
            path: path.to_owned(),
            batch_size,
            reader,
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
            let line = String::from_utf8_lossy(&self.line);
            let line = line.trim_end_matches('\n');
            if let Some(rest) = line.strip_prefix("##") {
                let (id, rest) = rest.split_once('\t').unwrap_or((rest, ""));
                if rest.trim().is_empty() {
                    self.text_id.clear();
                    self.text_id.push_str(id.trim());
                    continue;
                }
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
