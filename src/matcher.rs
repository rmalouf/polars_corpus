#![allow(clippy::upper_case_acronyms)]

use std::collections::HashMap;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_polars::PySeries;

use crate::span::Span;

#[pyclass(module = "polars_corpus")]
#[derive(Clone, Debug)]
pub enum Opcode {
    Token(Vec<u8>),
    Skip(),
    Jump(isize),
    Split(isize, isize),
    Match(),
    PushVar(),
    PopVar(),
    BindVar(String),
    UnBindVar(),
    Fail(),
}

#[pyclass(module = "polars_corpus")]
#[derive(Clone)]
pub struct Match {
    #[pyo3(get)]
    pub span: Span,
    #[pyo3(get)]
    pub bindings: HashMap<String, Span>,
}

#[pymethods]
impl Match {
    #[new]
    fn new(span: Span, bindings: HashMap<String, Span>) -> Self {
        Match { span, bindings }
    }
}

#[pyclass(module = "polars_corpus")]
pub struct OpcodeMatcher {
    opcodes: Vec<Opcode>,
    mask_vec: Vec<BooleanChunked>,
    var_name_cache: Vec<Option<Arc<str>>>,
}

#[pymethods]
impl OpcodeMatcher {
    #[new]
    fn new<'py>(opcodes: Vec<Opcode>, py_masks: &Bound<'py, PyList>) -> PyResult<Self> {
        let mask_vec = py_masks
            .iter()
            .map(|item| -> PyResult<_> {
                let series: PySeries = item.extract()?;
                series
                    .as_ref()
                    .bool()
                    .cloned()
                    .map_err(|_| PyRuntimeError::new_err("all masks must be boolean"))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Build var name cache aligned with opcodes for O(1) lookup by pc
        let var_name_cache: Vec<Option<Arc<str>>> = opcodes
            .iter()
            .map(|op| match op {
                Opcode::BindVar(name) => Some(Arc::from(name.as_str())),
                _ => None,
            })
            .collect();

        Ok(Self {
            opcodes,
            mask_vec,
            var_name_cache,
        })
    }

    fn matchall(&self) -> PyResult<Option<Vec<Match>>> {
        let mut cursor: usize = 0;
        let starts = &self.mask_vec[0];
        let mut matches = Vec::with_capacity(1000);
        while cursor < starts.len() {
            if starts.get(cursor).unwrap()
                && let Some(m) = self._match_opcodes(cursor)?
            {
                cursor = m.span.end;
                matches.push(m);
            } else {
                cursor += 1;
            };
        }

        if matches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(matches))
        }
    }

    fn _match_opcodes(&self, cursor: usize) -> PyResult<Option<Match>> {
        let match_start = cursor;
        let mut match_end = cursor;
        let mut match_bindings: Vec<(Arc<str>, Span)> = Vec::new();
        let mut stack = Vec::with_capacity(64);
        let mut var_stack: Vec<Option<usize>> = Vec::new();
        let mut bindings_stack: Vec<(Arc<str>, Span)> = Vec::new();
        let n = self.mask_vec[0].len();
        stack.push((cursor, 0));
        while let Some(task) = stack.pop() {
            let (mut cursor, mut pc) = task;
            loop {
                if pc >= self.opcodes.len() {
                    break;
                };
                match &self.opcodes[pc] {
                    Opcode::Token(_) | Opcode::Skip() => {
                        if cursor < n && self.mask_vec[pc].get(cursor).unwrap() {
                            cursor += 1;
                            pc += 1;
                        } else {
                            // Failure!
                            break;
                        };
                    },
                    Opcode::Split(offset1, offset2) => {
                        let pc2 = (pc as isize + offset2) as usize;
                        stack.push((cursor, pc2));
                        pc = (pc as isize + offset1) as usize;
                    },
                    Opcode::Jump(offset) => {
                        pc = (pc as isize + offset) as usize;
                    },
                    Opcode::PushVar() => {
                        var_stack.push(Some(cursor));
                        pc += 1;
                    },
                    Opcode::PopVar() => {
                        var_stack.pop();
                        pc += 1;
                    },
                    Opcode::BindVar(_) => {
                        let start = var_stack.pop().unwrap();
                        if let Some(start) = start {
                            let name_rc = self.var_name_cache[pc].as_ref().unwrap().clone();
                            bindings_stack.push((name_rc, Span::new(start, cursor)));
                        }
                        pc += 1;
                    },
                    Opcode::UnBindVar() => {
                        bindings_stack.pop();
                        var_stack.push(None);
                        pc += 1;
                    },
                    Opcode::Match() => {
                        if cursor > match_end {
                            match_end = cursor;
                            match_bindings.clear();
                            match_bindings.extend_from_slice(&bindings_stack);
                        }
                        pc += 1;
                    },
                    Opcode::Fail() => {
                        break;
                    },
                }
            }
        }
        if match_end > match_start {
            let bindings_map: HashMap<String, Span> = match_bindings
                .into_iter()
                .map(|(name, span)| (name.to_string(), span))
                .collect();
            Ok(Some(Match {
                span: Span::new(match_start, match_end),
                bindings: bindings_map,
            }))
        } else {
            Ok(None)
        }
    }
}
