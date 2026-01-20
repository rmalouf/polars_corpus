use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_polars::PySeries;

use crate::span::Span;

#[pyclass(module = "polars_corpus")]
#[derive(Clone)]
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

struct MatchBuffers {
    stack: Vec<(usize, usize)>,
    var_stack: Vec<Option<usize>>,
    bindings_stack: Vec<(usize, Span)>,
    match_bindings: Vec<(usize, Span)>,
}

impl MatchBuffers {
    fn new() -> Self {
        Self {
            stack: Vec::new(),
            var_stack: Vec::new(),
            bindings_stack: Vec::new(),
            match_bindings: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.stack.clear();
        self.var_stack.clear();
        self.bindings_stack.clear();
        self.match_bindings.clear();
    }
}

#[pyclass(module = "polars_corpus")]
pub struct OpcodeMatcher {
    opcodes: Vec<Opcode>,
    mask_vec: Vec<BooleanChunked>,
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

        Ok(Self { opcodes, mask_vec })
    }

    fn matchall(&self) -> PyResult<Option<Vec<Match>>> {
        let mut cursor: usize = 0;
        let starts = &self.mask_vec[0];
        let mut matches = Vec::with_capacity(1000);
        let mut buffers = MatchBuffers::new();
        while cursor < starts.len() {
            if starts.get(cursor).unwrap()
                && let Some(m) = self._match_opcodes(cursor, &mut buffers)?
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
}

impl OpcodeMatcher {
    fn _match_opcodes(&self, cursor: usize, buffers: &mut MatchBuffers) -> PyResult<Option<Match>> {
        let match_start = cursor;
        let mut match_end = cursor;

        buffers.clear();

        let n = self.mask_vec[0].len();
        buffers.stack.push((cursor, 0));
        while let Some(task) = buffers.stack.pop() {
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
                        buffers.stack.push((cursor, pc2));
                        pc = (pc as isize + offset1) as usize;
                    },
                    Opcode::Jump(offset) => {
                        pc = (pc as isize + offset) as usize;
                    },
                    Opcode::Match() => {
                        if cursor > match_end {
                            match_end = cursor;
                            buffers.match_bindings.clone_from(&buffers.bindings_stack);
                        }
                        pc += 1;
                    },
                    Opcode::PushVar() => {
                        buffers.var_stack.push(Some(cursor));
                        pc += 1;
                    },
                    Opcode::PopVar() => {
                        buffers.var_stack.pop();
                        pc += 1;
                    },
                    Opcode::BindVar(_) => {
                        let start = buffers.var_stack.pop().unwrap();
                        if let Some(start) = start {
                            buffers.bindings_stack.push((pc, Span::new(start, cursor)));
                        }
                        pc += 1;
                    },
                    Opcode::UnBindVar() => {
                        buffers.bindings_stack.pop();
                        buffers.var_stack.push(None);
                        pc += 1;
                    },
                    Opcode::Fail() => {
                        break;
                    },
                }
            }
        }
        if match_end > match_start {
            let bindings_map: HashMap<String, Span> = buffers
                .match_bindings
                .iter()
                .map(|(pc, span)| {
                    // Look up variable name from the opcode
                    let name = match &self.opcodes[*pc] {
                        Opcode::BindVar(name) => name.clone(),
                        _ => unreachable!(),
                    };
                    (name, span.clone())
                })
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
