#![allow(clippy::upper_case_acronyms)]

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3_polars::PySeries;

use crate::span::Span;

#[derive(Debug)]
enum Operation {
    Advance(),
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
pub enum Opcode {
    TOKEN,
    JUMP,
    SPLIT,
    SKIP,
    MATCH,
    PUSHVAR,
    POPVAR,
    BINDVAR,
    UNBINDVAR,
    FAIL,
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

fn compile_operations(opcodes: &Bound<PyList>) -> PyResult<Vec<Operation>> {
    let mut operations = Vec::new();
    for (pc, item) in opcodes.iter().enumerate() {
        let tuple = item.downcast::<PyTuple>()?;
        let py_opcode = tuple.get_item(0)?;
        let opcode = py_opcode.extract::<Opcode>()?;
        let operation = {
            match opcode {
                Opcode::TOKEN | Opcode::SKIP => Operation::Advance(),
                Opcode::MATCH => Operation::Match(),
                Opcode::JUMP => {
                    let offset: isize = tuple.get_item(1)?.extract()?;
                    let target = pc.strict_add_signed(offset);
                    if target >= opcodes.len() {
                        return Err(PyRuntimeError::new_err("JMP target out of bounds"));
                    }
                    Operation::Jump(offset)
                },
                Opcode::SPLIT => {
                    let offset1: isize = tuple.get_item(1)?.extract()?;
                    let offset2: isize = tuple.get_item(2)?.extract()?;
                    let target1 = pc.strict_add_signed(offset1);
                    let target2 = pc.strict_add_signed(offset2);
                    if target1 >= opcodes.len() || target2 >= opcodes.len() {
                        return Err(PyRuntimeError::new_err("SPLIT target out of bounds"));
                    }
                    Operation::Split(offset1, offset2)
                },
                Opcode::PUSHVAR => Operation::PushVar(),
                Opcode::POPVAR => Operation::PopVar(),
                Opcode::BINDVAR => Operation::BindVar(tuple.get_item(1)?.extract()?),
                Opcode::UNBINDVAR => Operation::UnBindVar(),
                Opcode::FAIL => Operation::Fail(),
            }
        };
        operations.push(operation);
    }
    Ok(operations)
}

#[pyclass(module = "polars_corpus")]
pub struct OpcodeMatcher {
    operations: Vec<Operation>,
    mask_vec: Vec<BooleanChunked>,
}

#[pymethods]
impl OpcodeMatcher {
    #[new]
    fn new<'py>(opcodes: &Bound<'py, PyList>, py_masks: &Bound<'py, PyList>) -> PyResult<Self> {
        let operations = compile_operations(opcodes)?;
        let mut mask_vec = Vec::with_capacity(py_masks.len());
        for m in py_masks.iter() {
            let series: PySeries = m.extract()?;
            let data = series
                .as_ref()
                .bool()
                .map_err(|_| PyRuntimeError::new_err("all masks must be boolean"))?;
            mask_vec.push(data.clone());
        }
        Ok(OpcodeMatcher {
            operations,
            mask_vec,
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
                matches.push(m.clone());
                cursor = m.span.end;
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
        let mut match_bindings = Vec::new();
        let mut stack = Vec::with_capacity(64);
        let mut var_stack: Vec<Option<usize>> = Vec::new();
        let mut bindings_stack = Vec::new();
        let n = self.mask_vec[0].len();
        stack.push((cursor, 0));
        while let Some(task) = stack.pop() {
            let (mut cursor, mut pc) = task;
            loop {
                if pc >= self.operations.len() {
                    break;
                };
                match &self.operations[pc] {
                    Operation::Advance() => {
                        if cursor < n && self.mask_vec[pc].get(cursor).unwrap() {
                            cursor += 1;
                            pc += 1;
                        } else {
                            // Failure!
                            break;
                        };
                    },
                    Operation::Split(offset1, offset2) => {
                        let pc2 = (pc as isize + offset2) as usize;
                        stack.push((cursor, pc2));
                        pc = (pc as isize + offset1) as usize;
                    },
                    Operation::Jump(offset) => {
                        pc = (pc as isize + offset) as usize;
                    },
                    Operation::PushVar() => {
                        var_stack.push(Some(cursor));
                        pc += 1;
                    },
                    Operation::PopVar() => {
                        var_stack.pop();
                        pc += 1;
                    },
                    Operation::BindVar(var_name) => {
                        let start = var_stack.pop().unwrap();
                        if let Some(start) = start {
                            bindings_stack.push((var_name.clone(), Span::new(start, cursor)));
                        }
                        pc += 1;
                    },
                    Operation::UnBindVar() => {
                        bindings_stack.pop();
                        var_stack.push(None);
                        pc += 1;
                    },
                    Operation::Match() => {
                        if cursor > match_end {
                            match_end = cursor;
                            match_bindings = bindings_stack.clone();
                        }
                        pc += 1;
                    },
                    Operation::Fail() => {
                        break;
                    },
                }
            }
        }
        if match_end > match_start {
            let bindings_map: HashMap<_, _> = match_bindings.into_iter().collect();
            Ok(Some(Match {
                span: Span::new(match_start, match_end),
                bindings: bindings_map,
            }))
        } else {
            Ok(None)
        }
    }
}
