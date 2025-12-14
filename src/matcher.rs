#![allow(clippy::upper_case_acronyms)]

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3_polars::PySeries;

use crate::span::Span;

enum Operation {
    Advance(),
    Jump(isize),
    Split(isize, isize),
    Match(),
    PushVar(String),
    BindVar(String),
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
    BINDVAR,
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
                Opcode::PUSHVAR => Operation::PushVar(tuple.get_item(1)?.extract()?),
                Opcode::BINDVAR => Operation::BindVar(tuple.get_item(1)?.extract()?),
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
        let mut stack = Vec::with_capacity(64);
        let mut bindings = HashMap::new();
        let n = self.mask_vec[0].len();
        let last_pc = self.operations.len() - 1;
        stack.push((cursor, 0, HashMap::new()));
        while let Some(task) = stack.pop() {
            let (mut cursor, mut pc, mut starts) = task;
            loop {
                if pc >= last_pc {
                    // Success!
                    if cursor > match_end {
                        match_end = cursor;
                    };
                    break;
                };
                if cursor >= n || !self.mask_vec[pc].get(cursor).unwrap() {
                    // Failure!
                    break;
                };
                match &self.operations[pc] {
                    Operation::Advance() => {
                        cursor += 1;
                        pc += 1;
                    },
                    Operation::Split(offset1, offset2) => {
                        let pc2 = (pc as isize + offset2) as usize;
                        stack.push((cursor, pc2, starts.clone()));
                        pc = (pc as isize + offset1) as usize;
                    },
                    Operation::Jump(offset) => {
                        pc = (pc as isize + offset) as usize;
                    },
                    Operation::PushVar(var_name) => {
                        starts.insert(var_name, cursor);
                        pc += 1;
                    },
                    Operation::BindVar(var_name) => {
                        // this goes with last binding -- do we want first binding instead?
                        let start = starts.remove(var_name).unwrap();
                        bindings.insert(var_name.clone(), Span::new(start, cursor));
                        pc += 1;
                    },
                    Operation::Match() => {
                        // if last_pc is right, then we should never get here
                        // match_end = cmp::max(match_end, sp);
                        unreachable!();
                    },
                }
            }
        }
        if match_end > cursor {
            Ok(Some(Match {
                span: Span::new(match_start, match_end),
                bindings,
            }))
        } else {
            Ok(None)
        }
    }
}
