#![allow(clippy::upper_case_acronyms)]

use polars::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3_polars::PySeries;

use crate::span::Span;

#[pyclass]
enum Operation {
    Advance(),
    Jump(isize),
    Split(isize, isize),
    Match(),
}

#[pyclass(module = "polars_corpus")]
#[derive(Copy, Clone)]
pub enum Opcode {
    TOKEN,
    JUMP,
    SPLIT,
    SKIP,
    MATCH,
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

    fn matchall(&self) -> PyResult<Option<Vec<Span>>> {
        let mut cursor: usize = 0;
        let starts = &self.mask_vec[0];
        let mut spans: Vec<Span> = Vec::with_capacity(1000);
        while cursor < starts.len() {
            if unsafe { starts.value_unchecked(cursor) }
                && let Some(match_end) = self._match_opcodes(cursor)?
            {
                spans.push(Span::new(cursor, match_end));
                cursor = match_end;
            } else {
                cursor += 1;
            };
        }

        if spans.is_empty() {
            Ok(None)
        } else {
            Ok(Some(spans))
        }
    }

    fn _match_opcodes(&self, cursor: usize) -> PyResult<Option<usize>> {
        let mut match_end = cursor;
        let mut stack = Vec::with_capacity(64);
        let n = self.mask_vec[0].len();
        let last_pc = self.operations.len() - 1;
        stack.push((cursor, 0));
        while let Some(task) = stack.pop() {
            let (mut sp, mut pc) = task;
            loop {
                if pc >= last_pc {
                    // Success!
                    if sp > match_end {
                        match_end = sp;
                    };
                    break;
                };
                // if sp >= n || unsafe { !self.mask_vec[pc].value_unchecked(sp) } {
                if sp >= n || !self.mask_vec[pc].get(sp).unwrap() {
                    // Failure!
                    break;
                };
                match self.operations[pc] {
                    Operation::Advance() => {
                        sp += 1;
                        pc += 1;
                    },
                    Operation::Split(offset1, offset2) => {
                        let pc2 = (pc as isize + offset2) as usize;
                        stack.push((sp, pc2));
                        pc = (pc as isize + offset1) as usize;
                    },
                    Operation::Jump(offset) => {
                        pc = (pc as isize + offset) as usize;
                    },
                    Operation::Match() => {
                        // we should never get here!!!
                        // match_end = cmp::max(match_end, sp);
                        break;
                    },
                }
            }
        }
        if match_end > cursor {
            Ok(Some(match_end))
        } else {
            Ok(None)
        }
    }
}
