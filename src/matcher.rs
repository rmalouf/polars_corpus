#![allow(clippy::upper_case_acronyms)]
// #![allow(clippy::unused_unit)]
// #![warn(unused_variables)]
// #![warn(dead_code)]

use std::cmp;

use ndarray::s;
use numpy::PyReadonlyArray2;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
// use pyo3_polars::PyExpr;
use pyo3_polars::PySeries;

#[pyfunction]
pub fn _to_spans(n: usize, spans: Vec<(usize, usize)>) -> PyResult<PySeries> {
    let mut span_vec = vec!["O"; n];
    for (start, end) in spans {
        if (start > n) | (end > n) {
            return Err(PyValueError::new_err("index out of bounds"));
        } else {
            span_vec[start] = "B";
            if start + 1 < end {
                span_vec[start + 1..end].fill("I");
            }
        }
    }
    let result = Series::new("spans".into(), &span_vec);
    Ok(PySeries(result))
}

#[pyfunction]
pub fn _make_spans_mask(n: usize, spans: Vec<(usize, usize)>) -> PyResult<PySeries> {
    let mut mask = vec![false; n];
    for (start, end) in spans {
        for i in start..end {
            mask[i] = true;
        }
    }
    let result = Series::new("mask".into(), &mask);
    Ok(PySeries(result))
}

#[pyclass]
enum Instruction {
    Token(),
    Jump(isize),
    Split(isize, isize),
    Skip(),
    Match(),
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum Opcode {
    TOKEN,
    JUMP,
    SPLIT,
    SKIP,
    MATCH,
}

#[pyclass]
pub struct OpcodeMatcher {
    instructions: Vec<Instruction>,
    mask_array: ndarray::Array2<bool>,
}

fn parse_opcodes<'py>(opcodes: &Bound<'py, PyList>) -> PyResult<Vec<Instruction>> {
    let mut instructions = Vec::new();
    for item in opcodes.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        let py_opcode = tuple.get_item(0)?;
        let opcode = py_opcode.extract::<Opcode>()?;
        let instruction = {
            match opcode {
                Opcode::TOKEN => Instruction::Token(),
                Opcode::MATCH => Instruction::Match(),
                Opcode::SKIP => Instruction::Skip(),
                Opcode::JUMP => {
                    let offset: isize = tuple.get_item(1)?.extract()?;
                    Instruction::Jump(offset)
                },
                Opcode::SPLIT => {
                    let offset1: isize = tuple.get_item(1)?.extract()?;
                    let offset2: isize = tuple.get_item(2)?.extract()?;
                    Instruction::Split(offset1, offset2)
                },
            }
        };
        instructions.push(instruction);
    }
    // be sure to check in advance to make sure all branches lead to valid pcs
    Ok(instructions)
}

#[pymethods]
impl OpcodeMatcher {
    #[new]
    fn new<'py>(opcodes: &Bound<'py, PyList>, mask: PyReadonlyArray2<bool>) -> PyResult<Self> {
        let mask_array = mask.as_array().to_owned();
        let instructions = parse_opcodes(opcodes)?;
        Ok(OpcodeMatcher {
            instructions,
            mask_array,
        })
    }

    fn matchall(&self) -> PyResult<Option<Vec<(usize, usize)>>> {
        let mut cursor: usize = 0;
        let starts = self.mask_array.slice(s![0, ..]);
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(1000);
        while cursor < starts.len() {
            if starts[cursor]
                && let Some(match_end) = self._match_opcodes(cursor)?
            {
                spans.push((cursor, match_end));
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
        let n = self.mask_array.len_of(ndarray::Axis(1));
        let last_pc = self.instructions.len() - 1;
        stack.push((cursor, 0));
        while let Some(task) = stack.pop() {
            let (mut sp, mut pc) = task;
            loop {
                if pc == last_pc {
                    // Success!
                    if sp > match_end {
                        match_end = sp;
                    };
                    break;
                };
                if sp >= n || !self.mask_array[[pc, sp]] {
                    // Failure!
                    break;
                };
                match self.instructions[pc] {
                    Instruction::Token() | Instruction::Skip() => {
                        sp += 1;
                        pc += 1;
                    },
                    Instruction::Split(offset1, offset2) => {
                        let pc2 = (pc as isize + offset2) as usize;
                        stack.push((sp, pc2));
                        pc = (pc as isize + offset1) as usize;
                    },
                    Instruction::Jump(offset) => {
                        pc = (pc as isize + offset) as usize;
                    },
                    Instruction::Match() => {
                        // we should never get here
                        match_end = cmp::max(match_end, sp);
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
