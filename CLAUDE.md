# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`polars-corpus` is a hybrid Python/Rust extension for Polars that provides corpus linguistics and text analysis functionality. The package implements CQP-style (Corpus Query Processor) search capabilities, concordance generation, and statistical association measures for large text corpora.

## Development Commands

### Environment Setup
```bash
# Create virtual environment and install example dependencies
make venv

# Compile and update dependency locks
make locks
```

### Building the Rust Extension
```bash
# Build release version with maturin
make develop

# Build wheel packages
make build
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest python/tests/test_assoc.py

# Run tests with coverage
coverage run -m pytest
coverage report
```

### Type Checking
```bash
mypy python/polars_corpus/
```

## Architecture

### Core Components

**Python Layer (`polars_corpus/`)**:
- `search.py`: SearchResults class and concordance functionality
- `matcher.py`: CQP query parsing and pattern compilation using pyparsing
- `assoc.py`: Statistical association measures (PMI, log-likelihood, minimum sensitivity)
- `exprs.py`: Polars namespace extensions (`.corpus` for DataFrames/LazyFrames/Expressions)
- `productivity.py`: Corpus productivity metrics and type-token analysis

**Rust Layer (`src/`)**:
- `matcher.rs`: High-performance pattern matching engine using finite automata
- `span.rs`: Span handling and concordance generation 
- `assoc.rs`: Performance-critical association computations
- `lib.rs`: PyO3 bindings and module exports

### Key Design Patterns

**Polars Integration**: Uses `@pl.api.register_*_namespace` decorators to extend Polars with `.corpus` methods on DataFrames, LazyFrames, and Expressions.

**CQP Query Language**: Implements a subset of CQP syntax for linguistic pattern matching:
- Token constraints: `[token="word"]`, `[pos="NOUN"]`
- Regex patterns: `[c5="AJ.*"]` (adjective patterns)
- Repetition operators: `+`, `*`, `?`, `{n}`, `{m,n}`
- Disjunction: `|`
- Grouping with parentheses

**Rust-Python Bridge**: Uses `pyo3-polars` for zero-copy data exchange between Python Polars DataFrames and Rust processing.

**Search Results Pattern**: The `SearchResults` class wraps search matches and provides methods for concordance generation, sampling, and filtering without re-executing queries.

## Testing Strategy

Tests are organized by module functionality:
- `test_assoc.py`: Statistical association measures and crosstab functionality
- `test_matcher.py`: CQP query parsing and pattern matching
- `test_spans.py`: Span operations and concordance generation
- `test_convert.py`: Data format conversions
- `test_text_corpus_reader.py`: Corpus reading utilities

## Development Workflow

1. **Rust Changes**: After modifying Rust code, run `make debug` to rebuild and copy the shared library
2. **Python Changes**: No rebuild needed, changes are immediately available
3. **Query Language**: Test CQP patterns in notebooks before implementing new syntax
4. **Performance**: Use the example notebooks with large corpora (BNC) to benchmark changes

## Corpus Data Format

The package expects Polars DataFrames with linguistic annotation columns:
- `token`: Word tokens
- `pos`/`c5`: Part-of-speech tags
- `mode`: Written/spoken distinction
- `fileid`: Source file identifier
- Additional annotation columns as needed

Example data loading pattern from notebooks:
```python
import polars as pl
import polars_corpus as plc

# Load corpus
c = pl.read_parquet('bnc.parquet')

# Search with CQP query
r = plc.search(c, '[c5="AJ.*"]+ [c5="NN.*"]+')  

# Generate concordance
conc = r.concordance('token', 5)
```
- use numpy-style docstrings