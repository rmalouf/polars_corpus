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
- `matcher.py`: Search interface with both simple and CQP query support
- `simple_parser.py`: Simple query language parser (BNCweb-style) that translates to CQP
- `cqp_parser.py`: CQP grammar and expression compilation using pyparsing
- `assoc.py`: Statistical association measures (PMI, log-likelihood, minimum sensitivity)
- `exprs.py`: Polars namespace extensions (`.corpus` for DataFrames/LazyFrames/Expressions)
- `productivity.py`: Corpus productivity metrics and type-token analysis
- `chunk.py`: BIO-tagged chunking and span indexing utilities
- `convert.py`: NLTK corpus conversion to Polars DataFrames
- `io.py`: Text corpus reading with Polars IO plugin integration
- `view.py`: Interactive concordance browser widget for Jupyter notebooks
- `utils.py`: Utility functions
- `_typing.py`: Type definitions

**Rust Layer (`src/`)**:
- `matcher.rs`: High-performance pattern matching engine using finite automata
- `span.rs`: Span handling and concordance generation
- `assoc.rs`: Performance-critical association computations
- `io.rs`: I/O utilities (commented/placeholder code)
- `lib.rs`: PyO3 bindings and module exports

### Key Design Patterns

**Polars Integration**: Uses `@pl.api.register_*_namespace` decorators to extend Polars with `.corpus` methods on DataFrames, LazyFrames, and Expressions.

**Query Languages**: The package supports two query syntaxes:

1. **Simple Query Language** (default, BNCweb-style): User-friendly syntax for corpus searches
   - Case-insensitive by default
   - Wildcards: `?` (single char), `*` (zero or more), `+` (one or more)
   - Alternatives: `[car,truck]`, `neighbo[u,]r`
   - Word sequences: `quick brown fox`
   - Gap tokens: `fox * over` (optional), `fox + over` (required)
   - POS tags: `word_TAG` (word+POS), `_TAG` (POS only)
   - Lemmas: `{lemma}` (all forms), `{lemma/POS}` (with POS constraint)
   - Escaping: `\?` for literal metacharacters

2. **CQP Query Language**: Advanced syntax for linguistic pattern matching
   - Token constraints: `[token="word"]`, `[pos="NOUN"]`
   - Regex patterns: `[c5="AJ.*"]` (adjective patterns)
   - Repetition operators: `+`, `*`, `?`, `{n}`, `{m,n}`
   - Disjunction: `|`
   - Grouping with parentheses

**Translation Architecture**: Simple queries are translated to CQP internally, allowing both syntaxes to share the same matching infrastructure and ensuring consistent behavior.

**Rust-Python Bridge**: Uses `pyo3-polars` for zero-copy data exchange between Python Polars DataFrames and Rust processing.

**Search Results Pattern**: The `SearchResults` class wraps search matches and provides methods for concordance generation, sampling, and filtering without re-executing queries.

## Testing Strategy

Tests are organized by module functionality:
- `test_assoc.py`: Statistical association measures and crosstab functionality
- `test_matcher.py`: CQP query parsing and pattern matching
- `test_simple_query.py`: Simple query language parsing and translation
- `test_spans.py`: Span operations and concordance generation
- `test_convert.py`: Data format conversions
- `test_text_corpus_reader.py`: Corpus reading utilities

## Development Workflow

1. **Rust Changes**: After modifying Rust code, run `make develop` to rebuild the extension
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

Example usage patterns from notebooks:
```python
import polars as pl
import polars_corpus as plc

# Load corpus
c = pl.read_parquet('bnc.parquet')

# Simple query (default, case-insensitive)
r = plc.search(c, 'quick brown fox')

# Simple query with wildcards
r = plc.search(c, '*able')  # Find words ending in "able"
r = plc.search(c, 's?ng')   # Find sing, sang, song, etc.

# Simple query with alternatives
r = plc.search(c, '[car,truck]')  # Find either car or truck

# Simple query with gaps
r = plc.search(c, 'fox + over')  # Find "fox" followed by any word, then "over"

# POS tag searches
r = plc.search(c, 'lights_NN2')  # Find "lights" tagged as NN2
r = plc.search(c, '*ly_AJ0')  # Find adjectives ending in "-ly"
r = plc.search(c, '_PNX')  # Find any reflexive pronoun

# Lemma searches
r = plc.search(c, '{light}')  # Find all forms of lemma "light"
r = plc.search(c, '{light/V}')  # Find verbal forms of "light" (simplified POS)
r = plc.search(c, '{walk}_VBD')  # Find lemma "walk" with exact POS tag VBD
r = plc.search(c, '{eat} * up')  # Find lemma "eat" followed by "up"

# Search in different column (for backward compatibility)
r = plc.search(c, 'NN*', column='pos')  # Find noun tags

# CQP query for advanced patterns
r = plc.search_cqp(c, '[c5="AJ.*"]+ [c5="NN.*"]+')

# Generate concordance
conc = r.concordance('token', 5)
```
- use numpy-style docstrings