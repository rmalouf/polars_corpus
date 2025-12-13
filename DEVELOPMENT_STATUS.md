let's # Development Status - polars-corpus

**Last Updated:** 2025-01-12
**Version:** 0.1.4-pre
**Status:** Pre-release, production-quality core

---

## Executive Summary

polars-corpus is a mature, well-engineered corpus linguistics toolkit with ~3,900 lines of clean Python/Rust code. Core functionality (search, concordancing, statistics) is production-ready. Simple query language is 75% complete (Phase 3 partial). Main gaps: user documentation (README empty) and proximity operators (Phase 4).

---

## Codebase Metrics

- **Python:** ~2,500 lines (15 modules)
- **Rust:** ~900 lines (6 files)
- **Tests:** ~500 lines (8 test files, 44 simple query tests)
- **Examples:** 15+ files including BNC corpus examples
- **Documentation:** 5 developer docs, 0 user guides

---

## Feature Completeness

### ✅ Production-Ready (100%)

1. **Core Search Engine**
   - NFA-based pattern matcher (Rust)
   - CQP query language (full support)
   - SearchResults API (concordance, collocates, view)

2. **Concordancing**
   - KWIC generation (Rust-optimized)
   - Interactive Jupyter widget
   - Pagination, sorting, filtering, sampling

3. **Statistical Analysis**
   - Association measures: PMI, log-likelihood, min sensitivity, Welch's t-test
   - Lexical diversity: TTR, MSTTR, Yule's K, MTLD
   - All implemented in Rust for performance

4. **Data I/O**
   - Text corpus reader (eager/lazy)
   - NLTK converter
   - BIO sentence tagging

5. **Polars Integration**
   - `.corpus` expression namespace
   - Plugin architecture
   - Zero-copy data exchange

### 🚧 Partially Complete (75%)

**Simple Query Language** (BNCweb-style)

Phase 1-3 Complete:
- ✅ Basic words, case-insensitive
- ✅ Wildcards: `?`, `*`, `+`
- ✅ Alternatives: `[a,b,c]`
- ✅ Word sequences
- ✅ Gap tokens: `*`, `+`, `++`, `***`, `+++**`
- ✅ POS tags: `word_TAG`, `_TAG`, `_{VERB}`
- ✅ Lemmas: `{lemma}`, `{lemma/POS}`, `{lemma}_TAG`
- ✅ Regex groups: `(pattern)?+*{m,n}`
- ✅ Disjunction: `(a | b | c)`

Phase 4 Not Started:
- ❌ Proximity operators: `<<s>>`, `<<3>>`, `<<5<<`, `>>5>>`
- ❌ Embedded alternatives: `neighbo[u,]r` (workaround available)

### ❌ Missing (Critical)

1. **User Documentation**
   - README.md is empty
   - No getting started guide
   - No API reference
   - No installation instructions

2. **Testing**
   - No Rust unit tests (Python integration only)
   - Examples not tested automatically

---

## Recent Development Focus

### Last 3 Commits (Jan 2025)
1. Bump version to 0.1.4-pre, update dependencies
2. Refactor documentation and format Python code
3. Refactor collocates method and update collocation example

### Untracked Work in Progress
- `chunk.py` - BIO tag chunking utilities (~92 lines, ready to commit)
- `SIMPLE_QUERY_STATUS.md` - Implementation tracking doc
- `Simple_query_language.pdf` - BNCweb reference
- `io.rs.hold`, `lib.rs.hold` - Old backups (deletable)

### Recent Achievements
- Clean refactor: removed AST layer from simple parser, direct CQP generation
- Phase 3 features: regex groups, POS/lemma support, consecutive gaps
- Upgraded to Python 3.10+, Rust Edition 2024
- Comprehensive test suite (44 tests verify actual matched content)

---

## Architecture Strengths

1. **Performance-First Design**
   - Rust for hot paths (matching, stats)
   - Zero-copy pyo3-polars bridge
   - Iterator-based, minimal allocations
   - Lazy evaluation support

2. **Code Quality**
   - Clean separation of concerns
   - Full type hints, strict mypy
   - Modern best practices
   - "Minimal defensive programming" principle

3. **Integration**
   - Extends Polars via `.corpus` namespace
   - Plugin architecture for custom expressions
   - NLTK compatibility

---

## Known Limitations

1. **Simple Query Language**
   - No proximity operators yet (Phase 4)
   - Embedded alternatives need workaround: `(neighbour|neighbor)` instead of `neighbo[u,]r`

2. **Documentation**
   - Developer docs excellent (CLAUDE.md, QUERY_LANGUAGE.md)
   - User docs non-existent (README empty)
   - No API reference

3. **Testing**
   - Python integration tests comprehensive
   - Rust unit tests missing
   - No performance benchmarks

4. **Release Status**
   - Still pre-release (0.1.4-pre)
   - No published wheels
   - No PyPI release

---

## Priority Roadmap

### P0 (Blocker for 0.1.4 Release)
1. Write README.md (installation, quick start, examples)
2. Commit pending work (`chunk.py`, status docs)
3. Clean up holdover files (`*.hold`)

### P1 (Blocker for 1.0)
4. Implement Phase 4 proximity operators
5. Generate API documentation (Sphinx/MkDocs)
6. Add tutorial/walkthrough
7. Release 0.1.4, then 1.0

### P2 (Quality Improvements)
8. Add Rust unit tests
9. Performance benchmarks
10. More introductory examples
11. Publish to PyPI

---

## Target Audience

- **Primary:** Linguistics students and researchers
- **Secondary:** Data scientists working with text corpora
- **Use cases:** 100M+ word corpora on 16GB memory

---

## Dependencies Management

- **Runtime:** polars ≥1.34, pyparsing ≥3.2, nltk ≥3.9, ipywidgets ≥8.1
- **Build:** maturin, pyo3 0.26, pyo3-polars 0.25
- **Rust:** polars 0.52, statrs 0.18, itertools 0.14
- **Dev:** pytest, mypy, ruff, jupyterlab
- **Managed via:** uv with 3-level requirements (runtime, dev, examples)

---

## Build & Test Commands

```bash
# Development
make develop      # Rebuild after Rust changes
ruff format       # Format Python
ruff check        # Lint Python
cargo fmt         # Format Rust
cargo clippy      # Lint Rust
mypy python/polars_corpus/

# Testing
pytest            # Python changes don't need rebuild

# Dependencies
make locks        # Regenerate lock files
make venv         # Sync virtual environment

# Release
make build        # Build wheels (M4 + x86_64)
```

---

## Design Notes

### Crosstab/Association Measures API Refactor ✅ IMPLEMENTED

**Completed:** 2025-01 (on `crosstab-api` branch)

The API has been successfully refactored to use a struct-based design for cleaner association measure computation.

**New API:**

1. `crosstab()` bundles frequencies into a struct column:
   ```python
   ct = crosstab(df, "word", "collocate")
   # Returns: word, collocate, freqs:{f12, f1, f2, n}

   # Optional: customize struct column name
   ct = crosstab(df, "word", "collocate", freqs_name="counts")
   ```

2. Association measures via expression namespace:
   ```python
   ct.with_columns(
       pl.col("freqs").corpus.loglik().alias("ll"),
       pl.col("freqs").corpus.pmi().alias("pmi"),
       pl.col("freqs").corpus.minsens().alias("minsens"),
   )
   ```

3. Standalone functions still work with explicit column names:
   ```python
   loglik("f12", "f1", "f2", "n")  # explicit columns
   ```

**Benefits:**
- Cleaner API: no need to repeatedly specify frequency column names
- Type safety: struct ensures all required fields are present
- Flexibility: struct can be unnested if needed, or renamed via `freqs_name` parameter
- Backwards compatible: standalone functions with explicit column names still work

---

## Notes

- Simple query → CQP translation is elegant (no intermediate AST)
- Test quality is excellent (verify actual matched content, not just counts)
- Code is ready for production use in core areas
- Main barrier to 1.0 is documentation and proximity operators
- Architecture choices (Rust + Polars + pyo3) are sound for performance goals
