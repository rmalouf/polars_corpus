# Development Status - polars-corpus

**Last updated:** 2026-08-16
**Version:** 0.2.0-pre
**Status:** Pre-release; core is stable, not yet published to PyPI

---

## Summary

A corpus linguistics toolkit for Polars, split between a Python API and a Rust
matching engine. Search, concordancing, collocation, keywords, dispersion, the
statistical measures and the plots are all working and covered by tests. The
user guide now exists. The main functional gap is proximity operators in the Simple query
language; the main process gaps are the absence of CI and of any Rust unit
tests.

---

## Codebase Metrics

| | |
|---|---|
| Python source | ~5,700 lines, 19 modules |
| Rust source | ~1,100 lines, 6 files |
| Tests | ~4,300 lines, 883 tests in 15 files |
| User guide | 6 pages plus 4 on the query languages (Quarto / great-docs) |
| Examples | 9 notebooks, 2 scripts |

---

## Feature Completeness

### Working

1. **Search engine** — NFA matcher in Rust; CQP and Simple query languages;
   variable bindings (`$x: ...`) in both. A LazyFrame corpus is searched out
   of core, one chunk of whole files at a time (`chunk_tokens` sets the
   budget), returning `LazySearchResults` with file-relative match spans; the
   corpus never has to fit in memory.
2. **Concordancing** — KWIC generation in Rust, `SearchResults.concordance()`,
   a column per bound variable, interactive `ConcordanceWidget` (anywidget)
   with pagination and sorting. Context clips at file boundaries when the
   search named a `file_id_column`.
3. **Collocation and keywords** — `collocates()`, `keywords()`, `crosstab()`
   bundling frequencies into a `freqs` struct consumed by the association
   measures.
4. **Association measures** — PMI, log-likelihood, minimum sensitivity,
   Kilgarriff's simple maths, chi-squared, Welch's t-test.
5. **Lexical diversity** — TTR, MSTTR, Yule's K, MTLD.
6. **Lexical dispersion** — `dispersion()` with range, range%, sd, cv, cv%,
   Juilland's D, Burch's DA, Gries's DP; several measures per call.
7. **I/O** — `read_text_corpus()` / `scan_text_corpus()`, `from_nltk()`.
8. **Chunking** — BIO tags to chunk IDs via `chunk_id()` / `with_chunk_index()`.
9. **Polars integration** — `.corpus` namespace on Expr, DataFrame, LazyFrame.
10. **Visualization** — `barcode_plot()`, `dispersion_plot()`, `keyword_plot()`,
    on seaborn and matplotlib from the `examples` extra.

### Incomplete

- **`visualizations.py`** — the three plots above work and are tested; the
  mosaic plot from a crosstab and the collocation graph are still TODOs.

### Not implemented

- **Proximity operators** (`<<s>>`, `<<3>>`, `<<5<<`, `>>5>>`) in the Simple
  query language. See SIMPLE_QUERY_STATUS.md.
- **Parallel chunk processing.** The lazy search loop is sequential; Polars
  already parallelizes the mask expressions within each chunk, and on the BNC
  the chunked search runs as fast as the eager one, so threading the loop has
  not been worth it yet.

---

## Known Issues

1. **No CI**, deliberately, for now. Nothing runs ruff, clippy, pyrefly, or
   pytest automatically; run them by hand. Worth revisiting closer to a release.
2. **No Rust unit tests.** The engine is exercised only through Python
   integration tests.
3. **`pyrefly check` reports two errors**, both from the `.corpus` namespace
   methods in `exprs.py` passing a `freqs_name` argument that `crosstab` no
   longer takes. No file carries a `# pyrefly: ignore-errors` marker.
4. **The `.corpus` namespace is invisible to type checkers.**
   `pl.api.register_expr_namespace` installs a descriptor onto polars' classes
   with a runtime `setattr`, which no stub can describe, so
   `pl.col("x").corpus.pmi()` does not type-check in user code either. This
   affects every polars plugin and there is no fix available to us. The
   standalone functions (`plc.pmi(...)`, `plc.loglik(...)`) are the
   statically-checkable path; the namespace is sugar over them. Library code
   calls the functions directly for this reason.
5. **`__init__.py` leaks names.** Modules without `__all__` are star-imported,
   so `polars_corpus.pl`, `.Any`, `.Optional` and most submodule names are bound
   at top level. The `keywords` function also shadows the `keywords` module.
6. **A zero-width binding is reported inconsistently.** `bindings_stack` in
   `MatchBuffers` is not part of the backtracking state: `_match_opcodes` pushes
   and pops `(cursor, pc)` tasks, but every task shares one binding stack, so
   which bindings reach the winning path depends on the order the branches were
   tried. A `*` or `?` binding that matched no token therefore sometimes reports
   an empty span and sometimes no binding at all —
   `[pos="DT"] ($mods: [pos="JJ"]*) [pos="NN"]` over "the dog" reports no
   `mods`, while `$adjs: ([pos="JJ"]*) [pos="NN"]` reports `Span(18, 18)` in
   test_matcher.py's `star-zero-match-empty-span`. `concordance()` shows the
   first as a null and the second as an empty list, which is now the visible
   face of the bug. The fix is probably to record the binding stack's depth
   with each task and truncate back to it when the task is resumed.
7. **No published wheels or PyPI release.**

Notebooks are excluded from ruff (`[tool.ruff] extend-exclude`). They are
working scratchpads and are expected to sit in unfinished states, so linting
them produced only noise.

---

## Roadmap

**Before 1.0**
1. Proximity operators.
2. Publish to PyPI.
3. Add CI, deferred until closer to release.

**Quality**
4. Make the binding stack part of the backtracking state, so a zero-width
   binding always reports its empty span (Known Issues 6).
5. Rust unit tests for the matcher.
6. Benchmarks (`examples/bench.py` is a starting point).

---

## Target Audience

Linguistics students and researchers first, data scientists working with text
second. Design target is 100M+ word corpora on 16GB of memory.

---

## Dependencies

- **Runtime:** polars >=1.35, lark >=1.2, nltk >=3.9, anywidget >=0.9
- **Build:** maturin, pyo3 0.28, pyo3-polars 0.27
- **Rust:** polars 0.54, statrs 0.19, itertools 0.15
- **Python:** >=3.11 (wheels are cp311-abi3)
- **Managed via:** `pyproject.toml` plus `uv.lock`; see CLAUDE.md

---

## Build & Test Commands

```bash
make develop          # Rebuild after Rust changes; ~1s incremental
make develop-release  # Same, full release profile (benchmarking)
make build            # Distribution wheels (M4 + x86_64)
make docs             # Build the user guide

ruff format && ruff check
cargo fmt && cargo clippy
pyrefly check python/polars_corpus/
pytest
```

The virtualenv lives outside the source tree at `~/.venvs/polars_corpus`; do not
run `uv run` in this directory. See the Environment section of CLAUDE.md for
why.

---

## Notes

- The Simple query language compiles straight to CQP with no intermediate AST.
- Tests assert on actual matched spans rather than match counts.
- Association measures take a `freqs` struct column, so frequency column names
  are named once in `crosstab()` rather than repeated at every call site.
