# Development Status - polars-corpus

**Last updated:** 2026-08-27
**Version:** 0.2.0-pre
**Status:** Pre-release; core is stable, not yet published to PyPI

---

## Summary

A corpus linguistics toolkit for Polars, split between a Python API and a Rust
matching engine. Search, concordancing, frequency lists, collocation, keywords,
dispersion, the statistical measures and the plots are all working and covered
by tests. Collocation and frequency lists, the two largest documentation holes
in the last survey, are closed: `collocations()` and `frequency_list()`, a
reference page each, and notebooks on the docs site. The concordance workflow
has caught up with them -- `kwic()` for the classic KWIC sort, `as_str=` for a
concordance `write_csv` will take -- and owes only a page for its interactive
widget. `collocations()` and `keywords()` both take a measure of your own -- a
callable -- wherever they take a built-in measure's name. Documentation is
otherwise mkdocs reference pages built from the docstrings, plus the example
notebooks and a reference page for each of the two query languages; there is
no narrative guide. The main functional gap is proximity operators in the Simple
query language; the main process gaps are the absence of CI and of any Rust
unit tests.

---

## Codebase Metrics

| | |
|---|---|
| Python source | ~6,950 lines, 20 modules |
| Rust source | ~1,100 lines, 6 files |
| Tests | ~5,600 lines, 1,094 tests in 17 files |
| Docs | 13 pages plus 6 example notebooks (mkdocs-material / mkdocstrings) |
| Examples | 8 notebooks, 3 scripts |

---

## Feature Completeness

### Working

1. **Search engine** — NFA matcher in Rust; CQP and Simple query languages;
   variable bindings (`$x: ...`) in both. A LazyFrame corpus is searched out
   of core, one chunk of whole files at a time (`chunk_tokens` sets the
   budget), returning `LazySearchResults` with file-relative match spans; the
   corpus never has to fit in memory. A LazyFrame always takes that path,
   as a single chunk when it carries no file id column to cut on.
2. **Concordancing** — KWIC generation in Rust, `SearchResults.concordance()`,
   a column per bound variable, interactive `ConcordanceWidget` (anywidget)
   with pagination and sorting. Context clips at file boundaries when the
   search named a `file_id_column`. `kwic()` returns the expression for a
   context position -- `"L1"`, `"node"`, `"R2"`, or the signed integer CQP
   writes it as -- so the classic KWIC sort is `conc.sort(kwic("L1"),
   kwic("L2"))`, and the same expression groups and filters; `as_str=True`
   joins the list columns into strings, the form `write_csv` and
   `great_tables` accept.
3. **Frequency lists** — `frequency_list()` returns one row per type with
   its count, its rate per `basis` words, and the number of files it occurs
   in. Normalizing and thresholding are the caller's: a case fold in `expr`,
   a `filter` on the corpus for what counts as a word, and a `filter` on the
   result to drop the rare words.
4. **Collocation and keywords** — `collocations()` ranks the words around a
   search's matches by one or more association measures; `collocates()`
   returns the window counts it is built on, and `keywords()` and `crosstab()`
   bundle frequencies into the same `freqs` struct the measures consume.
   Windows are symmetric or asymmetric (`window=(5, 0)`), or run to the edges
   of the chunk holding the match when given a `chunk_column`, which is how a
   span stops at a sentence boundary. Colligation is the same call over the
   `pos` column, or over a struct of token and tag; the notebook shows it.
5. **Association measures** — PMI, MI3, log-dice, t-score, z-score,
   log-likelihood, chi-squared, minimum sensitivity, Kilgarriff's simple
   maths, Welch's t-test. `collocations()` and `keywords()` each take the
   measures that make sense for them by name, or a callable of your own with
   the same `(f12, f1, f2, n)` signature, ranked and named alongside the
   built-ins (`_apply_measure` in `assoc.py`).
6. **Lexical diversity** — TTR, MSTTR, Yule's K, MTLD.
7. **Lexical dispersion** — `dispersion()` with range, range%, sd, cv, cv%,
   Juilland's D, Burch's DA, Gries's DP; several measures per call.
8. **I/O** — `read_text_corpus()` / `scan_text_corpus()`, `from_nltk()`.
9. **Chunking** — BIO tags to chunk IDs via `chunk_id()` / `with_chunk_index()`.
10. **Polars integration** — `.corpus` namespace on Expr, DataFrame, LazyFrame.
11. **Visualization** — `barcode_plot()`, `dispersion_plot()`, `keyword_plot()`,
    on matplotlib from the `examples` extra. (The notebooks still import
    seaborn; the library itself no longer does.)

### Incomplete

- **`visualizations.py`** — the three plots above work and are tested; the
  mosaic plot from a crosstab is still a TODO. So is a plot of collocates,
  but not as the collocation graph the file's TODO names: networks are not
  the direction, and clustering the collocate space -- spectral clustering
  is the likeliest first try -- is the shape to explore instead. Either way
  it is a task of its own, not part of `collocations()`.

### Not implemented

- **Proximity operators** (`<<s>>`, `<<3>>`, `<<5<<`, `>>5>>`) in the Simple
  query language. They are not in the grammar and raise
  `UnexpectedCharacters`. There is no direct CQP equivalent, so translation
  needs one of:
  1. *Expand to CQP disjunctions.* `day <<3>> night` becomes
     `([token="day"%c] []{0,3} [token="night"%c]) | ([token="night"%c] []{0,3} [token="day"%c])`.
     Simple, but combinatorial once constraints nest, e.g.
     `waste <<s>> (time <<3>> money)`.
  2. *Add proximity opcodes to the Rust matcher.* Efficient and the right
     long-term answer, but the largest change.
  3. *Filter after matching.* Search each term separately and post-process by
     distance. Least efficient, and awkward to compose.

  Sentence-level proximity (`<<s>>`) additionally needs a sentence boundary
  column, which `with_chunk_index()` can already supply.
- **Dependency-based collocation** (word sketches). Out of scope here: the
  library's data format is flat and nothing in it reads dependency arcs.
  Dependency annotation is a separate project, which this one may link to
  later.
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
7. **`{lemma/CLASS}_TAG` drops the class.** A Simple query that gives both a
   simplified POS class and an explicit tag keeps the `_TAG` and discards the
   class without complaint (`LEMMA` in `simple_parser.py`).
8. **No published wheels or PyPI release.**

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

**Coverage**

See [FEATURE_GAPS.md](FEATURE_GAPS.md) for what the toolkit is still missing
against what a linguist expects. Collocation, frequency lists and the
concordance workflow -- the big ones at the last three surveys -- are each
finished end to end: function, reference page, notebook. Next is keyness
effect sizes, log ratio and %DIFF, which fit the callable `_apply_measure`
already takes and need no new mechanism. `from_spacy()` and a page for the
`ConcordanceWidget` come after the release rather than before it: each is a
task with its own beginning and end, and nothing already shipped waits on
either.

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

- The Simple query language compiles straight to CQP with no intermediate AST:
  `simple_parser.simple_to_cqp()` emits a CQP string that the same matcher
  behind `search_cqp()` compiles. The grammar (`_GRAMMAR` in
  `simple_parser.py`) is lark, built into an LALR parser once at import; each
  call runs a `SimpleCompiler` transformer holding the requested column names.
  Because whitespace separates query items, a whole token (`{walk}_VB*`) has to
  lex as one terminal, so the transformer re-matches each terminal to recover
  its parts, against regexes (`_POS_TAG_PARTS`, `_LEMMA_PARTS`) built from the
  same fragments as the grammar so the two cannot drift. `docs/simple_query.md`
  documents the language for users, `docs/cqp_query.md` documents what it
  compiles to, and the two grammars are the specification.
- Tests assert on actual matched spans rather than match counts.
- Association measures take a `freqs` struct column, so frequency column names
  are named once in `crosstab()` rather than repeated at every call site.
- `collocates()` and `collocations()` share one counting pass,
  `_SearchResultsBase._collocate_counts`: the first returns the counts, the
  second joins the corpus frequencies and scores them. `f1` counts the context
  tokens the windows actually held rather than the positions they could have
  held, so a window truncated at a file or chunk boundary contributes only
  what it reached.
