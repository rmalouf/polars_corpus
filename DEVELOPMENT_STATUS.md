# Development Status - polars-corpus

**Last updated:** 2026-08-28
**Version:** 0.2.0-pre
**Status:** Pre-release; core is stable, not yet published to PyPI

---

## Summary

A corpus linguistics toolkit for Polars: a Python API over a Rust matching
engine. Search, concordancing, frequency lists, collocation, keywords,
dispersion, the statistical measures and the plots all work and are tested.
The main functional gap is proximity operators in the Simple query language;
the main process gap is the absence of any Rust unit tests. What else is left
is mostly writing -- a page for the concordance widget, an
effect-size section in the keywords notebook, prose on three stub reference
pages.

---

## Codebase Metrics

| | |
|---|---|
| Python source | ~7,300 lines, 20 modules |
| Rust source | ~1,100 lines, 6 files |
| Tests | ~5,600 lines, 1,095 tests in 17 files |
| Docs | 13 pages plus 6 example notebooks (mkdocs-material / mkdocstrings) |
| Examples | 8 notebooks, 3 scripts |

---

## Feature Completeness

### Working

1. **Search engine** — NFA matcher in Rust; CQP and Simple query languages,
   with variable bindings (`$x: ...`) in both. A LazyFrame corpus is searched
   out of core, one chunk of whole files at a time (`chunk_tokens` sets the
   budget), returning `LazySearchResults` with file-relative spans, so the
   corpus never has to fit in memory; a frame with no file id column to cut on
   goes through as a single chunk.
2. **Concordancing** — KWIC generation in Rust, `SearchResults.concordance()`,
   a column per bound variable, interactive `ConcordanceWidget` (anywidget).
   Context clips at file boundaries when the search named a `file_id_column`.
   `kwic()` returns the expression for a context position -- `"L1"`, `"node"`,
   `"R2"`, or the signed integer CQP writes it as -- so the classic sort is
   `conc.sort(kwic("L1"), kwic("L2"))` and the same expression groups and
   filters; `as_str=True` joins the list columns into the strings `write_csv`
   and `great_tables` accept.
3. **Frequency lists** — `frequency_list()` gives one row per type: its count,
   its rate per `basis` words, and the number of files it occurs in.
   Normalizing and thresholding stay the caller's (see Deliberately out).
4. **Collocation and keywords** — `collocations()` ranks the words around a
   search's matches by one or more measures; `collocates()` returns the window
   counts underneath it, and `keywords()` and `crosstab()` bundle frequencies
   into the same `freqs` struct the measures consume. Windows are symmetric,
   asymmetric (`window=(5, 0)`), or run to the edges of the chunk holding the
   match (`chunk_column`), which is how a span stops at a sentence boundary.
5. **Association measures** — PMI, MI3, log-dice, t-score, z-score,
   log-likelihood, chi-squared, minimum sensitivity, Kilgarriff's simple maths,
   Welch's t-test, plus the keyness effect sizes the field reports beside a
   significance score: log ratio, %DIFF, the odds ratio, Bayes-factor BIC.
   Either function takes a measure by name or as a callable with the same
   `(f12, f1, f2, n)` signature (`_apply_measure` in `assoc.py`). Log ratio and
   %DIFF read `f2` as the size of the corpus `f12` was counted in, so they are
   keyness-only; they and `oddsratio` take a `discount` for the zero cell
   (0.5, CQPweb's convention), which `keywords()` does not expose for the
   reason it does not expose `chisq`'s `yates`: the argument belongs to the
   measure.
6. **Lexical diversity** — TTR, MSTTR, Yule's K, MTLD.
7. **Lexical dispersion** — `dispersion()` with range, range%, sd, cv, cv%,
   Juilland's D, Burch's DA, Gries's DP; several measures per call.
8. **I/O** — `read_text_corpus()` / `scan_text_corpus()`, `from_nltk()`.
9. **Chunking** — BIO tags to chunk IDs via `chunk_id()` / `with_chunk_index()`.
10. **Polars integration** — `.corpus` namespace on Expr, DataFrame, LazyFrame.
11. **Visualization** — `barcode_plot()`, `dispersion_plot()`, `keyword_plot()`,
    on matplotlib from the `examples` extra. The notebooks still import seaborn;
    the library no longer does.

### Incomplete

- **`visualizations.py`** — the three plots above work and are tested; the
  mosaic plot from a crosstab is still a TODO, as is a plot of collocates.

### Not implemented

- **Proximity operators** (`<<s>>`, `<<3>>`, `<<5<<`, `>>5>>`) in the Simple
  query language. They are not in the grammar and raise `UnexpectedCharacters`.
  There is no direct CQP equivalent, so translation needs one of:
  1. *Expand to CQP disjunctions.* `day <<3>> night` becomes
     `([token="day"%c] []{0,3} [token="night"%c]) | ([token="night"%c] []{0,3} [token="day"%c])`.
     Simple, but combinatorial once constraints nest, e.g.
     `waste <<s>> (time <<3>> money)`.
  2. *Add proximity opcodes to the Rust matcher.* Efficient and the right
     long-term answer, but the largest change.
  3. *Filter after matching.* Least efficient, and awkward to compose.

  Sentence-level proximity (`<<s>>`) additionally needs a sentence boundary
  column, which `with_chunk_index()` can already supply.
- **Parallel chunk processing.** The lazy search loop is sequential; Polars
  already parallelizes the mask expressions within each chunk, and on the BNC
  the chunked search runs as fast as the eager one, so it has not been worth it.

### Deliberately out

Decisions already made, recorded here so they are not re-proposed as gaps.

- **Dependency-based collocation** (word sketches). The data format is flat and
  nothing reads dependency arcs; annotating for them is a separate project.
- **A `colligations()` function.** `collocations(expr=...)` over `pos`, or over
  a struct of token and tag, already is it.
- **Collocation networks.** Clustering the collocate space -- spectral, most
  likely -- is the direction worth trying instead.
- **Stopword lists.** A judgement about which words carry no meaning, made once
  for no particular question and then applied to every question. `is_in` over
  your own list is the version you can defend.
- **`lowercase=`, `letters_only=`, `min_freq=`, `min_range=` on
  `frequency_list()`.** All written, all removed: normalizing is `expr`'s job,
  restricting what counts as a word is a `filter` on the corpus, and
  thresholding is a `filter` on the result, which leaves the rate alone because
  it is computed first. The `min_freq` on `dispersion()` and `collocations()` is
  a different argument -- it guards a measure that misbehaves on rare words.
- **`by=` on `frequency_list()`.** The expression-returning form that would let
  Polars' own `group_by` do the breakdown was tried: it cannot sort itself,
  costs the caller `.explode().unnest()` at every call site, runs twice as slow
  at 20M rows, and has no frame to check column names against. `by=` can still
  come back later without changing the frame form.
- **A hits-by-metadata breakdown and an HTML export for concordances.** The raw
  counts are a Polars `group_by`; normalizing them against category size is
  `with_spans_as_chunks()` plus one `group_by` over the tagged corpus; and after
  `as_str=True` the HTML export is `conc.style` from `great_tables`.
- **`logdice` for keyness.** Here `f2` is the size of the target corpus rather
  than a second word's frequency, so `2 f12 / (f1 + f2)` is dominated by `f2`
  and reduces to a monotone function of relative frequency.

---

## Known Issues

1. **No Rust unit tests.** The engine is exercised only through Python.
2. **`pyrefly check` reports two errors**, both from the `.corpus` namespace
   methods in `exprs.py` passing a `freqs_name` argument `crosstab` no longer
   takes. Third-party stubs produce errors of their own, which is why CI does
   not run pyrefly as a gate.
3. **The `.corpus` namespace is invisible to type checkers.**
   `register_expr_namespace` installs the descriptor with a runtime `setattr`,
   which no stub can describe, so `pl.col("x").corpus.pmi()` does not
   type-check in user code either. This affects every polars plugin. The
   standalone functions (`plc.pmi(...)`) are the statically-checkable path, and
   library code calls them directly for that reason.
4. **`__init__.py` leaks names.** Modules without `__all__` are star-imported,
   so `polars_corpus.pl`, `.Any` and most submodule names are bound at top
   level; the `keywords` function also shadows the `keywords` module.
5. **A zero-width binding is reported inconsistently.** `bindings_stack` in
   `MatchBuffers` is not part of the backtracking state -- `_match_opcodes`
   pushes and pops `(cursor, pc)` tasks that all share one binding stack -- so a
   `*` or `?` binding that matched no token reports an empty span or no binding
   at all depending on the order the branches were tried
   (test_matcher.py's `star-zero-match-empty-span`); `concordance()` shows the
   two as a null and an empty list. The fix is probably to record the stack's
   depth with each task and truncate back to it when the task resumes.
6. **`{lemma/CLASS}_TAG` drops the class.** A Simple query giving both a
   simplified POS class and an explicit tag keeps the `_TAG` and discards the
   class without complaint (`LEMMA` in `simple_parser.py`).
7. **No published wheels or PyPI release.**

---

## Roadmap

**Before 1.0**
1. Proximity operators.
2. Publish to PyPI.

**Quality**
3. Make the binding stack part of the backtracking state (Known Issues 5).
4. Rust unit tests for the matcher.
5. Benchmarks (`examples/bench.py` is a starting point).

**Future plans**

Coverage against what a linguist expects of a corpus toolkit. Nothing shipped
is waiting on any of these. A feature that isn't documented is a feature users
don't have, so the writing entries rank with the code ones.

6. **A page for `ConcordanceWidget`.** Written and tested, but documented only
   here and absent from the docs site.
7. **An effect-size section in the keywords notebook.** It argues that
   log-likelihood alone misleads and then offers nothing instead; ranking the
   same words by `ll` and by `logratio` side by side is what closes it.
8. **N-grams and clusters.** `ngrams()` sits in `docs/utils.md` with no prose
   or example; clusters around a node and lexical-bundle extraction, the
   phraseology staple, do not exist.
9. **Text-level descriptive measures.** Per-text sentence length, mean word
   length and readability (Flesch, ARI) -- the basic descriptive battery, none
   of which is implemented.
10. **`from_spacy()`.** Raw text to `token`, `lemma`, `pos`, `tag` and
    `sentence_tag`, batched over `nlp.pipe()` -- the entry that most widens who
    can use the library, and the only documented way to get a `lemma` column.
    `from_stanza()` is the sibling, but spaCy has the users.
11. **Docs for `chunk_id()` and `with_chunk_index()`.** They supply the
    sentence boundaries sentence-scoped work needs, and are on no page.
12. **A narrative getting-started guide.** The User Guide nav entries are still
    commented out in `mkdocs.yml`.
13. **Prose on `assoc.md`, `lexical.md` and `utils.md`.** Bare mkdocstrings
    stubs -- nothing on what a measure means or when to reach for it.

---

## Dependencies

- **Runtime:** polars, lark, nltk, anywidget; versions in `pyproject.toml`
- **Build:** maturin, pyo3, pyo3-polars; the Rust side (polars, statrs,
  itertools) is pinned in `Cargo.toml`
- **Python:** >=3.11; wheels are cp311-abi3

---

## Notes

- Build and test commands, and why the venv sits outside the source tree, are
  in CLAUDE.md.
- CI (`.github/workflows/`) runs on every push to main and every PR: `test.yml`
  lints (cargo fmt, clippy, ruff) and runs pytest on 3.11 through 3.14, Linux
  plus one macOS job to guard the arm64 build; `docs.yml` builds the site.
  pyrefly is not a gate (Known Issues 2).
- The repo is mirrored to tangled.org, whose CI (`.tangled/workflows/`) runs the
  same checks on a hosted spindle, one file per pipeline: `test.yml` for cargo
  fmt, clippy and pytest, `lint.yml` for ruff, `docs.yml` for the site. One
  Python version, Linux only, and no build cache between runs, so GitHub stays
  the stricter gate for now.
- The Simple query language compiles straight to CQP with no intermediate AST:
  `simple_to_cqp()` emits a CQP string that the matcher behind `search_cqp()`
  compiles. Its lark grammar (`_GRAMMAR` in `simple_parser.py`) is built into an
  LALR parser once at import, and each call runs a `SimpleCompiler` transformer
  holding the requested column names. Because whitespace separates query items,
  a whole token (`{walk}_VB*`) has to lex as one terminal, so the transformer
  re-matches each terminal against regexes (`_POS_TAG_PARTS`, `_LEMMA_PARTS`)
  built from the same fragments as the grammar, so the two cannot drift.
  `docs/simple_query.md` and `docs/cqp_query.md` document the two languages;
  the grammars are the specification.
- Tests assert on actual matched spans rather than match counts.
- Association measures take a `freqs` struct column, so frequency column names
  are named once in `crosstab()` rather than repeated at every call site.
- `collocates()` and `collocations()` share one counting pass,
  `_SearchResultsBase._collocate_counts`. `f1` counts the context tokens the
  windows actually held rather than the positions they could have held, so a
  window truncated at a file or chunk boundary contributes only what it reached.
- Notebooks are excluded from ruff (`[tool.ruff] extend-exclude`): they are
  working scratchpads expected to sit in unfinished states, and linting them
  produced only noise.
