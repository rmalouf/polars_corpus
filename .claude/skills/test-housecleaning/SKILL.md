---
name: test-housecleaning
description: Prune and consolidate the pytest suite -- drop redundant and unreachable tests, collapse near-duplicates into parameterized ones, and retire tests written against an API that no longer exists. Use when asked to clean up, tidy, prune, thin out or consolidate the tests, or after a refactor has left the suite fitting the old code.
---

# Housecleaning the test suite

The suite is 17 files in `python/tests/`, about 440 written tests, 1094 after
parametrize expansion, 41s wall. It is thorough and mostly well written. This
is pruning, not rescue: the default outcome of reading a test is that it stays.

Housecleaning is its own commit. Do not add coverage while you are here, and do
not fix the module under test unless a test you were about to delete turns out
to have found a bug.

## Before you start

Take a baseline and keep it:

```bash
coverage run -m pytest -q
coverage report --include="python/polars_corpus/*"
```

96% total at the time of writing, and every module in the low 90s or better.
Record the per-file numbers. If a file's coverage drops, a test you deleted was
doing work that nothing else does -- put it back or replace it.

Coverage is a floor, not the goal. Two tests can execute the same lines and
only one of them assert anything about the result. The line-count check catches
the deletions that were wrong; it does not bless the ones that were right.

## Work by module, not by test file

The mapping is not 1:1, and the redundancy hides in the gap. `search.py` alone
is exercised by `test_concordance.py`, `test_lazy_search.py`, `test_matcher.py`,
`test_view.py`, `test_spans.py` and `test_embeddings.py`; `cqp_parser.py` and
`exprs.py` have no file of their own at all. Read the module first, then gather
every test file that imports from it, then decide. Deciding one file at a time
is how the same case ends up tested three ways.

`pyproject.toml` puts `--doctest-modules` on `testpaths`, which includes
`python/polars_corpus/simple_parser.py`. Those doctests are the reference for
the query language and are documentation as much as test. Leave them. The
doctest in `helpers.corpus` runs too.

## 1. Redundant

**The shared guards.** `utils.py` holds `as_corpus`, `as_eager`, `as_expr`,
`check_columns` and `check_choice`, and every public function opens with them.
Their behavior belongs to `test_utils.py`. It is currently re-tested in
`test_dispersion.py`, `test_frequency.py`, `test_visualizations.py`,
`test_keywords.py`, `test_embeddings.py` and `test_concordance.py` --
`match="the corpus must be a polars"` appears four times over, `"the corpus is
empty"` three.

A module's own tests cover only what that module *decides*: which columns it
reads, which name it passes as `param=`, which role name it gives each frame.
So `keywords()` calling its two frames "target" and "reference", and
`concordance()` naming `chunk_column=` in the message that reports a misspelling
-- those are the module's, keep them. `as_corpus` rejecting a list is not, and
certainly not six times.

**The free function and the method.** Most of the API is both. One test that
the two are the same call is the whole obligation; testing the behavior again
through the second entry point is not. `TestFunctionalInterface` in
`test_concordance.py` has the shape right and is three tests where one
parameterized test would do.

**Eager and lazy.** Principle 2 in CLAUDE.md means most functions take either.
Test the type-in/type-out rule once per function; do not run the function's
whole behavior twice. `test_lazy_search.py` is the exception and stays as it is
-- there the two paths are genuinely different code, and agreeing is the point.

## 2. Collapse

Prefer `@pytest.mark.parametrize` -- CLAUDE.md asks for it, and the suite
already leans that way. Near-identical bodies differing in one literal, a
method name, or an expected value collapse cleanly. Use `pytest.param(...,
id=...)` when the auto-generated id would be unreadable; the suite does this
well already and the ids are worth preserving through a collapse.

Do not collapse when:

- the arrange steps differ, and the parameter would have to carry a fixture or
  a branch into the body;
- the id would have to encode the assertion to stay meaningful;
- a failure would then name a case that isn't the one that broke.

A `class Test...` grouping with a one-line docstring saying what the group is
about is house shape here and worth keeping. Collapsing within a class is
usually right; merging two classes usually is not.

## 3. Things that can never happen

CLAUDE.md: public APIs validate, internal functions trust invariants. A test
that feeds an internal helper something the public entry point has already
rejected is testing a branch that cannot run in the shipped package. Delete it,
or move the case up to the public function where the input really can arrive.

Also unreachable, and also deletable:

- asserting a fixture holds what the fixture just built;
- guarding a dtype polars will not produce for that column;
- a null check on something a required parameter cannot be;
- an error path in Rust-facing code that the Python wrapper filters first.

**Keep the boundary cases.** They look like edge-case noise and are not: an
empty corpus, a single file, a window reaching past the corpus edge, an `I` tag
with no preceding `B`, `sample(n)` where `n == len(results)`, a query with no
matches. Those are inputs a student will actually produce.

## 4. Written for a version that is gone

Signals, each of which this repo's history has produced at least once:

- **A file named after a module that no longer exists.** `concordance.py` was
  folded into `search.py`; `test_concordance.py` remains. `productivity.py` was
  folded into `lexical.py` (4057327). The name outliving the module is a hint
  that the tests inside were written against the old boundary -- check whether
  they are still testing the seam that used to be there.
- **A plotting backend that changed twice.** seaborn to plotly (a8e99e6) to
  matplotlib (582e721, fbf32db, 436001c). Assertions about figure internals
  date fast.
- **`pytest.warns` for a warning that was dropped** -- 3b3716a removed the
  null-row warnings. (The four `pytest.warns` in the suite now are all live;
  this is the pattern to watch, not a standing finding.)
- **Reaching into private state.** `test_lazy_search.py` asserts on
  `lazy._matches` and `results._query`. Sometimes that is the only way in;
  often the attribute was public, or the only accessor, when the test was
  written. Check for a public accessor now.
- **A docstring describing behavior the code no longer has.** The prose dates
  faster than the assertions and is the cheapest tell.

To settle it: `git log -S'<symbol>' -- python/polars_corpus/` finds when the
thing the test is about arrived or left.

## 5. Compatibility shims -- look hardest here

**When a refactor breaks a test, the test is what changes.** A shim added to
the library so old tests keep passing pushes test convenience into the shipped
API, where students find it and depend on it. The test suite is not a
constituency the API design has to satisfy.

The precedent is 241c01c. `SearchResults.variables` was optional and worked out
alphabetically from the matches when omitted -- but every caller that omitted
it was a test, because a real search always knows the order its query binds in.
The parameter became required, the alphabetical fallback moved to
`search_results` in `python/tests/helpers.py`, and the tests were updated.

That is the shape of the fix, in order:

1. the library gets stricter, not more accommodating;
2. the convenience the tests genuinely wanted lands in `python/tests/helpers.py`;
3. the test bodies are rewritten to say what they mean.

What a shim looks like before you have named it:

- a parameter whose default no shipped caller relies on;
- an alias or old name kept "for compatibility";
- a function accepting two shapes of input where one shape is only ever passed
  from `python/tests/`;
- a keyword that is silently ignored under some `method`;
- `try/except ImportError` around something that is now a core dependency
  (`anywidget` is core; matplotlib is the `examples` extra).

The test: grep for a caller outside `python/tests/`. Check `python/polars_corpus/`,
`docs/`, and the notebooks in `examples/`. If every caller is a test, the
parameter is a fixture wearing the library's clothes.

**Shims inside the tests** are the same fault inverted -- a helper whose job is
to rebuild the old API's call shape so the test body need not be touched.
Rewrite the body.

**Duplication is not a shim.** `log_ratio` and `named_by_alias` are defined
identically in `test_keywords.py` and `test_collocations.py`. That is a helper
to hoist into `helpers.py`; the two tests it serves cover two different public
functions and both stay. `chunk_ids_via_function` / `chunk_ids_via_expression`
in `test_spans.py` looks like an adapter pair and is not -- it parameterizes
over two real public entry points to the same feature. Read what the helper is
*for* before calling it a shim.

## What not to do

- **Do not delete a failing test.** It is either a live bug or a stale test.
  Read the module and decide which; convenience is not the tiebreak.
- **Do not weaken an assertion to make a test pass.** Loosening `assert_frame_equal`
  to a shape check, or an exact value to `pytest.approx` with a wide tolerance,
  is deleting the test while leaving the name behind.
- **Do not drop the last test of a boundary case** because it resembled a
  duplicate. Check what the survivor actually asserts, not what it is named.
- **Do not rename test functions in bulk** in this commit. Renames are cheap
  and they make the diff unreadable; separate commit if they are wanted.
- **Do not touch the `simple_parser.py` doctests.**

## Check

1. `pytest` passes. The test count went down and the wall time did not go up.
2. `coverage report --include="python/polars_corpus/*"`: no module below where
   it started.
3. Every deletion falls under one of the five headings above, and you can say
   which. Put that in the commit message -- what went and why, not how many.
4. Nothing under `python/polars_corpus/` gained a parameter, a default, or a
   branch in order to keep a test passing.
5. For any library parameter removed as a shim: grep confirms no caller outside
   `python/tests/`, including `examples/` and `docs/`.
6. `ruff format` and `ruff check` leave the files alone.
7. Helpers that moved to `helpers.py` are imported by more than one file. One
   caller means it belonged where it was.
8. The docstring on every test and class you touched still describes what the
   test now does.
