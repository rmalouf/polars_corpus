# CLAUDE.md

## Overview
`polars-corpus` is a hybrid Python/Rust extension for Polars providing corpus
search, concordancing, and statistical measures: high-level APIs in Python, the
matching engine in Rust, bridged zero-copy by pyo3-polars and hung off Polars
frames and exprs as a `.corpus` namespace. Designed to comfortably handle 100M+
word corpora on 16GB memory. Target audience: linguists and data scientists,
especially students.

## Development Workflow
```bash
ruff format && ruff check          # Format and lint Python
cargo fmt && cargo clippy          # Format and lint Rust
pyrefly check python/polars_corpus/
pytest                             # Python changes need no rebuild
make develop                       # Rebuild after Rust changes (required); ~1s
make develop-release               # Same, release profile (benchmarking)
make build                         # Distribution wheels (slow: full LTO)
make docs                          # Build the user guide
```

## Environment
The venv lives **outside** the source tree, at `~/.venvs/polars_corpus`, and is
activated automatically by a shell extension. Run `pytest`, `ruff`, etc.
directly -- do **not** prefix them with `uv run`, which would create a `./.venv`
here. That breaks two things: this directory is synced via Dropbox to machines
with different paths, and nltk's `inisec` import guard rejects any module
resolving under the cwd, so a venv in the project root makes `import nltk` fail
during test collection.

## Dependencies
All declared in `pyproject.toml` and pinned by `uv.lock` (both tracked):
- `[project] dependencies`: runtime
- `[project.optional-dependencies] examples`: published extra for end users
- `[dependency-groups] dev`: development tools (pytest, pyrefly, maturin, ...)
- `[dependency-groups] notebooks`: `dev` plus what the example notebooks import

Re-resolve with `uv lock` after editing `pyproject.toml`. It only writes the
lockfile and does not touch the venv, so it is safe to run here.

## Data Format
DataFrame with: `token`, `pos`/`c5`, `mode`, `file_id`, plus annotation columns.
Column names are defaults, not requirements: every function that reads one of
these roles takes a `*_column` parameter to point it elsewhere.

## Query Languages
The package supports Simple (BNCweb-style) and CQP query syntaxes. See
[docs/simple_query.md](docs/simple_query.md) for the Simple language; the CQP
grammar in `cqp_parser.py` and the `search_cqp()` docstring cover the other.

## Coding Standards
- Use numpy-style docstrings. Keep them minimal, focused on API usage
- Use **parameterized tests** when possible
- Avoid code bloat - keep implementations focused
- **Avoid gratuitous defensive programming**: validate inputs at public APIs and
  give helpful error messages; trust invariants in internal functions. Use
  assertions for debugging, not runtime validation of established invariants

### Design principles
1. When possible, functions should take exprs and return exprs.
2. When a function must take a frame, prefer accepting both DataFrame and
   LazyFrame, and return whichever type it was given.
3. Work lazily inside even when given a DataFrame; principle 2 decides the
   return type.
4. When a function can only work on a DataFrame, check up front and raise
   an error if given a LazyFrame.

### Public functions
Student-facing functions share a shape, with the pieces in `utils.py`
(internal, so not in its `__all__`):
```python
def analyze(corpus, expr, method="ll", file_id_column="file_id"):
    method = check_choice(method, METHODS)       # names the options, suggests near misses
    term = as_expr(expr)                         # column name or expression
    lf = as_corpus(corpus)                       # rejects non-frames and empty frames
    check_columns(lf, [file_id_column], param="file_id_column")
    ...                                          # all internal work lazy
    return collect_like(result, corpus)          # eager in, eager out
```
Read only the columns the arguments name, so corpora annotated differently
still work together and column errors surface here rather than out of a
query plan.

A function that can only work eagerly (principle 4) opens with
`as_eager(corpus)` in place of `as_corpus`, and has no `collect_like` to
return through: `SearchResults` and `encode_terms` take a corpus that way.
`search` and `search_cqp` accept a LazyFrame too, but down a separate
out-of-core path (chunked on `file_id_column`, or a single chunk when the
frame has no such column) rather than through `as_corpus`.

### Rust-Specific
- **Minimize allocations**: Use `&str` over `String`, `&[T]` over `Vec<T>`; avoid `.clone()` in hot paths
- **Use iterators** without collecting when possible
- **Reuse buffers** in hot loops rather than allocating per iteration
