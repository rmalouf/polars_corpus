# CLAUDE.md

## Overview
`polars-corpus` is a hybrid Python/Rust extension for Polars providing corpus search,
concordancing, and statistical measures. Designed to comfortably handle 100M+ word
corpora on 16GB memory. Target audience: linguists and data scientists, especially students.

## Architecture
- **Python/Rust split**: High-level APIs in Python, matching engine in Rust for performance
- **Zero-copy bridge**: pyo3-polars for efficient data exchange
- **Polars integration**: Extends DataFrames via `.corpus` namespace

## Development Workflow
```bash
ruff format       # Format Python
ruff check        # Lint Python
cargo fmt         # Format Rust
cargo clippy      # Lint Rust
pyrefly check python/polars_corpus/
make develop      # Rebuild after Rust changes (required); ~1s incremental
make build        # Release wheels for distribution (slow: full LTO)
pytest            # Run tests (Python changes don't need rebuild)
```

## Environment
The venv lives **outside** the source tree, at `~/.venvs/polars_corpus`, and is
activated automatically by a shell extension. Run `pytest`, `ruff`, etc. directly
-- do **not** prefix them with `uv run`, which would create a `./.venv` here.

That matters for two reasons: this directory is synced via Dropbox to machines
with different paths, and a venv inside the project root breaks `pytest` --
nltk's `inisec` import guard rejects any module resolving under the cwd, so
`import nltk` fails during test collection.

## Dependencies
All declared in `pyproject.toml` and pinned by `uv.lock` (both tracked):
- `[project] dependencies`: runtime
- `[project.optional-dependencies] examples`: published extra for end users
- `[dependency-groups] dev`: development tools (pytest, pyrefly, maturin, ...)
- `[dependency-groups] notebooks`: `dev` plus what the example notebooks import

Re-resolve with `uv lock` after editing `pyproject.toml`. `uv lock` only writes
the lockfile and does not touch the venv, so it is safe to run here.

## Data Format
DataFrame with: `token`, `pos`/`c5`, `mode`, `file_id`, plus annotation columns
Column names are defaults, not requirements: every function that reads one of
these roles takes a `*_column` parameter to point it elsewhere.

## Query Languages
The package supports Simple (BNCweb-style) and CQP query syntaxes. See [QUERY_LANGUAGE.md](QUERY_LANGUAGE.md) for syntax details.

## Coding Standards
- Use numpy-style docstrings. Keep them minimal, focused on API usage
- Use **parameterized tests** when possible
- Avoid code bloat - keep implementations focused
- **Avoid gratuitous defensive programming**:
    - Public APIs: Validate inputs and provide helpful error messages
    - Internal functions: Trust invariants - skip redundant checks
    - Use assertions for debugging, not runtime validation of established invariants

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

### Rust-Specific
- **Minimize allocations**: Use `&str` over `String`, `&[T]` over `Vec<T>`; avoid `.clone()` in hot paths
- **Use iterators** without collecting when possible
- **Reuse buffers** in hot loops rather than allocating per iteration