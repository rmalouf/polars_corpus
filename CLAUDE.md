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
make develop      # Rebuild after Rust changes (required)
pytest            # Run tests (Python changes don't need rebuild)
```

## Dependencies
Managed via `uv` with three levels of requirements:
- `requirements.txt`: Runtime dependencies only
- `requirements-dev.txt`: Runtime + development tools (pytest, pyrefly, etc.)
- `requirements-examples.txt`: Runtime + dev + example notebooks
- Regenerate lock files: `make locks`

## Data Format
DataFrame with: `token`, `pos`/`c5`, `mode`, `fileid`, plus annotation columns

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

### Rust-Specific
- **Minimize allocations**: Use `&str` over `String`, `&[T]` over `Vec<T>`; avoid `.clone()` in hot paths
- **Use iterators** without collecting when possible
- **Reuse buffers** in hot loops rather than allocating per iteration