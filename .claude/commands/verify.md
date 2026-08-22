---
description: Run full verification suite (format, lint, type check, build, test)
---

Run the complete verification workflow before committing changes:

1. Format Python code with ruff
2. Lint Python code with ruff
3. Type check Python with mypy
4. Format Rust code with cargo fmt
5. Lint Rust code with cargo clippy
6. Rebuild the Rust extension with make develop
7. Run the test suite with pytest

Report any failures clearly. If all steps pass, confirm the codebase is ready to commit.
