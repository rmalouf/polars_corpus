# Makefile for managing Python dependencies with uv

# The venv lives outside the source tree (see CLAUDE.md) and is activated by a
# shell extension. Dependencies are declared in pyproject.toml; re-resolve with
# `uv lock`.
VENV_DIR = ${HOME}/.venvs/polars_corpus

.PHONY: develop develop-release build docs grid

# The Python versions the wheels claim (pyproject classifiers), and the profile
# `make grid` builds them with.
GRID_VERSIONS = 3.11 3.12 3.13 3.14
GRID_PROFILE ?= dev-fast
# The oldest polars pyproject allows. The lockfile's pin is the newest, so the
# two legs bracket the range the dependency claims to support.
GRID_POLARS_MIN = 1.36.*

# Serve docs locally
docs:
	#quarto convert user_guide/04-frequencies.ipynb
	mkdocs build

# Rebuild the Rust extension for the edit/test loop. Optimized and native-CPU,
# but without LTO, so this takes seconds rather than minutes. Use this one.
develop:
	RUSTFLAGS="-C target-cpu=native" maturin develop --profile dev-fast

# Same, but with the full release profile. Only needed when benchmarking or
# checking something that depends on LTO; `make develop` is the normal path.
develop-release:
	RUSTFLAGS="-C target-cpu=native" maturin develop --release

# Run the test suite on every supported Python version. Not for the edit/test
# loop -- this is the pre-release check, the half of the CI matrix that GitHub
# covers with a single macOS job. Each version gets its own environment beside
# the development one, outside the source tree (see CLAUDE.md), so it does not
# disturb the extension `make develop` built. The install is non-editable, so
# what the tests import is a built wheel rather than python/, and cargo reuses
# one build across the four: the extension is abi3.
# Each version runs twice, against the lockfile's polars and against the oldest
# pyproject allows: `explode(empty_as_null=)` and `LazyFrame.pivot` are the kind
# of thing that raises the floor without anyone noticing.
# `make grid GRID_PROFILE=release` runs them against what actually ships.
grid:
	@for v in $(GRID_VERSIONS); do \
		printf "\n=== Python %s ===\n" $$v ; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(VENV_DIR)-$$v \
			MATURIN_PEP517_ARGS="--profile $(GRID_PROFILE)" \
			uv sync --python $$v --group dev --extra examples --no-editable || exit 1 ; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(VENV_DIR)-$$v \
			uv run --no-sync pytest -q || exit 1 ; \
		printf -- "--- polars %s ---\n" "$(GRID_POLARS_MIN)" ; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(VENV_DIR)-$$v \
			uv run --no-sync --with "polars==$(GRID_POLARS_MIN)" pytest -q || exit 1 ; \
	done

# Build release wheels for distribution with architecture-specific optimizations
build:
	RUSTFLAGS="-C target-cpu=apple-m4" maturin build --release --target aarch64-apple-darwin
	RUSTFLAGS="-C target-cpu=icelake-server" maturin build --release --target x86_64-unknown-linux-gnu --zig

#compile:
#	#maturin build --release
#	pip install --force-reinstall target/wheels/*.whl

#compile:
#	maturin build --release
#	cp target/release/libpolars_corpus.dylib polars_corpus/_internal.abi3.so
#	codesign --force --sign - polars_corpus/_internal.abi3.so
#
#debug:
#	maturin build
#	cp target/debug/libpolars_corpus.dylib polars_corpus/_internal.abi3.so
#	codesign --force --sign - polars_corpus/_internal.abi3.so