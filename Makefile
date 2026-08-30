VENV_DIR = ${HOME}/.venvs/polars_corpus

.PHONY: develop develop-release build docs grid

GRID_VERSIONS = 3.11 3.12 3.13 3.14
GRID_PROFILE ?= dev-fast
# Oldest supported polars version (also in pyproject.toml and .github/workflows/release.yml)
GRID_POLARS_MIN = 1.36.*

docs:
	mkdocs build

develop:
	RUSTFLAGS="-C target-cpu=native" maturin develop --profile dev-fast

develop-release:
	RUSTFLAGS="-C target-cpu=native" maturin develop --release

grid:
	@for v in $(GRID_VERSIONS); do \
		printf "\n=== Python %s ===\n" $$v ; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(VENV_DIR)-$$v \
			MATURIN_PEP517_ARGS="--profile $(GRID_PROFILE)" \
			uv sync --python $$v --group dev --no-editable || exit 1 ; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(VENV_DIR)-$$v \
			uv run --no-sync pytest -q || exit 1 ; \
		printf -- "--- polars %s ---\n" "$(GRID_POLARS_MIN)" ; \
		env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(VENV_DIR)-$$v \
			uv run --no-sync --with "polars==$(GRID_POLARS_MIN)" pytest -q || exit 1 ; \
	done

# Distribution wheels: the full release profile plus the codegen flags in .cargo/config.toml
build:
	maturin build --release --target aarch64-apple-darwin
	maturin build --release --target x86_64-unknown-linux-gnu --zig
