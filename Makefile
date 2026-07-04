# Makefile for managing Python dependencies with uv

VENV_DIR = ${HOME}/.venvs/polars-corpus

.PHONY: install dev lock clean docs

venv:
	uv venv --allow-existing --quiet $(VENV_DIR)
	source $(VENV_DIR)/bin/activate
	uv pip sync requirements-examples.txt

# Regenerate all requirement lock files from .in sources
# Three levels: runtime (requirements.txt) → dev → examples
locks:
	uv pip compile requirements.in >requirements.txt
	uv pip compile requirements.in requirements-dev.in >requirements-dev.txt
	uv pip compile requirements.in requirements-examples.in requirements-dev.in >requirements-examples.txt

# Serve docs locally
docs:
	quarto convert user_guide/04-frequencies.ipynb
	great-docs build

# Build Rust extension for local development with native CPU optimizations
develop:
	RUSTFLAGS="-C target-cpu=native" maturin develop --release

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