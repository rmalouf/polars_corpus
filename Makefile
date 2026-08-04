# Makefile for managing Python dependencies with uv

# The venv lives outside the source tree (see CLAUDE.md) and is activated by a
# shell extension. Dependencies are declared in pyproject.toml; re-resolve with
# `uv lock`.
VENV_DIR = ${HOME}/.venvs/polars_corpus

.PHONY: develop develop-release build docs

# Serve docs locally
docs:
	quarto convert user_guide/04-frequencies.ipynb
	great-docs build

# Rebuild the Rust extension for the edit/test loop. Optimized and native-CPU,
# but without LTO, so this takes seconds rather than minutes. Use this one.
develop:
	RUSTFLAGS="-C target-cpu=native" maturin develop --profile dev-fast

# Same, but with the full release profile. Only needed when benchmarking or
# checking something that depends on LTO; `make develop` is the normal path.
develop-release:
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