# Makefile for managing Python dependencies with uv

.PHONY: install dev lock clean

venv:
	uv venv --allow-existing --quiet
	uv pip sync requirements-examples.txt

locks:
	uv pip compile requirements.in >requirements.txt
	uv pip compile requirements.in requirements-dev.in >requirements-dev.txt
	uv pip compile requirements.in requirements-examples.in requirements-dev.in >requirements-examples.txt

develop:
	RUSTFLAGS="-C target-cpu=native" maturin develop --release

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