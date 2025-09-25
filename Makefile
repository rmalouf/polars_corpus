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
	maturin develop --release

build:
	maturin build --release --target aarch64-apple-darwin

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