# Makefile for managing Python dependencies with uv

.PHONY: install dev lock clean

## Install production dependencies
install:
	uv pip sync requirements.txt

## Install dev dependencies
dev:
	uv pip sync requirements-examples.txt 

## Compile locked requirements
locks:
	uv pip compile requirements.in >requirements.txt
	uv pip compile requirements.in requirements-dev.in >requirements-dev.txt
	uv pip compile requirements.in requirements-examples.in requirements-dev.in >requirements-examples.txt

## Clean compiled files
clean:
	rm -f requirements.txt requirements-dev.txt

#compile:
#	#maturin build --release
#	pip install --force-reinstall target/wheels/*.whl

compile:
	maturin build --release
	cp target/release/libpolars_corpus.dylib polars_corpus/_internal.abi3.so
	codesign --force --sign - polars_corpus/_internal.abi3.so

debug:
	maturin build
	cp target/debug/libpolars_corpus.dylib polars_corpus/_internal.abi3.so
	codesign --force --sign - polars_corpus/_internal.abi3.so