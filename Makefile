# Makefile for managing Python dependencies with uv

.PHONY: install dev lock clean

## Install production dependencies
install:
	uv pip sync requirements.txt

## Install dev dependencies
dev:
	uv pip sync requirements.txt requirements-examples.txt requirements-dev.txt

## Compile locked requirements
locks:
	uv pip compile requirements.in >requirements.txt
	uv pip compile requirements.in requirements-examples.in >requirements-examples.txt
	uv pip compile requirements.in requirements-examples.in requirements-dev.in >requirements-dev.txt

## Clean compiled files
clean:
	rm -f requirements.txt requirements-dev.txt

compile:
	maturin build --release
	pip install --force-reinstall target/wheels/*.whl
