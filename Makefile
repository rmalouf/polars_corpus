# Makefile for managing Python dependencies with uv

.PHONY: install dev lock clean

## Install production dependencies
install:
	uv pip sync requirements.txt

## Install dev dependencies
dev:
	uv pip sync requirements.txt requirements-dev.txt

## Compile locked requirements
lock:
	uv pip compile requirements.in >requirements.txt
	uv pip compile requirements-dev.in >requirements-dev.txt

## Clean compiled files
clean:
	rm -f requirements.txt requirements-dev.txt
