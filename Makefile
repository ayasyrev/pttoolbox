.ONESHELL:
SHELL := /bin/bash

.PHONY: sync lint test pre-commit dist publish clean

sync:
	uv sync --locked

lint:
	uv run ruff format --check .
	uv run ruff check .

test:
	uv run pytest

pre-commit:
	uv run pre-commit run --all-files

publish: dist
	uv publish dist/*

dist: clean
	uv build --no-sources

clean:
	rm -rf dist
