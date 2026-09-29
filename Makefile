.PHONY: help install lock test test-cov lint format type-check docs clean

help:
	@echo "Apeiron - Development Tasks"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          Sync the project environment and dev dependencies with uv"
	@echo "  make lock             Update uv.lock after changing dependencies in pyproject.toml"
	@echo "  make test             Run tests with pytest"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Run ruff lint and format checks"
	@echo "  make format           Format code with ruff"
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make docs             Build the Sphinx documentation"
	@echo "  make clean            Remove build artifacts and cache files"
	@echo "  make help             Show this help message"

install:
	uv sync

lock:
	uv lock

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=html --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

type-check:
	uv run mypy .

# Docs build in a throwaway environment without the project: the heavy runtime
# deps are mocked in conf.py, so docs/requirements.txt is all that is needed.
# See docs/README.md.
docs:
	uv run --no-project --with-requirements docs/requirements.txt \
		sphinx-build -b html -W docs docs/_build/html
	@echo "Built docs/_build/html/index.html"

clean:
	find . -type f -name '*.py[cod]' -delete
	find . -type f -name '*$$py.class' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf docs/_build/
