.PHONY: help install dev lint format test check build clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev:  ## Install with dev + full dependencies
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,full]"

lint:  ## Run linter
	ruff check conflux/ tests/

format:  ## Format code
	ruff format conflux/ tests/

test:  ## Run tests
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=conflux --cov-report=term-missing --cov-report=html

typecheck:  ## Run type checker
	mypy conflux/ --ignore-missing-imports --no-strict-optional

syntax:  ## Validate syntax of all modules
	@python -c "\
	import py_compile, pathlib; \
	[print(f'  ✓ {f}') if not py_compile.compile(str(f), doraise=True) or True else None \
	 for f in sorted(pathlib.Path('conflux').rglob('*.py'))]"

check: lint syntax test  ## Run all checks (lint + syntax + test)
	@echo "\n✓ All checks passed"

build:  ## Build distribution package
	python -m build
	twine check dist/*

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf conflux_cache/ conflux_output*/
