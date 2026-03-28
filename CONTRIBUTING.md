# Contributing to CONFLUX

Thank you for your interest in CONFLUX. This document covers the development workflow and contribution guidelines.

## Development Setup

```bash
git clone https://github.com/nage-ai/conflux.git
cd conflux

python -m venv .venv
source .venv/bin/activate

# CPU-only torch for development (faster install)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,full]"
```

## Development Workflow

```bash
# Lint
ruff check conflux/ tests/

# Format
ruff format conflux/ tests/

# Test
pytest tests/ -v

# Test with coverage
pytest tests/ -v --cov=conflux --cov-report=term-missing

# Type check
mypy conflux/ --ignore-missing-imports

# All checks (recommended before pushing)
make check
```

## Code Standards

- Python 3.10+ with type hints on all public functions
- Docstrings on all public classes and functions (NumPy style)
- Maximum line length: 120 characters
- All new modules must have corresponding tests in `tests/`

## Pull Request Process

1. Fork the repository and create a feature branch from `dev`
2. Write tests for new functionality
3. Ensure all CI checks pass (`make check`)
4. Update README.md if adding new public API
5. Submit PR against `dev` branch

## Module Architecture

When adding new functionality, follow the existing module pattern:

- Each module is a single file in `conflux/`
- Public API is exported through `__init__.py`
- Heavy dependencies (transformers, peft) are imported lazily inside functions
- Core math (CKA, SVD, loss) depends only on torch + numpy

## Release Process

Maintainers only:

```bash
# 1. Update version in pyproject.toml and conflux/__init__.py
# 2. Commit and push to main
# 3. Tag the release
git tag v0.2.0
git push origin v0.2.0
# 4. GitHub Actions handles PyPI publish + GitHub Release
```

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
