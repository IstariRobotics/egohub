.PHONY: help lint test

help:
	@echo "Commands:"
	@echo "  lint   : Run all static analysis (ruff, black, mypy)."
	@echo "  test   : Run the core unit test suite."
	@echo "  install: Install project in editable mode with dev dependencies."

# ==============================================================================
# Development
# ==============================================================================
install:
	.venv/bin/python -m pip install -e ".[dev,full]"
	pre-commit install

# ==============================================================================
# Quality
# ==============================================================================
lint:
	@echo "--- Running Ruff linter ---"
	ruff check egohub
	@echo "\n--- Checking Black formatting ---"
	black --check .
	@echo "\n--- Running Mypy type checker ---"
	mypy egohub

test:
	python -m pytest -m "not integration and not e2e and not perf" -n auto --cov=egohub 