VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit

SOURCE ?= local
DATA_PATH ?= data/handbook

.DEFAULT_GOAL := help

.PHONY: help venv install hooks lint format check test index run eval clean

help: ## Show available "make" targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv ## Install project and dev dependencies into the virtual environment
	$(PIP) install -e ".[dev]"

hooks: ## Install pre-commit git hooks
	$(PRECOMMIT) install

lint: ## Run Ruff linter (auto-fixes safe issues like unused/unsorted imports)
	$(RUFF) check --fix .

format: ## Format code with Ruff (line-length = 120)
	$(RUFF) format .

check: ## Run CI checks (Ruff lint, format check, and pytest)
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYTEST) -v

test: ## Run pytest
	$(PYTHON) -m pytest -v --cov --cov-report=term-missing --cov-fail-under=80

index: ## Build/reset the vector index (override: make index SOURCE=confluence)
	$(PYTHON) ingest.py --source $(SOURCE) --path $(DATA_PATH) --reset

run: ## Launch the Gradio web app
	$(PYTHON) -m app

eval: ## Run the RAG evaluation benchmark
	$(PYTHON) eval/run_eval.py

clean: ## Remove build, cache, and test artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage .coverage.* htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
