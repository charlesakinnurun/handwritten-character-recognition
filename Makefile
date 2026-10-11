# Handwriting Intelligence - developer task runner.
# Usage: `make <target>`. On Windows, see README for the equivalent raw commands.
PYTHON ?= python
CONFIG ?= configs/mnist_cnn.yaml
ARTIFACT ?= models/mnist_cnn/artifact.pt

.PHONY: help install install-dev lint format typecheck test test-unit test-integration test-smoke \
        cov download train evaluate predict api demo docker-build docker-run clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime deps only).
	$(PYTHON) -m pip install -e .

install-dev: ## Install the package with development extras.
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run Ruff lint checks.
	$(PYTHON) -m ruff check src tests scripts app

format: ## Auto-format and auto-fix with Ruff.
	$(PYTHON) -m ruff format src tests scripts app
	$(PYTHON) -m ruff check --fix src tests scripts app

typecheck: ## Run MyPy over the package.
	$(PYTHON) -m mypy src

test: ## Run the full test suite (unit + integration + smoke).
	$(PYTHON) -m pytest

test-unit: ## Run unit tests only.
	$(PYTHON) -m pytest tests/unit -m "not dataset"

test-integration: ## Run integration tests only.
	$(PYTHON) -m pytest tests/integration -m "not dataset"

test-smoke: ## Run smoke tests only.
	$(PYTHON) -m pytest tests/smoke

cov: ## Run tests with a coverage report.
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml

download: ## Download / verify datasets described in the config.
	$(PYTHON) -m handwriting_intelligence.cli.download --config $(CONFIG)

train: ## Train using CONFIG=configs/<file>.yaml.
	$(PYTHON) -m handwriting_intelligence.cli.train --config $(CONFIG)

evaluate: ## Evaluate a trained artifact (ARTIFACT=...).
	$(PYTHON) -m handwriting_intelligence.cli.evaluate --artifact $(ARTIFACT) --config $(CONFIG)

predict: ## Predict a single image (IMAGE=path/to/img.png).
	$(PYTHON) -m handwriting_intelligence.cli.predict --artifact $(ARTIFACT) --image $(IMAGE)

api: ## Run the FastAPI inference server.
	$(PYTHON) -m uvicorn handwriting_intelligence.api.main:app --host 0.0.0.0 --port 8000

demo: ## Run the Streamlit demo.
	$(PYTHON) -m streamlit run app/streamlit_app.py

docker-build: ## Build the API container image.
	docker build -t handwriting-intelligence:latest .

docker-run: ## Run the API container.
	docker run --rm -p 8000:8000 -v "$(CURDIR)/models:/app/models:ro" handwriting-intelligence:latest

clean: ## Remove caches and build artifacts (keeps datasets and models).
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','.mypy_cache','.ruff_cache','htmlcov','.coverage']]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('src').rglob('__pycache__')]"
