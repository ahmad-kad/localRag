.PHONY: setup clean test format lint run build docs

# Variables
PYTHON = python3
VENV = venv
BIN = $(VENV)/bin

setup: $(VENV)/touchfile

$(VENV)/touchfile: requirements.txt requirements-dev.txt
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -r requirements-dev.txt
	touch $(VENV)/touchfile

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .mypy_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

test:
	$(BIN)/pytest tests/ --cov=src --cov-report=term-missing

format:
	$(BIN)/black src/ tests/
	$(BIN)/isort src/ tests/

lint:
	$(BIN)/flake8 src/ tests/
	$(BIN)/mypy src/ tests/
	$(BIN)/black --check src/ tests/
	$(BIN)/isort --check-only src/ tests/

run:
	$(BIN)/python -m src.code_rag.web.app

build:
	$(BIN)/python -m build

docs:
	cd docs && $(MAKE) html

install: clean
	$(BIN)/pip install -e .

update-deps:
	$(BIN)/pip-compile requirements.in
	$(BIN)/pip-compile requirements-dev.in