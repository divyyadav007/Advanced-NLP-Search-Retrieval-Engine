.PHONY: setup install run-backend run-frontend test lint format clean help

PYTHON ?= python
PIP ?= pip

help:
	@echo "======================================================================="
	@echo "                     Enterprise Hybrid RAG Makefile"
	@echo "======================================================================="
	@echo "setup         - Create virtual environment and install dependencies"
	@echo "install       - Install dependencies in existing environment"
	@echo "run-backend   - Run the FastAPI backend service"
	@echo "run-frontend  - Run the Streamlit frontend interface"
	@echo "test          - Execute test suite using pytest"
	@echo "lint          - Perform static analysis checks via ruff"
	@echo "format        - Reformat code using black"
	@echo "clean         - Remove temp files and build caches"
	@echo "======================================================================="

setup:
	python -m venv myvenv
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt
	$(PIP) install -r frontend/requirements.txt

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt
	$(PIP) install -r frontend/requirements.txt

run-backend:
	$(PYTHON) backend/run.py

run-frontend:
	$(PYTHON) frontend/run.py

test:
	$(PYTHON) -m pytest backend/tests/ -v

lint:
	$(PYTHON) -m ruff check backend/app/ backend/tests/ frontend/src/

format:
	$(PYTHON) -m black backend/app/ backend/tests/ frontend/src/

clean:
	rm -rf myvenv/ .venv/ build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete
