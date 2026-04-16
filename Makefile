PYTHON := python3
UV := uv
SRC := src
PROG := main

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make run         - Run the application"
	@echo "  make clean       - Clean temporary files"
	@echo "  make clean-all   - Remove the virtual environment"
	@echo "  make debug       - Run the application in debug mode"
	@echo "  make lint        - Run linters and type checkers"
	@echo "  make lint-strict - Run linters and type checkers in strict mode"

install:
	@$(UV) sync

run: serve
	@$(UV) run $(PYTHON) -m $(SRC).$(PROG) $(ARGS)

debug:
	@$(PYTHON) -m pdb $(SRC)/$(PROG).py

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@ind . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete

serve:
	@pgrep -x "ollama" > /dev/null || nohup ollama serve > /dev/null 2>&1 &

clean-all:
	@rm -rf .venv/

lint:
	@$(UV) run $(PYTHON) -m flake8 src/*.py src/classes/*.py
	@$(UV) run $(PYTHON) -m mypy src/*.py src/classes/*.py \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \

lint-strict:
	@$(UV) run $(PYTHON) -m flake8 src/*.py src/classes/*.py
	@$(UV) run $(PYTHON) -m mypy src/*.py src/classes/*.py --strict

.PHONY: help install clean-all run clean debug lint lint-strict serve