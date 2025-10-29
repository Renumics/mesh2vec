SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

.PHONY: help
help: ## Print this help message
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

.PHONY: init
init: ## Locally install all dev dependencies
init:
	poetry install

.PHONY: check-format
check-format: ## Check code formatting
	poetry run black --check docs mesh2vec scripts tests

.PHONY: format
format: ## Fix code formatting
	poetry run black docs mesh2vec scripts tests

.PHONY: lint
lint: ## Lint source files
	poetry run pylint docs/examples mesh2vec scripts tests

.PHONY: typecheck
typecheck: ## Typecheck source files
	poetry run mypy docs/examples mesh2vec scripts tests

.PHONY: audit
audit: ## Audit project dependencies
	poetry run pip-audit

.PHONY: check
check: ## Run all checks
check: check-format lint typecheck audit

.PHONY: doctest
doctest: ## Run doctests
	poetry run sphinx-build docs/source build/documentation/ -W -b html
	poetry run sphinx-build docs/source build/documentation/ -W -b doctest

.PHONY: unit-test
unit-test: ## Run unit tests
	poetry run pytest -s tests

.PHONY: test
test: ## Run all tests
test: unit-test doctest

.PHONY: wheel
wheel: ## Build the project wheel
	poetry build -f wheel -vvv
	poetry run check-wheel-contents dist/

.PHONY: docs
docs: ## Build documentation
	poetry run sphinx-build docs/source build/documentation/ -W -b html

.PHONY: build
build: ## Build all distributables
build: wheel docs
