.PHONY: install test lint format run-ui help

help: ## Show available make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies via uv
	uv sync --all-extras

test: ## Run pytest suite
	uv run pytest -v

lint: ## Run ruff linter
	uv run ruff check src tests

format: ## Format code with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

run-ui: ## Launch Streamlit workbench
	uv run streamlit run src/subllm/app.py
