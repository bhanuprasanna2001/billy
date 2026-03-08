.PHONY: help install test lint up down build logs clean train demo

# ── Help ────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Local dev ───────────────────────────────────────────────────
install: ## Install package + dev deps locally
	pip install -e ".[dev]"
	pre-commit install

test: ## Run pytest
	pytest -v --tb=short

lint: ## Lint, type-check, and format
	ruff check .
	ruff format --check .
	mypy src

# ── Docker ──────────────────────────────────────────────────────
up: ## Start all 11 services
	docker compose up -d --build

down: ## Stop all services
	docker compose down

build: ## Build all services
	docker compose build

logs: ## Tail logs (make logs s=api)
	docker compose logs -f $(s)

clean: ## Stop services + remove volumes
	docker compose down -v

# ── ML pipeline ─────────────────────────────────────────────────
train: ## Train model and register in MLflow
	python scripts/train_model.py

demo: ## Run the full demo walkthrough
	bash scripts/demo_walkthrough.sh
