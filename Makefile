.PHONY: install install-backend install-frontend dev api ui test test-backend test-frontend lint build serve hygiene clean-data

install: install-backend install-frontend

install-backend:
	cd backend && uv sync --all-extras

install-frontend:
	cd frontend && npm ci

# API on :8000 (auto-reload) and Vite dev server on :5173 (proxies /api).
dev:
	$(MAKE) -j2 api ui

api:
	cd backend && uv run uvicorn safeframe.api.app:create_app --factory --reload --port 8000

ui:
	cd frontend && npm run dev

test: test-backend test-frontend

test-backend:
	cd backend && uv run pytest -q

# Slow: downloads models and runs them on a synthetic clip.
test-models:
	cd backend && uv run pytest -q -m models

test-frontend:
	cd frontend && npm test

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .
	cd frontend && npm run typecheck

build:
	cd frontend && npm run build

# Production-style: build the UI and serve it from FastAPI on :8000.
serve: build
	cd backend && uv run safeframe serve

hygiene:
	python3 scripts/check_forbidden_files.py --all

# Delete all job data (uploads, previews, outputs) from this machine.
clean-data:
	rm -rf backend/data data
