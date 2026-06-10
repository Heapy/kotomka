PORT ?= 8000

.DEFAULT_GOAL := serve
.PHONY: serve sync test

serve:
	uv run kotomka serve --port $(PORT)

sync:
	uv sync --extra dev

test:
	uv run pytest
