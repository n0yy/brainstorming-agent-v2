.PHONY: setup-db

setup-db:
	mkdir -p .uv-cache
	UV_CACHE_DIR=.uv-cache uv run python -m scripts.setup_resources
