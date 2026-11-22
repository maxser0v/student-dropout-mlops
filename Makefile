.PHONY: demo plagiarism-runtime-lock plagiarism-runtime-image plagiarism-encoder-stage plagiarism-encoder-smoke

demo:
	uv run python scripts/run_demo.py

# Resolve only the isolated plagiarism target; never rewrite the dropout lock.
plagiarism-runtime-lock:
	UV_CACHE_DIR=/private/tmp/uvcache-plagiarism uv lock --directory plagiarism-runtime
	UV_CACHE_DIR=/private/tmp/uvcache-plagiarism uv export --directory plagiarism-runtime --locked --no-emit-project --emit-index-url --format requirements-txt -o requirements.lock.txt
	UV_CACHE_DIR=/private/tmp/uvcache-plagiarism uv export --directory plagiarism-runtime --locked --no-emit-project --group test --emit-index-url --format requirements-txt -o requirements-ci.lock.txt

plagiarism-runtime-image:
	docker compose --profile jobs build plagiarism-train-fixture plagiarism-fixture-workflow plagiarism-train plagiarism-score plagiarism-encoder-smoke

plagiarism-encoder-stage:
	mkdir -p models/minilm
	docker compose --profile staging run --rm plagiarism-stage-encoder

plagiarism-encoder-smoke:
	docker compose --profile jobs run --rm plagiarism-encoder-smoke
