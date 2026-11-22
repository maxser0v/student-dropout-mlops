# Passage reuse runbook

This separate runtime trains an XGBoost passage reviewer, registers only candidates that pass the validation gate, and scores submissions as JSON Lines. The committed fixture is authored synthetic material; it is for checking the workflow and makes no benchmark or real-world performance claim.

The completed real-data run and custom held-out comparison are documented in [PAN 2013 passage reuse results](PLAGIARISM_RESULTS.md).

## Run the end-to-end fixture workflow

The isolated runtime needs Python 3.11, `uv`, and Docker Compose. Resolve the runtime dependencies when the package indexes are reachable, then create host directories writable by the numeric UID/GID used inside the containers:

```sh
make plagiarism-runtime-lock
export PROJECT_ROOT="$PWD"
export RUNTIME_UID="$(id -u)"
export RUNTIME_GID="$(id -g)"
mkdir -p reports/plagiarism artifacts/plagiarism
make plagiarism-runtime-image
docker compose -p plagiarism-fixture build mlflow
```

`make plagiarism-runtime-lock` exports a normal runtime lock and a CI lock with the pytest test group. Both exports preserve the explicit PyTorch CPU index with `uv export --emit-index-url`; the Docker image installs only the runtime lock with pip hash checks. If package-index DNS is unavailable, lock generation and image build must wait until it is restored.

Start an isolated MLflow registry and run the fixture workflow:

```sh
docker compose -p plagiarism-fixture --profile init run --rm volume-init
docker compose -p plagiarism-fixture --profile jobs up -d mlflow
docker compose -p plagiarism-fixture --profile jobs run --rm plagiarism-fixture-workflow
```

The job runs the Metaflow flow twice with the actual XGBoost implementation, verifies an intentionally strict validation-gate failure, promotes each passing concrete version explicitly, scores the same authored query before and after rollback, and asserts the JSONL output returns to its original bytes and concrete version. It also checks for a flagged passage overlapping an authored annotation and writes `workflow-summary.json` with validation baseline/candidate false-alert and recall counts, promoted versions, and rollback outcome. It stops with an error if either ordinary fixture run does not pass the configured gate. Reports are under `reports/plagiarism/fixture-workflow/`; Metaflow state and encoder caches are under `artifacts/plagiarism/`. The `jobs` profile includes training, scoring, and encoder smoke services.

Compose jobs run as `${RUNTIME_UID}:${RUNTIME_GID}` and write their Metaflow datastore/card files under the host-mapped `artifacts/plagiarism` directory. Keep these numeric IDs aligned with the account that owns `reports/` and `artifacts/plagiarism/`; do not run jobs with the default image UID if those bind mounts are not writable.

The same orchestrator can run directly after `uv sync --directory plagiarism-runtime --group test`:

```sh
uv run --directory plagiarism-runtime python "$PWD/scripts/plagiarism_fixture_workflow.py" \
  --tracking-uri sqlite:///reports/plagiarism/fixture-workflow.db \
  --output-dir reports/plagiarism/fixture-workflow
```

Use a fresh tracking database and output directory for a new rehearsal. The workflow deliberately refuses to overwrite a release history.

## Score with a concrete release

The fixture job accepts JSONL records with only `submission_id` and `text`. It emits one JSON record per submission; each record names the concrete model version and source-index hash, and includes flagged excerpts and offsets for human review.

```sh
cat > reports/plagiarism/submissions.jsonl <<'JSONL'
{"submission_id":"review-001","text":"A reliable experiment keeps a final test set untouched."}
JSONL
docker compose -p plagiarism-fixture --profile jobs run --rm plagiarism-score \
  --input /app/reports/plagiarism/submissions.jsonl \
  --output /app/reports/plagiarism/review-output.jsonl \
  --tracking-uri http://mlflow:5000 \
  --model-name passage-reuse-fixture --version champion --fixture
```

The scorer resolves `champion` once, then checks the registered bundle, feature transforms, source index, gate status, and encoder provenance before returning JSONL. A concrete `--version N` can be used instead of the alias.

## Use PAN data and frozen MiniLM

PAN is not bundled. Obtain the archive, check its published MD5 and current terms, then use [the corpus preparation guide](PLAGIARISM_DATA.md). `dvc add plagiarism_data` creates `plagiarism_data.dvc`; this pointer is mounted only by the real-data training job. Restore the corpus with `dvc pull plagiarism_data.dvc` before starting that job. In restricted environments set `DVC_SITE_CACHE_DIR="$PWD/.dvc/tmp/plagiarism-site-cache"` before `dvc add`, `dvc push`, or `dvc pull`.

Stage the configured immutable MiniLM revision once while outbound access is available, then smoke-check that exact staged snapshot offline:

```sh
mkdir -p models/minilm
docker compose -p plagiarism-fixture --profile staging run --rm plagiarism-stage-encoder
docker compose -p plagiarism-fixture --profile jobs run --rm plagiarism-encoder-smoke
```

Staging allows only Sentence Transformers' required tokenizer/configuration files and one PyTorch weight file; ONNX, OpenVINO, and alternate model exports are excluded. Runtime jobs have Hugging Face offline mode enabled and verify the manifest hashes plus required file set before loading MiniLM on CPU.

Run the real-data service only after the DVC corpus and pinned encoder are present:

```sh
docker compose -p plagiarism-fixture --profile jobs run --rm plagiarism-train
```

Do not use the authored fixture results as PAN evaluation evidence. The final PAN test split is held out for one evaluation after a candidate specification is frozen.

After reviewing the validation report and choosing one concrete, gate-passing PAN model version, invoke the one-time held-out evaluation with that version (never the mutable alias):

```sh
docker compose -p plagiarism-fixture --profile jobs run --rm plagiarism-train \
  python scripts/evaluate_plagiarism_final_test.py \
  --data-dir /app/plagiarism_data/pan13.json \
  --tracking-uri http://mlflow:5000 \
  --model-name passage-reuse --version "$FROZEN_VERSION" \
  --config /app/config/plagiarism.yaml --encoder-dir /app/models/minilm
```

The final evaluator writes a freeze record, an attempt record, and one report under `reports/plagiarism/`. Those paths are single-use; do not rerun the test split after a started or completed attempt.

## CI behavior and cleanup

The `plagiarism-fixture` CI job resolves the isolated Python 3.11 runtime with the test group, runs focused plagiarism unit checks, then executes the same fixture release workflow using actual XGBoost and a temporary SQLite tracking store. It downloads no PAN corpus or MiniLM model.

Stop the local MLflow service while retaining its dedicated registry volume:

```sh
docker compose -p plagiarism-fixture down
```
