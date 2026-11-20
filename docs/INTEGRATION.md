# Local MLflow, container training and batch scoring

This is a local demonstration using fictional data. Docker runs the pinned training/scoring environment; DVC stays on the host. The MLflow UI is the model/run dashboard, and Metaflow cards use its lightweight Local Card Viewer. There is no prediction API or automatic training on page load.

## Start the local dashboard

The Compose file builds the same pinned arm64 Python image for MLflow, training and scoring. It uses a fresh named volume for the MLflow SQLite database and proxied model/run artifacts. The dashboard is published only on loopback at [http://127.0.0.1:5000](http://127.0.0.1:5000).

```sh
docker compose build
docker compose up -d mlflow
docker compose ps
```

`docker compose up -d mlflow` starts only the tracking server. Training and scoring are one-shot commands in the `jobs` profile. The container uses the pinned `linux/arm64` Python base image verified in `docs/ENVIRONMENT.md`; this digest is not claimed to be an amd64 image. MLflow listens inside the container on port 5000, but the host port is bound to `127.0.0.1`; no public network exposure is intended. The tracking server checks host headers for `localhost:5000`, `127.0.0.1:5000`, and the Compose service name.

The `mlflow_data` Docker volume keeps the SQLite database and the `/mlflow/artifacts` store across container restarts. The tracking server serves artifacts through MLflow's artifact proxy, so training/scoring clients use the HTTP tracking URI `http://mlflow:5000` inside Compose. This creates a clean HTTP-backed registry and avoids carrying host-only `file://` artifact paths into a container. The host's previously trained local SQLite runs are separate from this fresh Compose registry.

## Train a candidate

Restore and verify the DVC data on the host first. The flow mounts `data/` read-only and validates the complete export and cohort labels, then uses only the training and validation labels for fitting, threshold selection and release checks. It does not use the held-out test labels.

```sh
uv run dvc pull
uv run python scripts/verify_synthetic_data.py
docker compose --profile jobs run --rm train
```

The command runs the eight-step Metaflow flow and logs its run, metrics, model, gate report and candidate bundle to the HTTP-backed MLflow server. It does not change the `champion` alias. Use a separate explicit host command to promote a candidate that passed the gate:

```sh
uv run python scripts/promote_model.py 1 --tracking-uri http://127.0.0.1:5000
```

Replace `1` with the concrete registered version printed by the flow. Rollback uses the same HTTP tracking URI:

```sh
uv run python scripts/rollback_model.py --tracking-uri http://127.0.0.1:5000
```

Each training run writes Metaflow state into the ignored project folder `.metaflow/`, mounted into the container. To view the local cards, run this in a second host terminal from the project directory:

```sh
METAFLOW_USER=student-mlops \
METAFLOW_DATASTORE_SYSROOT_LOCAL="$PWD" \
METAFLOW_CARD_LOCALROOT="$PWD/.metaflow/mf.cards" \
uv run python flows/train_flow.py card server
```

Open [http://localhost:8324](http://localhost:8324). The viewer updates as a flow completes. The flow and viewer must use the same local datastore path. Stop the viewer with Ctrl+C.

## Score a complete batch

Use a concrete registered version from the MLflow UI or training output. The scorer never follows the mutable `champion` alias during a batch. It verifies the registered version, its gate-passed tag, immutable model URI, deployment bundle, gate report, target/schema versions, and saved threshold before scoring.

```sh
docker compose --profile jobs run --rm score \
  --input /app/data/examples/scoring_input.csv \
  --output /app/outputs/predictions.csv \
  --model-version 1 \
  --tracking-uri http://mlflow:5000 \
  --config /app/config/model.yaml
```

Input is read from the host-mounted, read-only `data/` directory. Output is written to the host `outputs/predictions.csv` file. The scorer validates the entire snapshot export against `docs/LMS_EXPORT_SPEC.md`, derives the same 14 allowlisted features as training, calls the registered sklearn estimator's `predict_proba`, and selects class 1 through the estimator's `classes_` array. It atomically replaces the output only after every row has scored successfully. A bad schema, unavailable model, invalid probability or unwritable destination exits nonzero; no partial CSV is written.

Each result row contains the export join key, `predicted_at`, `dropout_probability`, `risk_band`, `prediction_status`, concrete `model_version`, `feature_schema_version`, `training_data_hash` and `batch_id`. `risk_band` is `review` when probability is at least the saved threshold and `routine` otherwise. The `training_data_hash` is the SHA-256 of the training manifest. These are synthetic interface examples, not predictions about real students.

The scoring job accepts only locally registered models that match the project's release bundle and use MLflow's `skops` serialization with the explicit `numpy.dtype` trust entry. It does not enable MLflow's broad model-code-execution setting. Treat this local registry as trusted project output; do not import arbitrary third-party model artifacts into it.

The training job also keeps `MLFLOW_ALLOW_MODEL_CODE_EXECUTION=false`. With the pinned MLflow release, its optional PyFunc input-example check therefore emits a warning during `log_model`; the flow still writes the native sklearn model, signature and example, and the concrete-version scorer independently checks and loads the skops artifact through its exact `numpy.dtype` allowlist. This warning does not indicate a failed training or scoring run.

## Stop without deleting stored runs

```sh
docker compose stop
```

This stops the containers while preserving the database, artifacts and Metaflow state. Do not use `docker compose down -v` for normal shutdown; `-v` deletes the persistent MLflow volume.

## Verified interfaces

- Docker and Compose: the project's verified local Docker Engine is listed in `docs/ENVIRONMENT.md`; `docker compose up -d mlflow` starts only the selected server service.
- MLflow 3.18: `mlflow server` supports `--backend-store-uri`, `--artifacts-destination`, `--host`, `--port` and `--allowed-hosts`. The HTTP client URI differs inside and outside Compose.
- Metaflow 2.19: `<flow.py> card server` runs the Local Card Viewer on port 8324, and `METAFLOW_DATASTORE_SYSROOT_LOCAL` selects the shared local datastore.
- MLflow sklearn artifacts: the scorer checks the `MLmodel` flavor metadata and uses the explicit skops allowlist before native deserialization. Probabilities come from `predict_proba`, not pyfunc's default class-label prediction.

References: [Docker Compose](https://docs.docker.com/compose/gettingstarted/), [MLflow tracking server](https://mlflow.org/docs/latest/tracking/server/), [MLflow network security](https://mlflow.org/docs/latest/self-hosting/security/network/), [Metaflow Local Card Viewer](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards), [MLflow sklearn flavor](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html), and [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
