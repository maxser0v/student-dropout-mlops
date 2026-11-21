# One-command local demo and container runtime

This project demonstrates a local model lifecycle with deterministic fictional data. Docker supplies the pinned training, scoring, and MLflow environment; DVC remains a host-side tool. The demo registers a candidate but never promotes it or evaluates the held-out final cohort. It provides no prediction API or automatic training on page load.

## Run the complete demo

Prerequisites are Docker with Compose, `uv`, and a non-root host account. From the project directory, run:

```sh
make demo
```

`uv run` prepares the locked Python environment. The runner then:

1. Checks Docker/Compose and the local project files.
2. Generates the deterministic fictional dataset only if `data/` does not exist. If any dataset already exists, it is never overwritten. The runner stops if its manifest says it is not synthetic or if its contract/checksums fail.
3. Verifies the complete `data/` file set and hashes against the DVC `.dir` pointer in `data.dvc`, without relying on a DVC cache or remote.
4. Records the actual Git HEAD and dirty state where Git metadata exists, hashes the source/config/build/docs/test files, builds the ARM64 image, checks its OCI provenance labels, and obtains the exact Docker image ID.
5. Creates a unique Compose project and named MLflow volume, starts the MLflow server on loopback, waits for its health check, and verifies that MLflow, training, and scoring use a non-root UID.
6. Runs the eight-step Metaflow flow with the actual source and image metadata, reads its machine result, and scores the gate-passing concrete candidate version.

The dashboard URL, concrete model version and run IDs, provenance, DVC check, and output paths are printed as JSON. The same result is saved as `reports/demo_runs/<demo-id>/demo-result.json`. The eight-step flow writes its own candidate reports under `reports/<metaflow-run-id>/`; predictions go to `outputs/demo_runs/<demo-id>/predictions.csv`. Each invocation uses a fresh UUID-based directory, image tag, Compose project, and volume, so existing reports and the existing MLflow registry remain separate.

The default dashboard port is `5000`. If it is already in use, the runner selects another available loopback port and adds it to MLflow's allowed host list. The service stays running for inspection. To stop only the new Compose project and keep its data volume, copy the exact `docker compose ... down` command printed at the end; do not add `-v` if you want to preserve the registry and artifacts. If a run fails after MLflow starts, the runner stops only its own project without deleting its volume and leaves logs/reports in that run's folder.

## Local settings and user identity

`scripts/run_demo.py` reads `.env.example` automatically; no manual copy is needed. A plain `.env` file can override these settings:

| Setting | Default | Effect |
| --- | --- | --- |
| `DEMO_MLFLOW_PORT` | `5000` | Preferred loopback port; a free port is selected when busy. |
| `DOCKER_PLATFORM` | `linux/arm64` | Matches the pinned Python base-image digest. Other architectures are not claimed. |
| `RUNTIME_UID` | `auto` | Uses the current non-root host UID for writable bind mounts. |
| `RUNTIME_GID` | `auto` | Uses the current host primary GID. |
| `METAFLOW_USER` | `student-mlops` | Metaflow run metadata identity; the actual process UID remains separately enforced. |

Values are parsed as plain `KEY=value` text; shell expressions are rejected and never executed. All application containers use a nonzero UID. The image creates a passwd entry and home directory for the selected runtime identity. A root-only `volume-init` job changes ownership only inside the isolated project's persistent `/mlflow` volume; it does not delete or recreate database/artifact files. The demo uses a fresh project volume, so initialization cannot race the existing MLflow service. For a manual migration of a volume used by a running MLflow service, stop that service first, then run the initializer before restarting it.

Manual Compose commands should set `RUNTIME_UID` and `RUNTIME_GID` to the current nonzero host IDs so bind-mounted `reports/`, `outputs/`, and `.metaflow/` stay writable. The `make demo` runner derives and passes these values automatically. `METAFLOW_USER` is only the flow's display identity; it does not set the operating-system process user.

## Provenance

The runner hashes the current source tree deterministically over project source, config, tests, docs, CI, build, and lock files; it excludes datasets, run outputs, caches, and secrets. The image records `org.opencontainers.image.revision`, `org.opencontainers.image.revision.dirty`, `org.opencontainers.image.source-tree.sha256`, the source label, and the hash algorithm. Before training, the runner compares those labels to its freshly computed values, inspects the exact image ID, and injects that ID plus source metadata into the training container. A source archive without `.git` records the commit and dirty state as `unknown` while retaining a real source-tree digest; it never fabricates a clean revision.

Manual raw image builds without verified metadata use explicit `unknown`/`unspecified` label defaults. The CI image build computes and verifies its checked-out commit and tree hash before setting those labels.

## Direct Compose operation

`make demo` is the supported one-command path. For manual use, build the image and initialize the named volume before starting MLflow:

```sh
export RUNTIME_UID="$(id -u)"
export RUNTIME_GID="$(id -g)"
docker compose build mlflow
docker compose --profile init run --rm volume-init
docker compose up -d mlflow
docker compose ps
```

The tracking dashboard is published only on `127.0.0.1`; inside Compose, clients use `http://mlflow:5000`. The SQLite database and proxied model/run artifacts persist in the named `mlflow_data` volume. A direct Compose user changing ownership of an existing volume must stop its current MLflow server before running `volume-init`. Normal shutdown uses `docker compose stop` and does not delete the volume. Avoid `docker compose down -v` unless deleting the registry is intended.

MLflow validates its allowed host headers for both internal `mlflow:5000` traffic and the configured loopback host port. The host port is not bound to a public interface.

## Metaflow cards

The containerized flow writes local Metaflow state and cards into the same absolute project path mounted from the host. To view cards in a separate host terminal, run:

```sh
METAFLOW_USER=student-mlops \
METAFLOW_DATASTORE_SYSROOT_LOCAL="$PWD" \
METAFLOW_CARD_LOCALROOT="$PWD/.metaflow/mf.cards" \
uv run python flows/train_flow.py card server
```

Open [http://localhost:8324](http://localhost:8324). The lightweight Local Card Viewer displays saved step cards; it does not schedule flows or alter the MLflow registry.

## Batch scoring

Use a concrete gate-passed model version from the training result or MLflow dashboard. The scorer never follows the mutable `champion` alias. It verifies the registry version, release gate, bundle, configured target/schema versions, threshold, and skops serialization metadata before loading the native pipeline with the explicit `numpy.dtype` allowlist.

For the isolated project created by `make demo`, its `.env` file is saved in the result folder. A manual score job uses the same Compose project and config file:

```sh
docker compose --env-file reports/demo_runs/<demo-id>/compose.env \
  --project-name <compose-project> --profile jobs run --rm score \
  --input /app/data/examples/scoring_input.csv \
  --output /app/outputs/manual-predictions.csv \
  --model-version <concrete-version> \
  --tracking-uri http://mlflow:5000 \
  --config /app/config/model.yaml
```

The output is written atomically only after the entire CSV validates and scores. Each result row includes the export join key, prediction timestamp and probability, risk band, status, concrete model version, feature-schema version, training manifest hash, and batch ID. `risk_band` is `review` when probability is at least the saved candidate threshold and `routine` otherwise. These are synthetic interface examples, not predictions about real students.

With MLflow 3.18's broad model-code execution switch disabled, its optional PyFunc input-example validation logs a warning when the registered sklearn example is loaded. The project scorer uses native skops loading after checking the artifact's exact serialization format and trusted-type list; the warning does not indicate a failed training or scoring run.

## Candidate policy and historical model

New candidate runs use configuration version 1.1. Its validation minimum recall is 0.80; other release gates are unchanged. The previous configuration 1.0 is preserved exactly as `config/model-v1.yaml` and remains the recorded configuration for the original frozen model and its once-only final assessment. The final cohort is not consumed by `make demo` or ordinary candidate training.

References: [Docker Compose](https://docs.docker.com/compose/gettingstarted/), [Docker volumes](https://docs.docker.com/engine/storage/volumes/), [MLflow tracking server](https://mlflow.org/docs/latest/tracking/server/), [MLflow network security](https://mlflow.org/docs/latest/self-hosting/security/network/), [Metaflow Local Card Viewer](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards), [MLflow sklearn flavor](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html), and [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
