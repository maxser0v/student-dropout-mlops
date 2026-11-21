# Local environment

## Architecture

```mermaid
flowchart LR
    Git[Git code and DVC pointers]
    DVC[DVC data snapshots]
    MF[Metaflow local training flow]
    ML[MLflow runs and model registry]
    Docker[Docker training and batch scoring]
    CI[GitHub Actions checks and image build]
    Data[(Synthetic LMS export)]
    Git --> MF
    DVC --> Data
    Data --> MF
    MF --> ML
    Docker --> ML
    CI --> Git
```

Git records code/configuration and DVC pointer history. DVC versions input files and provides local restore/push/pull mechanics. Metaflow runs the linear training workflow. MLflow records experiments and model versions. Docker supplies repeatable runtimes for training and scoring. GitHub Actions runs checks and builds the image. DVC is not a workflow engine; MLflow is not a scheduler.

## Verified machine and package versions

Verified on 2026-11-20 in the project Python 3.11 environment on macOS arm64. Docker CLI and daemon were reachable. All dependencies are locked in `uv.lock`; `requirements.lock.txt` is the hash-pinned pip export.

| Component | Verified version |
|---|---:|
| Python | 3.11.7 |
| uv | 0.12.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| MLflow | 3.18.0 |
| Metaflow | 2.19.41 |
| DVC | 3.67.1 |
| boto3 (Metaflow dependency pin) | 1.44.7 |
| pytest | 9.2.0 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| Docker Engine | 28.5.2 |
| Docker base image | `python:3.11-slim-bookworm`, pulled as `python@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39` (`linux/arm64`) |

Metaflow 2.19.41 declares an unbounded `boto3` dependency. The direct boto3 pin prevents the resolver from selecting the unrelated historical 0.0.8 release. The complete resolved dependency tree is pinned in the lock files.

## Create and use the environment

```sh
uv python pin 3.11
uv sync --locked
uv run python scripts/verify_synthetic_data.py
uv run pytest
```

The lock fixes exact wheels and hashes. Runtime outputs, local databases, DVC cache/remote, and Python environments are ignored by Git.

For the full local Docker demonstration from a clean checkout, the supported entry point is:

```sh
make demo
```

The runner uses `.env.example` as its defaults without requiring a copy, generates the fictional data only if the `data/` directory is absent, verifies its manifest and DVC directory pointer without a cache/remote, and uses a unique Compose project/volume. It records the host Git revision and dirty state, a SHA-256 source-tree digest, OCI image labels, and the inspected image ID. Runtime application services use the current non-root host UID/GID. If port 5000 is occupied, it chooses another loopback port. Result JSON and prediction output are written under UUID-based `reports/demo_runs/` and `outputs/demo_runs/` paths.

## DVC local recovery remote

`data/` is tracked as one DVC bundle. The default remote is configured only in the ignored `.dvc/config.local` and points to `local_dvc_store/` outside `data/`:

```sh
uv run dvc init
uv run dvc add data
uv run dvc remote add --local -d localstore "$PWD/local_dvc_store"
uv run dvc push
uv run dvc pull
```

The same-disk local remote demonstrates restore mechanics; it is not an off-device backup. Keep the original dataset in place during recovery demonstrations. The verified recovery check pulled the DVC pointer into a disposable checkout and matched `data/raw/lms_snapshots.csv` to SHA-256 `f4c3ccfc84626b6bbbaa9a10c7d617a5f67ce1246c768d865d5cd34e3583995d`.

## Verified CLI/API checks

- `dvc --version` reported 3.67.1.
- `mlflow server --help` exposes `--backend-store-uri`, `--artifacts-destination`, `--host`, and `--port` for the planned local SQLite/artifact server command.
- A temporary two-step Metaflow flow passed `show` and `run` on 2.19.41 (run ID `1795190935027835`); its local datastore was isolated under `/private/tmp`.
- The existing generator was rerun; the verifier confirmed 3,600 snapshots, 3,600 outcomes, 853 withdrawals, all manifest checksums, and rejection of the invalid fixture.
- Contract and feature tests passed: `16 passed`.
- DVC recovery pulled the bundle into a disposable checkout and matched the original and temporary-backup SHA-256 for `lms_snapshots.csv`.

## Phase 4 container verification

- Built `student-dropout:local` for `linux/arm64` from the pinned Python image and locked, hash-verified `requirements.lock.txt`.
- Started the Compose MLflow server with its published port bound to `127.0.0.1:5000`, SQLite backend and artifact proxy in the persistent `mlflow_data` volume.
- Ran a complete containerized Metaflow flow through `http://mlflow:5000`; it registered gate-passing model version 1. Promoted version 1 explicitly, then a second container flow loaded it through the strict shared skops loader, passed champion comparison and registered version 2. A further card-viewer verification run registered version 3; the champion remains version 1.
- Scored the 12-row example batch in the container before and after restarting MLflow. Both runs wrote all 12 predictions with finite probabilities and exact version 1; this verified that the registry database, model and run artifacts survived restart.
- Host and container scoring of the same version-1 model and 12-row input matched row keys and model version; maximum absolute probability difference was `1.11e-16` (within `1e-6`).
- Container Metaflow state and cards are bind-mounted at the same absolute project path as the host. Host `card list` and the Local Card Viewer on port 8324 both displayed all eight cards for container run `1795193668049314`.
- MLflow's optional PyFunc input-example check logs a warning while `MLFLOW_ALLOW_MODEL_CODE_EXECUTION=false`; the locked sklearn model and input example are still logged. The shared native sklearn scorer independently validates the registered `skops` artifact's exact `numpy.dtype` allowlist and successfully calls `predict_proba`.

## Current candidate policy

- New training candidates use configuration 1.1, with minimum validation recall 0.80; the other release gates remain unchanged. The previous configuration is preserved byte-for-byte as `config/model-v1.yaml` (SHA-256 `b71fb714b1657910e64ba155aa57563f400cde3551822538466e760edbf69f9b`) for the original frozen version 1 and its recorded one-time final assessment.
- `make demo` does not promote candidates and does not evaluate the held-out final cohort. It uses an isolated Compose project and registry volume, leaving the existing project and champion unchanged.

## Isolated non-root `make demo` verification

- Ran `make demo` on the current dirty working tree. Data checks passed for 3,600 synthetic records and the full six-file DVC pointer (`58dd4f08e5a36015f5b265a093195256.dir`). No existing data or report was overwritten.
- Built ARM64 image `student-dropout:demo-be10b3d5b598`, image ID `sha256:188cc937b1b6476b0c2957d1223472c0798d731f01e97806c374e707e22f8618`. OCI revision/source labels matched the computed host provenance; the training run recorded the exact image ID.
- All three application services—MLflow, training, and scoring—reported UID 501 and GID 20. Only the isolated MLflow volume initializer ran as root.
- The dashboard stayed isolated at `http://127.0.0.1:52106` because the existing dashboard occupied port 5000. The candidate registry contained version 1, passed the 0.80 recall gate, and was not promoted. Metaflow run `1795244875752748`; MLflow run `334287e40c8f483d9996e9573477ddac`.
- The machine result is `reports/demo_runs/be10b3d5b5984668bd033740a73580a7/demo-result.json`. The concrete version 1 scorer produced 12 rows at `outputs/demo_runs/be10b3d5b5984668bd033740a73580a7/predictions.csv`.
- The existing model 1 freeze/registry remained separate. The new demo endpoint and files are local outputs, not a public deployment. Training's optional MLflow PyFunc input-example warning is expected with broad model-code execution disabled; native skops batch scoring completed successfully.

## Primary references

- [Metaflow: constructing flows](https://docs.metaflow.org/api/flowspec)
- [MLflow tracking server](https://mlflow.org/docs/latest/tracking/server/)
- [MLflow CLI reference](https://mlflow.org/docs/latest/api_reference/cli.html)
- [DVC local remotes](https://doc.dvc.org/command-reference/remote/add)
- [Docker Compose getting started](https://docs.docker.com/compose/gettingstarted/)
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
