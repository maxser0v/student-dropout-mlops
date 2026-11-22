# Student course-withdrawal MLOps demo

This project demonstrates a local model lifecycle using a deterministic fictional LMS export. It predicts withdrawal after the day-35 snapshot and by course end among enrolments active at the cutoff. It does not predict university departure or failure without withdrawal.

All bundled students and outcomes are synthetic. Metrics and probabilities demonstrate the workflow only; they do not establish real student risk, real-world accuracy, calibration, causality, or deployment suitability.

## Start here

1. [LMS export contract](docs/LMS_EXPORT_SPEC.md) for the backend handoff fields and validation rules.
2. [Synthetic data guide](docs/SYNTHETIC_DATA.md) for cohorts, target, split, and limitations.
3. [Environment guide](docs/ENVIRONMENT.md) for locked dependencies and DVC recovery.
4. [Integration guide](docs/INTEGRATION.md) for the local MLflow dashboard, Docker training/scoring, and Metaflow card viewer.
5. [Monitoring guide](docs/MONITORING.md), [demo runbook](docs/DEMO_RUNBOOK.md), and [model card](docs/MODEL_CARD.md) for monitoring and release evidence.
6. [Plagiarism implementation plan](docs/PLAGIARISM_IMPLEMENTATION_PLAN.md) and [plagiarism runbook](docs/PLAGIARISM_RUNBOOK.md) for the separate passage-reuse workflow.
7. [PAN 2013 passage reuse results](docs/PLAGIARISM_RESULTS.md) for the real-data Docker run, registered model, and held-out evaluation.

## Quick start

From a clean checkout with Docker/Compose and `uv` installed, run one command:

```sh
make demo
```

The runner prepares the locked Python environment, generates the deterministic fictional dataset only when `data/` is absent, verifies both the data manifest and its DVC directory pointer, builds an ARM64 image, starts an isolated MLflow service, runs the eight-step training flow, and scores a 12-row example batch with the new concrete candidate version. Existing `data/` is never regenerated or replaced. A present non-synthetic or stale dataset stops the run before training.

The demo does not promote the candidate or evaluate the held-out final cohort. It leaves the new MLflow dashboard available on a loopback port and saves its machine-readable result under `reports/demo_runs/<demo-id>/demo-result.json`; predictions go to the matching `outputs/demo_runs/<demo-id>/` directory. If port 5000 is occupied, the runner selects another free loopback port and prints its URL. It uses a unique Compose project and volume, so an existing registry and champion remain separate.

Settings are read from `.env.example` automatically. You can add a plain `.env` file to override `DEMO_MLFLOW_PORT`, `DOCKER_PLATFORM`, `RUNTIME_UID`, `RUNTIME_GID`, or `METAFLOW_USER`; the runner does not execute shell expressions. Application services run as your non-root host UID/GID by default. See [the integration guide](docs/INTEGRATION.md) for cleanup, manual service commands, provenance labels, and the Local Card Viewer at [http://localhost:8324](http://localhost:8324). For a plain-English walkthrough, read [SIMPLE_EXPLANATION.md](docs/SIMPLE_EXPLANATION.md).

## Model workflow

The Metaflow flow validates the complete export, trains one fixed-seed logistic-regression pipeline on cohorts 2023_S1 through 2024_S2, chooses its threshold on 2025_S1, and records a release gate. Current candidate configuration 1.1 requires at least 0.80 recall on the validation cohort; the other gates are unchanged. The original configuration 1.0 is preserved byte-for-byte in `config/model-v1.yaml` for the historical frozen model. The final cohort, 2025_S2, has already been assessed once and is excluded from new candidate runs and `make demo`. Training does not promote candidates automatically. Use an explicit promotion command only after reviewing a passing gate report.

The original local demonstration's frozen model and completed once-only final-test assessment were recorded under `reports/demo/`; its results appear in the [model card](docs/MODEL_CARD.md). Local reports, model artifacts, databases and prediction outputs are excluded from Git. A fresh clone starts with a new local tracking store; historical run IDs in the documentation describe the original demonstration. The original held-out cohort is now consumed and must not be reused for tuning. All results are synthetic demonstration evidence, not real-world performance claims.

## Scope

The backend colleague remains responsible for LMS extraction, identity mapping, permissions, and storing or displaying returned predictions. This repository provides no frontend, prediction API, automatic student messages, or intervention decisions.
