# Student course-withdrawal MLOps demo

This project demonstrates a local model lifecycle using a deterministic fictional LMS export. It predicts withdrawal after the day-35 snapshot and by course end among enrolments active at the cutoff. It does not predict university departure or failure without withdrawal.

All bundled students and outcomes are synthetic. Metrics and probabilities demonstrate the workflow only; they do not establish real student risk, real-world accuracy, calibration, causality, or deployment suitability.

## Start here

1. [LMS export contract](docs/LMS_EXPORT_SPEC.md) for the backend handoff fields and validation rules.
2. [Synthetic data guide](docs/SYNTHETIC_DATA.md) for cohorts, target, split, and limitations.
3. [Environment guide](docs/ENVIRONMENT.md) for locked dependencies and DVC recovery.
4. [Integration guide](docs/INTEGRATION.md) for the local MLflow dashboard, Docker training/scoring, and Metaflow card viewer.
5. [Monitoring guide](docs/MONITORING.md), [demo runbook](docs/DEMO_RUNBOOK.md), and [model card](docs/MODEL_CARD.md) for monitoring and release evidence.

## Quick start

From the project directory:

```sh
uv sync --locked
uv run python scripts/verify_synthetic_data.py
docker compose build
docker compose up -d mlflow
```

The MLflow dashboard is at [http://127.0.0.1:5000](http://127.0.0.1:5000). Training and scoring are one-shot jobs; `docker compose up -d mlflow` does not train a model. See the [integration guide](docs/INTEGRATION.md) for the job commands and the Local Card Viewer at [http://localhost:8324](http://localhost:8324).

## Model workflow

The Metaflow flow validates the complete export, trains one fixed-seed logistic-regression pipeline on cohorts 2023_S1 through 2024_S2, chooses its threshold on 2025_S1, and records a release gate. The final cohort, 2025_S2, is reserved for one separately frozen final assessment. Training does not promote candidates automatically. Use an explicit promotion command only after reviewing a passing gate report.

The frozen model and completed once-only final-test assessment are recorded under `reports/demo/`; the results appear in the [model card](docs/MODEL_CARD.md). The held-out cohort is now consumed and must not be reused for tuning. All results are synthetic demonstration evidence, not real-world performance claims.

## Scope

The backend colleague remains responsible for LMS extraction, identity mapping, permissions, and storing or displaying returned predictions. This repository provides no frontend, prediction API, automatic student messages, or intervention decisions.
