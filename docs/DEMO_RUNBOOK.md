# Demonstration runbook

This is a local lifecycle demonstration using deterministic fictional LMS records. It is not a live student-risk service. The final `2025_S2` cohort has already been used once to assess frozen model version 1 at threshold `0.15`; do not rerun the final evaluation or use that cohort for tuning. The completed record is `reports/demo/final_test_attempt.json`.

## Start the local services

From the project directory, generate and validate the synthetic data, build the pinned local image, and start the tracking server. A fresh clone does not contain the original machine's DVC remote, saved model versions, reports or databases. The scenario records below describe the original demonstration; create a candidate and explicitly promote a passing version using [INTEGRATION.md](INTEGRATION.md) before scoring on a fresh installation.

```sh
uv sync --locked
uv run python scripts/generate_synthetic_data.py
uv run python scripts/verify_synthetic_data.py
docker compose build
docker compose up -d mlflow
docker compose ps
```

MLflow is at [http://127.0.0.1:5000](http://127.0.0.1:5000). To inspect the saved Metaflow cards, start this in a second terminal:

```sh
METAFLOW_USER=student-mlops \
METAFLOW_DATASTORE_SYSROOT_LOCAL="$PWD" \
METAFLOW_CARD_LOCALROOT="$PWD/.metaflow/mf.cards" \
uv run python flows/train_flow.py card server
```

The Local Card Viewer is at [http://localhost:8324](http://localhost:8324). Detailed Docker scoring, shutdown, and tracking instructions are in [INTEGRATION.md](INTEGRATION.md); data recovery notes are in [ENVIRONMENT.md](ENVIRONMENT.md).

## Evidence for the eight scenarios

These are the outcomes already observed in the local demonstration. Reports and run IDs are the evidence; a command below may create another run or registry change if repeated. The final-test script is once-only and must not be rerun.

| # | Scenario and observed result | Evidence and boundary |
|---|---|---|
| 1 | **Recover and verify data.** DVC restored the bundled dataset in a disposable checkout and the snapshot checksum matched. The deterministic-data verifier confirmed 3,600 snapshots, 3,600 outcomes, checksums, and repeat-generation identity. | [ENVIRONMENT.md](ENVIRONMENT.md), [DATA_VALIDATION.json](DATA_VALIDATION.json). This uses the same-disk local DVC remote; it is not an off-device backup. |
| 2 | **Train and inspect a candidate.** The containerized Metaflow flow completed and registered a gate-passing candidate; the run produced all eight cards. | Metaflow run `1795193217753679`; MLflow run `980c428a6c444c6f9f2e84fb6a696c00`; `reports/1795193217753679/` contains the gate, validation, lineage, and bundle reports. Viewer run `1795193668049314` showed all eight cards. The flow itself does not promote a model. |
| 3 | **Reject an invalid snapshot.** The `active_days=19` fixture was rejected by the synthetic-data verifier; an invalid validation flow run is also recorded. | `data/fixtures/invalid_snapshot.csv`, [DATA_VALIDATION.json](DATA_VALIDATION.json), and `reports/1795192177589487/`. The verifier rejection is directly evidenced; do not present it as a separate end-to-end production ingestion test. |
| 4 | **Block a poor candidate.** The constant train-prevalence validation baseline failed the release gate, was not registered, and left champion version 1 unchanged. It failed AP and precision checks; recall and Brier checks passed. | [failed_baseline_gate.json](../reports/demo/failed_baseline_gate.json); MLflow run `81e3291b98704e7d9dcc809c26f7b0cd`. |
| 5 | **Score a complete batch.** Docker scored all 12 example rows with finite probabilities and concrete model version 1. The post-restart output matched the earlier probabilities. | [phase6_smoke_v1.csv](../outputs/phase6_smoke_v1.csv); see [INTEGRATION.md](INTEGRATION.md) for the exact scoring command and output contract. This is synthetic batch output, not a live integration with a university backend. |
| 6 | **Detect feature drift.** The clean 2,400-row TRAIN reference had PSI 0.0; the prepared 600-row drift fixture alerted on all three monitored features. Both had `data_not_ready` because no outcomes were attached. | [clean_reference_report.json](../reports/monitoring/clean_reference_report.json), [drift_fixture_report.json](../reports/monitoring/drift_fixture_report.json), [reference-profile-v1.json](../reports/monitoring/reference-profile-v1.json), and [phase6_monitor_v1.json](../outputs/phase6_monitor_v1.json). Drift is not evidence of degraded model performance and did not trigger retraining or promotion. |
| 7 | **Promote and roll back.** In the earlier host-local SQLite registry, a second gate-passing version was promoted, then rollback restored version 1 and its deployment bundle. The later fresh Compose registry separately promoted and froze its version 1. | [releases.json](../reports/releases.json) records both histories; the earlier promotion/rollback uses run IDs `d2835300c94d4737b00e9cc844d3d3cd` and `a942eeada80f453f861bf6c81f320b1c`, while the final Compose registry freeze uses run `980c428a6c444c6f9f2e84fb6a696c00`. The freeze record [frozen_model.json](../reports/demo/frozen_model.json) binds version 1 at threshold 0.15. Retraining used the same algorithm and synthetic data; this demonstrates version lifecycle, not model improvement. |
| 8 | **Restart services and check automation.** The MLflow service was restarted; model/run artifacts persisted, and version-1 scoring before and after restart matched to maximum absolute probability difference `1.11e-16`. The CI workflow is configured, but no hosted GitHub Actions run has been verified. | Container verification is summarized in [ENVIRONMENT.md](ENVIRONMENT.md); local workflow is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Hosted CI remains **unverified** until connected and run. |

## Frozen final assessment

The freeze record is [frozen_model.json](../reports/demo/frozen_model.json). It binds model URI `models:/student-dropout/1`, threshold `0.15`, training manifest, configuration, and deployment bundle. The once-only final evaluation is recorded in [final_test_metrics.json](../reports/demo/final_test_metrics.json), [final_test_attempt.json](../reports/demo/final_test_attempt.json), and MLflow run `3ced7e20746441c6be1630f8417414dd`.

On the 600-row synthetic `2025_S2` cohort (124 positive, prevalence 0.2067), AP was 0.4656, ROC-AUC 0.7342, precision 0.2934, recall 0.8306, F1 0.4337, and Brier score 0.1410 versus 0.1655 for the train-prevalence constant baseline. Precision is below the illustrative 0.30 validation floor. No threshold change, retuning, or selection followed this result. These numbers describe only the generated simulation; the final cohort is consumed and a future redesign needs a new held-out cohort. See the [model card](MODEL_CARD.md) for scope, validation results, limitations, and release details.

## Shutdown

Stop services while preserving the MLflow volume and saved state:

```sh
docker compose stop
```

Avoid `docker compose down -v` during the demonstration; it removes the persistent MLflow volume.
