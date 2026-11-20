# Feature drift monitoring

## Purpose and limits

Monitoring compares incoming day-35 snapshot features with a reference built from the fixed TRAIN cohorts only. It reports changes in the input distribution; a drift alert does not establish that model accuracy or student outcomes changed. The PSI alert threshold of 0.20 is an assignment demonstration setting, not a universal standard.

The monitor validates a whole batch before calculating PSI. Invalid batches are operational failures, and no partial rows are scored or measured for drift. Inputs below 100 valid rows receive `insufficient_data`. The monitored feature set is `active_days_14d`, `interactions_14d`, and `days_since_last_activity`. Missingness is reported separately from PSI. Reference bin boundaries come only from TRAIN feature values: exact integer bins for active days and unique TRAIN deciles for the other features. PSI uses epsilon `1e-6` for empty bins and renormalizes probabilities after smoothing.

The reference profile is generated from `lms_snapshots.csv` alone. The command validates the snapshot export, selects the four TRAIN cohorts before deriving features, and checks the manifest and snapshot checksums against the model bundle. It does not open the outcomes export. The profile is stored under the registered model version's originating MLflow run and bound by the `monitoring_profile_sha256` model-version tag, run ID, training manifest hash, target version, feature-schema version, and feature columns. The monitor verifies these bindings before use.

## Build the profile for a concrete model version

Start local MLflow and use the concrete version being monitored:

```sh
uv run python scripts/build_monitor_profile.py \
  --model-version 1 \
  --tracking-uri http://127.0.0.1:5000 \
  --data-dir data
```

The profile artifact path includes the version, such as `monitoring/reference-profile-v1.json`. Re-running the command is idempotent when the profile checksum matches. It refuses to replace a different profile for an already released version. It does not change the champion alias.

## Monitor a complete batch

Run the monitor against a concrete model version. A one-off input batch can use the same export format as the scorer:

```sh
PYTHONPATH=src uv run python -m dropout.monitor \
  --input data/fixtures/drifted_scoring_input.csv \
  --model-version 1 \
  --tracking-uri http://127.0.0.1:5000 \
  --config config/model.yaml \
  --report reports/monitoring/drift-report.json
```

The report records batch, model, target, schema, profile and training-manifest identifiers; row counts; per-feature missingness and summary statistics; PSI values; mean probability; review rate; outcome support; and retraining readiness. A drift alert is a valid report with exit code zero. If outcomes are missing or immature, the recommendation remains `data_not_ready`, even when drift alerts; mature labels are required before supervised retraining. Validation, registry, profile-integrity, and file errors return nonzero and do not become drift scores. The report path must differ from the input and optional outcomes file.

To join mature outcomes later, supply a separate CSV matching the scored batch:

```sh
PYTHONPATH=src uv run python -m dropout.monitor \
  --input path/to/scored-batch.csv \
  --outcomes path/to/mature-outcomes.csv \
  --model-version 1 \
  --tracking-uri http://127.0.0.1:5000 \
  --config config/model.yaml \
  --report reports/monitoring/outcome-report.json
```

Missing and `unknown` outcomes, plus completed/failed outcomes observed before course end and the 14-day recording delay, remain pending. Only matured labels contribute to precision, recall, Brier score and support counts. Historical outcomes from this fictional dataset are simulated evidence; do not attach the held-out 2025_S2 outcomes to the Phase 5 drift demonstration. A batch without mature labels is reported as `data_not_ready` and cannot start supervised retraining.

## Demonstration evidence

The clean profile-reference comparison used 2,400 unlabeled TRAIN snapshots and had PSI 0.0 for all three monitored features. The 600-row unlabeled drift fixture alerted on the engagement features; its PSI values were 11.7648 for active days, 10.6508 for interactions, and 11.1593 for days since last activity. Both reports returned `data_not_ready` because no outcomes were attached. These values only show that the pipeline detects the prepared distribution shift; the fixture has no changed outcomes and says nothing about performance degradation.

## Automated checks

GitHub Actions generates the deterministic fictional dataset directly because the DVC remote is local-only. It runs the locked test suite and an isolated SQLite-backed Metaflow training smoke run; it does not need the local DVC store, a shared MLflow service, a registry alias, or final-test metrics. The smoke run uses a temporary model name and database. The image build targets the repository's pinned `linux/arm64` base through QEMU and Buildx on the standard amd64 GitHub runner. No image is pushed. A successful GitHub-hosted workflow remains unverified until the repository is connected and Actions runs.

## Viewing runs and cards locally

The MLflow dashboard is at [http://127.0.0.1:5000](http://127.0.0.1:5000). To view the Metaflow cards from the latest containerized run, start the local card viewer from the project directory in a separate terminal:

```sh
METAFLOW_USER=student-mlops \
METAFLOW_DATASTORE_SYSROOT_LOCAL="$PWD" \
METAFLOW_CARD_LOCALROOT="$PWD/.metaflow/mf.cards" \
uv run python flows/train_flow.py card server
```

Open [http://localhost:8324](http://localhost:8324). The viewer presents saved step cards and updates as a local flow runs. It does not schedule flows or alter the MLflow registry. The flow and viewer must use the same local datastore path; stop the viewer with Ctrl+C.
