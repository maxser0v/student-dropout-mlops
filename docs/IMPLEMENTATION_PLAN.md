# Student dropout MLOps implementation plan

## Current model-policy note (2026-11-21)

Frozen model version 1 retains its original `config/model-v1.yaml` policy and historical 0.60 minimum-recall gate. Future candidates use config version 1.1 in `config/model.yaml`, with minimum recall 0.80; other gate settings are unchanged. This future policy does not rewrite frozen reports or the completed version-1 evaluation.

## Primary goal

Demonstrate a reproducible student course-withdrawal model lifecycle: version a dataset, run a checked training pipeline, track and approve a model, execute containerised predictions, detect changed inputs, and roll back safely. The professor should be able to see evidence of each practice; model optimisation is secondary.

## Scope and current assets

The export contract, a deterministic synthetic generator, 3,600 fictional snapshot rows, separate outcomes, scoring examples, monitoring/failure fixtures and a manifest already exist. Read `docs/LMS_EXPORT_SPEC.md` and `docs/SYNTHETIC_DATA.md` first. The MLOps stack below is planned, not implemented. Do not spend implementation time rebuilding the synthetic data or developing the application frontend/backend.

Target: withdrawal after the day-35 cutoff and by course end, conditional on being active at the cutoff. It is not university-wide dropout and does not include failure without withdrawal. Use one logistic regression pipeline. No LLM, XGBoost, feature store, Kafka, Kubernetes, cloud deployment, SHAP or second orchestrator is required. Real-data access remains the colleague's responsibility. Metaflow replaces the earlier proposed Prefect role for this component.

## Tool responsibilities

| Tool | Sole responsibility in this MVP | Professor evidence |
|---|---|---|
| Git | Code, configuration, docs and DVC pointer history | Commit linked to a run |
| DVC | Dataset snapshots and restore/push/pull | Restored bytes match a recorded checksum |
| Metaflow | One local linear training workflow and step artifacts | Successful and failed flow runs |
| MLflow | Experiments, metric/artifact logging, registry and model aliases | Runs, model versions, explicit champion |
| Docker and Compose | Repeatable training/scoring runtime and persistent local MLflow | Same batch scored in a container |
| GitHub Actions | Lightweight automated checks and image build | Passing and deliberately failing checks |

DVC does not orchestrate training; no dvc.yaml/repro graph is needed. MLflow does not schedule jobs. Metaflow runs manually for the demonstration; no scheduler service is required. A scheduled backend job can invoke the same scorer later.

## Phase 0 Documentation and environment

**How this serves the primary goal:** Makes the implementation runnable on the student's machine and keeps the four tools' roles clear.

Implementation: inspect existing project instructions before editing. Create a Python 3.11 environment, choose mutually compatible current package releases, pin the tested versions in a lock/requirements file and pin the Docker base image. Check Docker availability. Use scikit-learn, pandas, MLflow, Metaflow, DVC and pytest; do not add infrastructure dependencies without a specific need. Create a short architecture diagram and `docs/ENVIRONMENT.md` containing verified versions and commands.

Documentation and allowed patterns, checked during planning:

- Metaflow: [Creating flows](https://docs.metaflow.org/metaflow/basics), sections Structure, Linear, Artifacts and Parameters. Use `FlowSpec`, `Parameter`, `@step`, `self.next`, persisted `self.<artifact>`. CLI: `python flows/train_flow.py show` and `python flows/train_flow.py run --data-dir data`.
- MLflow: [Tracking server](https://mlflow.org/docs/latest/tracking/server/). SQLite database plus served artifact storage is enough. [Registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/): use registered model versions and aliases, not deprecated Staging/Production lifecycle stages.
- MLflow sklearn: [Flavor API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html). Native sklearn model loading supports `predict_proba`; generic pyfunc `predict` must not accidentally become a class-label substitute for probability.
- DVC: [Local remote](https://doc.dvc.org/command-reference/remote/add). `dvc init`, `dvc add data`, `dvc remote add --local -d localstore <absolute-directory>`, `dvc push`, `dvc pull`, `dvc checkout`.
- Docker: [Compose getting started](https://docs.docker.com/compose/gettingstarted/) and [compose up](https://docs.docker.com/reference/cli/docker/compose/up/). Use persistent volumes, readiness checks and explicit service profiles.
- Scikit-learn: [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) and [Calibration](https://scikit-learn.org/stable/modules/calibration.html). Pipeline owns fitted preprocessing; learned transformations never see validation/test during fitting.

Verification: record successfully executed version/help checks, validate the selected MLflow CLI flags against that installed version, and confirm the basic Metaflow skeleton runs locally. Exact package pins are an implementation output, not assumptions in this plan.

Guards: no invented APIs, unpinned `latest` images, disabled MLflow security middleware or credentials committed to Git. Only local fictional data is required.

## Phase 1 Data contract and versioning

**How this serves the primary goal:** Makes bad inputs fail clearly and makes each model's training data recoverable.

Implement `src/dropout/validation.py`, `src/dropout/features.py` and `tests/test_data_contract.py`. Use explicit named input/feature allowlists and validate the existing export contract. Check key uniqueness, exact one-to-one label joins, timestamps, cutoff eligibility, coverage watermarks, event-count ranges, score availability and assignment consistency. Reject a withdrawn outcome outside the prediction window. Unknown outcomes are excluded with counts, never relabelled as retained. Never silently drop bad feature rows from a scoring batch.

Derive features exactly as listed in `docs/SYNTHETIC_DATA.md`; keep transformations shared by training and scoring. For real repeat students, report cross-split overlap and an unseen-student evaluation where feasible. DVC-track `data/` as one small dataset bundle. Keep DVC remote outside `data/`, e.g. project `local_dvc_store/`, explicitly Git-ignored. Ignore DVC cache, generated processed data, environment files, local service databases, reports and secrets. Commit DVC pointers/config/code only when Git commits are requested or part of the authorised implementation workflow.

Remote demonstration: `dvc push`, copy one file to a temporary backup, remove that local copy only in a disposable recovery directory/checkout, then `dvc pull` and confirm SHA-256 equality. A same-disk local remote demonstrates recovery mechanics, not off-device backup. Do not delete the user's only dataset copy as a demonstration.

Documentation: export specification, synthetic guide and DVC local-remote docs from Phase 0.

Verification: clean data passes; active_days=19 fails; duplicate key, future last_activity, unavailable coverage, missing score/count inconsistency and target leakage fail; derived values match hand-worked examples. Train/validation/test are 2400/600/600. Document exclusions on real data. Re-running the generator produces identical dataset and manifest hashes.

Guards: no outcomes/IDs fed through `select_dtypes` into the model; no random row split; no fitting imputation on all cohorts; no training on scoring or drift fixtures.

## Phase 2 Training flow and experiment tracking

**How this serves the primary goal:** Produces an inspectable sequence of steps and a complete record explaining how one model was created.

Implement `flows/train_flow.py` as one local linear Metaflow flow:

`start -> validate -> prepare -> train -> evaluate -> gate -> register -> end`

Use Parameters for data directory, config path and deployment-independent run settings. Store handoff artifacts on `self`, not transient globals. Reuse pure functions from `src/dropout`; do not duplicate feature logic in steps. Local Metaflow state lives in `.metaflow/`, ignored by Git and preserved in a volume if run in a container.

Implement one scikit-learn Pipeline: median imputation with indicators, StandardScaler, LogisticRegression. Use seed 903 and no hyperparameter search. Use exact cohort assignments in the manifest. Compute the constant-prevalence baseline from the TRAIN label proportion. Choose a threshold using validation only: among thresholds 0.05 through 0.95 by 0.05 with precision >= 0.30, select highest recall, breaking ties by higher precision then higher threshold. If none qualify, record a failed gate. These are assignment demonstration settings, not validated university operating criteria.

Log a single MLflow experiment run for each full training flow: Git commit plus dirty flag, DVC pointer/data SHA-256, seed, config, dependency versions, feature-schema version, split IDs/counts, Metaflow run ID, exclusions, threshold, parameters, validation metrics, dataset manifest, confusion matrix, reliability plot and model artifact. Persist one MLflow run ID between steps and explicitly reopen that run for each logging context; verify no accidental extra run per step. Tag the run as synthetic.

Metrics: validation average precision (PR-AUC summary), ROC-AUC, precision, recall, F1 and Brier score. Use probabilities for Brier/ROC/AP; thresholded decisions for precision/recall/F1. Compare Brier with the train-prevalence constant predictor. Use MLflow native sklearn logging with input example and actual estimator signature; document the scorer's probability output separately. Keep test labels inaccessible to training and gate modules. Final test evaluation belongs to Phase 6 only.

Documentation: Metaflow linear-flow/artifact/Parameter examples, MLflow tracking server and sklearn logging API, scikit-learn Pipeline/calibration docs.

Verification: `show` displays the flow; a valid run completes and all expected logs/artifacts appear in one MLflow run; invalid fixture stops at validate before train/register; repeat run gives matching probabilities within tolerance and metrics. No live registry change occurs just because training finished.

Guards: no LLM metrics or grading-override thresholds; no claiming synthetic calibration validates real students; no test-set threshold selection; no Metaflow cloud deployment configuration.

## Phase 3 Release gate and model registry

**How this serves the primary goal:** Demonstrates that a model cannot become the deployed version merely because it trained successfully.

Implement `src/dropout/gates.py`, `scripts/promote_model.py`, `scripts/rollback_model.py` and `config/model.yaml`. Register candidates that pass the gate in MLflow under `student-dropout`. Leave the champion alias unchanged until explicit promotion. Resolve and record immutable model versions, plus threshold/config and feature-schema artifacts, as a single deployment bundle.

Initial configurable demo gates: all input/lineage checks pass; validation average precision >= validation prevalence + 0.05; recall >= 0.60; precision >= 0.30; Brier <= the constant predictor's Brier. Report raw metrics and split size alongside pass/fail. If baseline does not pass, diagnose and record failure; do not lower thresholds silently. It is acceptable for the demo to include an unreleased candidate.

When a champion exists, score champion and candidate on the SAME frozen validation rows using each version's saved threshold. Require initial gates plus no recall or precision drop > 0.05 absolute, and no Brier increase > 0.02. Compare only matching target and feature-schema versions. Report an incompatible champion as requiring a separate migration, rather than forcing a comparison. Save `gate_report.json` even for failure. A failed candidate remains a tracked experiment and cannot be promoted. Repeated candidate decisions reuse validation, so final test is held for the single final assessment; later real releases need a refreshed out-of-time evaluation window.

Promote via MLflow `MlflowClient.set_registered_model_alias(..., 'champion', version)` and record the old champion in a release log. Make promotion explicit and reject versions without a passed gate. Use aliases, not deprecated stage APIs. Rollback restores the previous concrete version AND threshold/schema bundle, then restarts or reruns scoring. Make repeat promotion/rollback calls idempotent and prevent concurrent promotions for this local demo.

Documentation: MLflow registry registration/aliases/version-loading sections from Phase 0.

Verification: a deterministic deliberately poor candidate fixture is blocked with champion unchanged; a passing candidate can be promoted; previous version is recoverable; release log includes both model and config versions. Failed-gate coverage must be an actual guard, not a fake screenshot or manual edit of pass/fail output.

Guards: never auto-promote every registered version; alias changes alone do not reload a running scorer; no final-test-based candidate selection.

## Phase 4 Docker and application handoff

**How this serves the primary goal:** Lets the colleague run the model without recreating a training environment or building an ML backend.

Implement one project `Dockerfile`, `.dockerignore`, `compose.yaml`, `src/dropout/score.py` and `docs/INTEGRATION.md`. Use the same pinned image for training and batch scoring. Define a persistent MLflow service and optional `jobs` profile for one-shot train/score containers, so `docker compose up -d mlflow` does not inadvertently retrain models. No FastAPI service is necessary for the MVP.

MLflow server example to verify against installed version:

`mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts`

Persist `/mlflow` for DB and artifacts, and bind published port to `127.0.0.1:5000`. Host clients use `http://127.0.0.1:5000`; Compose clients use `http://mlflow:5000`. Configure allowed hostnames for the pinned release if needed. Artifact serving must work across host/container paths. Add a service healthcheck or bounded client retries. Persist Metaflow state for containerised training. Do not include real exports or model artifacts in the Docker image; mount input read-only and output writable.

Batch CLI contract: input path, output path, concrete model version. A deployment helper may resolve champion once before execution, but the actual job logs and uses that fixed version throughout. Load the sklearn Pipeline natively and call `predict_proba`, selecting class 1 through `classes_`. Output the contract fields from the export specification; initial risk_band is `review` when probability >= saved threshold, otherwise `routine`. Model unavailable, schema mismatch, invalid batch or unwritable output must exit nonzero with a clear report and no apparently successful partial output. Write completed CSV atomically.

Colleague's work is limited to providing a complete input file, invoking the documented container/job, ingesting its output, and displaying probability/version/date/status under existing permissions. Retraining is a separate command and never triggered by opening a dashboard.

Documentation: Docker Compose getting-started, MLflow server and native sklearn model-loading references.

Verification: build image; start MLflow; train with persistent artifacts; score 12 example rows; probabilities finite in [0,1]; exact version recorded; local/container predictions agree within tolerance; restart services and confirm run/model/artifacts survive. Unavailable model causes a controlled failure. Do not use `docker compose down -v` in the normal guide.

Guards: no public MLflow exposure, no source-code mounts required to run a built scorer, no per-request retraining, no hot alias lookup changing model mid-batch.

## Phase 5 Monitoring retraining and automated checks

**How this serves the primary goal:** Shows a visible feedback loop and ensures changes are checked before release.

Implement `src/dropout/monitor.py`, `scripts/build_monitor_profile.py`, tests and `.github/workflows/ci.yml`. Keep monitoring lightweight: Python CSV/JSON reports are sufficient. Do not add a second dashboard platform. Build `monitoring/reference-profile-v<model-version>.json` from the canonical snapshot CSV alone, selecting TRAIN cohorts before feature derivation. Bind it to the concrete model run with its SHA-256 version tag, training manifest hash, target, feature schema and feature list; the monitor checks these against the immutable deployment bundle. Do not open or join outcome files when building a feature reference. Compare operational data with that reference using fixed train-derived bins and PSI for active_days_14d, interactions_14d and days_since_last_activity. Use epsilon 1e-6 with renormalisation for zero bins, track missingness separately, require at least 100 complete valid batch rows, and label smaller batches insufficient_data. Proposed demonstration alert: PSI > 0.20 for any selected feature. This is a configurable heuristic, not a universal statistical threshold.

Monitoring report includes batch/schema/model IDs, row counts, completeness, rejected rows, feature statistics, PSI, mean probability and review rate. Drift fixture must produce an alert on the engagement change. Validation errors remain operational failures, not drift scores. Keep outcomes out of input-drift calculations.

An alert creates a retraining recommendation with reason and identifiers. It must not relabel unlabelled students, blindly train on drift fixtures or automatically replace champion. Operator initiates the SAME Metaflow training flow only when new mature labelled data passes validation; candidate then passes the SAME release gate and explicit promotion. Record data_not_ready when labels are unavailable. Demonstrate this state as a valid outcome.

When genuine outcomes mature, join by snapshot key and report recall, precision, Brier and support counts. Mark missing outcomes pending, not negative. In the demo, historical synthetic outcomes can demonstrate this join, clearly labelled simulated.

CI: install locked dependencies, generate the deterministic synthetic data (the configured DVC remote is local-only), run contract/feature/gate/scorer/monitor unit tests and an offline training smoke test using a temporary SQLite MLflow backend and temporary model name, then build the pinned ARM64 image through QEMU/Buildx on amd64 runners. CI should not require a shared MLflow server or modify the real champion. Test registry interactions with a temporary local backend when necessary. Keep full training/releases manual for this assignment. If no GitHub repository/remote is configured, report hosted CI as unverified; do not claim a passing Actions run.

Documentation: scikit-learn metrics/pipeline docs, MLflow tracking/registry docs, [GitHub Actions Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python). PSI is our small documented implementation; explicitly unit-test equal distributions, shifted distributions and empty bins rather than implying a library implements it.

Verification: clean reference comparison has negligible PSI; drift fixture alerts; small sample yields insufficient_data; unlabelled batch cannot start supervised retraining; deliberately broken contract/gate code fails CI; no monitoring action changes champion.

Guards: no claim that drift proves accuracy loss, no automatic deployment on drift, no Kafka/Evidently service requirement, no scheduling infrastructure.

## Phase 6 Final verification and professor demonstration

**How this serves the primary goal:** Provides observable evidence that the lifecycle works end to end and that its failure controls are real.

Implement `docs/DEMO_RUNBOOK.md`, `docs/MODEL_CARD.md` and final evidence outputs. On the frozen chosen model evaluate final 2025_S2 test cohort once, log metrics as final_test, and report them without changing model/threshold to chase a result. Any subsequent redesign invalidates that cohort's untouched-test status and requires a new held-out evaluation.

Run these scenarios from a clean environment:

1. Recover a versioned dataset and verify its hash. Record a repeatable generator run as additional provenance.
2. Train the flow, show each Metaflow step, one MLflow run and the gate report.
3. Feed active_days=19; show validation stops training with no model release.
4. Attempt a known poor candidate; show failed release and unchanged champion.
5. Score the unlabelled example via Docker; show probability, timestamp and model version reaching the agreed backend boundary.
6. Feed drift fixture; show alert, data_not_ready and no unapproved replacement.
7. Promote a second legitimately passing version, then restore the previous bundle and demonstrate matching original probabilities. If a second version is only a re-training of the same algorithm/data, say so; the goal is version lifecycle, not fake improvement.
8. Restart services; confirm model artifacts and experiment history persist. Run CI-equivalent checks locally or attach actual CI output if a repository is configured.

Model card records intended target, fixed day-35 scope, synthetic source, explicit feature list, split dates, thresholds, metrics, limitations and release/rollback process. Store evidence in `reports/demo/` with commands, timestamps, exit codes, run IDs, checksums and model versions. Screenshots may supplement real output, never replace it. Do not publish or push a repository unless separately requested.

Documentation: all preceding phase references. Recheck commands against pinned installed versions and ensure README/runbook match executed commands.

Verification: all eight scenarios have actual evidence or are explicitly marked blocked/unverified; no success claim for unrun services; generator and fixtures remain deterministic; final test was not reused for tuning. The colleague can follow the handoff without installing training libraries.

Guards: avoid additional model experiments after completion; no synthetic-results-as-real-world-performance claims; no fabricated CI screenshots.

## Expected repository layout after implementation

```text
README.md
requirements.lock.txt
Dockerfile
.dockerignore
compose.yaml
config/model.yaml
flows/train_flow.py
src/dropout/{validation,features,training,gates,score,monitor}.py
scripts/{generate_synthetic_data,promote_model,rollback_model}.py
data/                         # existing assets, later DVC-tracked
 data/raw/{lms_snapshots,student_outcomes}.csv
 data/examples/scoring_input.csv
 data/fixtures/{invalid_snapshot,drifted_scoring_input}.csv
 data/manifest.json
data.dvc                      # created during implementation
.dvc/
.github/workflows/ci.yml
tests/
docs/{LMS_EXPORT_SPEC,SYNTHETIC_DATA,IMPLEMENTATION_PLAN}.md
reports/                      # generated, ignored
```

## Agent execution contract

Execute phases sequentially; only delegate parallel file ownership where dependencies are already agreed. Each phase agent reads this plan and both data docs. It must report files changed, commands run, evidence paths, exact versions, remaining gaps and the next phase's interface. Do not overwrite another agent's files or generate a second conflicting environment.

Suggested ownership: Agent A Phase 0–1 (environment/data); Agent B Phase 2–3 (training/registry); Agent C Phase 4 (Docker/integration); Agent D Phase 5–6 (monitoring/verification). Reuse agents or fresh contexts; four agents do not need to run simultaneously. Each agent must run appropriate tests and leave the project runnable before handoff. The main implementer resolves shared dependency/Compose changes.

Copy-ready assignment:

> Read docs/IMPLEMENTATION_PLAN.md, docs/LMS_EXPORT_SPEC.md and docs/SYNTHETIC_DATA.md. Implement Phase [N] only, using the verified output of the previous phase. Keep logistic regression and a local batch architecture. Follow the phase's documentation, verification checklist and guards. Reuse the existing synthetic dataset. Do not build the frontend/backend, add cloud infrastructure, publish data, or silently change the target/schema. Report changes, executed checks, artifacts, unresolved limitations and the next handoff.

## Estimated effort

Approximately 4–6 focused working days including documentation and the demo: data/versioning 0.5–1 day, flow/tracking/registry 1–1.5 days, Docker/handoff 1 day, monitoring/CI 1 day, and verification 0.5–1 day. This is a planning estimate, not a completion guarantee. Stop adding scope when the eight demonstration scenarios are evidenced.
