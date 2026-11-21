# Model card: student course-withdrawal demonstration

## Summary

This is a synthetic-data demonstration model for estimating whether an active student–course enrolment withdraws after the day-35 snapshot and by the scheduled course end. It is not a university-dropout, academic-failure, or student-intervention model.

The model card documents one frozen registry entry: `student-dropout`, version `1`, URI `models:/student-dropout/1`. The freeze record is [reports/demo/frozen_model.json](../reports/demo/frozen_model.json). Its threshold is **0.15**. The model was frozen before the final 2025_S2 assessment; the single authorized evaluation is now complete.

The frozen version 1 used historical config 1.0 and a 0.60 minimum-recall gate ([config/model-v1.yaml](../config/model-v1.yaml), retained byte-for-byte). Future candidate runs use config 1.1 and a 0.80 minimum-recall gate in [config/model.yaml](../config/model.yaml); this policy change does not alter the frozen model, its reports, or its evaluation.

## Intended use and limits

The intended use is to demonstrate validation, reproducible training, model registration, explicit promotion/rollback, batch scoring, and drift monitoring. A backend may use the documented output contract in a fictional integration exercise.

Do not use this model to make decisions about real students. The data generator encodes assumed activity/withdrawal relationships and does not represent an institution. Probabilities, risk bands, metrics, reliability bins, and PSI alerts are not validated operating criteria. They do not establish causal explanations, fairness, calibration on real populations, or benefit from interventions. No automatic messages or punitive actions follow a score.

## Data and target

- **Source:** deterministic fictional LMS snapshots and outcomes; generator seed 903.
- **Row unit:** one student–course enrolment at exactly day 35, active at cutoff.
- **Positive target:** withdrawal strictly after the cutoff and no later than course end.
- **Negative target:** completed or failed without withdrawal, after course end and the 14-day recording delay.
- **Unknown or immature outcomes:** excluded from supervised training; never recoded as retained.
- **Temporal cohorts:** train `2023_S1`, `2023_S2`, `2024_S1`, `2024_S2` (2,400 rows); validation `2025_S1` (600 rows); held-out final test `2025_S2` (600 rows).
- **Student overlap:** zero cross-split overlap in the synthetic bundle.

The training data manifest SHA-256 is `5976aab5e3e240793d24af12826fc75286507bab954194f8e9693ea052dc767c`. The freeze record also binds the model to configuration SHA-256 `b71fb714b1657910e64ba155aa57563f400cde3551822538466e760edbf69f9b` and deployment bundle SHA-256 `4848ca1f074817412c460c5e3f14085b5f5bf21bea263623e831c3964608dbec`.

## Inputs and method

The estimator receives these 14 allowlisted features:

`active_days_14d`, `active_days_previous_14d`, `interactions_14d`, `interactions_previous_14d`, `resource_views_14d`, `forum_posts_14d`, `late_submissions_to_date`, `assignments_due_to_date`, `scores_available_count`, `mean_score_pct_to_date`, `missed_assignments`, `activity_change`, `days_since_last_activity`, and `no_activity_history`.

The model is a scikit-learn pipeline with median imputation and missingness indicators, standard scaling, and logistic regression using fixed seed 903. Learned preprocessing is fit on train only. There was no hyperparameter search or calibration step. Identifiers, dates, cohort, source-system metadata, and outcome columns are excluded from the estimator.

The threshold was selected using validation only: evaluate 0.05–0.95 in 0.05 increments, require precision of at least 0.30, maximize recall, then break ties by precision and higher threshold. The selected threshold is 0.15 and was frozen before final-test evaluation.

## Validation results

These are validation metrics for model version 1 on the 600-row 2025_S1 validation cohort, from [reports/1795193217753679/validation_metrics.json](../reports/1795193217753679/validation_metrics.json). They are synthetic demonstration evidence.

| Metric | Validation result |
|---|---:|
| Average precision | 0.6191 |
| ROC-AUC | 0.7917 |
| Precision at 0.15 | 0.3495 |
| Recall at 0.15 | 0.8273 |
| F1 at 0.15 | 0.4915 |
| Brier score | 0.1336 |
| Train-prevalence constant Brier | 0.1782 |
| Validation prevalence | 0.2317 |

## Final held-out results

The single frozen evaluation completed on 2025_S2, support 600, at the unchanged threshold 0.15. The run is `3ced7e20746441c6be1630f8417414dd`; the metrics and immutable bundle/config/data bindings are recorded in [reports/demo/final_test_metrics.json](../reports/demo/final_test_metrics.json), [reports/demo/final_test_attempt.json](../reports/demo/final_test_attempt.json), and [reports/demo/frozen_model.json](../reports/demo/frozen_model.json).

| Metric | Final synthetic test result |
|---|---:|
| Average precision | 0.4656 |
| ROC-AUC | 0.7342 |
| Precision at 0.15 | 0.2934 |
| Recall at 0.15 | 0.8306 |
| F1 at 0.15 | 0.4337 |
| Brier score | 0.1410 |
| Train-prevalence constant Brier | 0.1655 |
| Test prevalence | 0.2067 |

These are results on simulated outcomes only. Final precision (0.2934) is below the illustrative validation precision floor of 0.30; the frozen threshold and model were not changed after seeing this result. No post-test tuning or result-based model selection was performed. Do not claim real-world student performance. The test cohort has now been used and is no longer an untouched holdout; a redesign requires a new held-out cohort.

## Release and monitoring

The model is registered as an immutable MLflow version. Promotion and rollback use explicit commands and restore the saved model version and deployment bundle. `reports/releases.json` contains an earlier host-local SQLite registry's version-1/version-2 promotion and rollback, followed by a fresh Compose registry's version-1 promotion; these registry histories are separate.

The TRAIN-only reference profile for version 1 is `reports/monitoring/reference-profile-v1.json`. The clean 2,400-row TRAIN comparison had PSI 0.0 for all monitored features. The 600-row prepared drift fixture alerted on active days (PSI 11.7648), interactions (10.6508), and days since last activity (11.1593). Both reports had `data_not_ready` outcome status because no outcomes were attached. This only demonstrates detection of a prepared input shift; it is not measured performance degradation. See [docs/MONITORING.md](MONITORING.md).

## Known limitations

- All evidence is synthetic and generated from one deterministic set of assumptions.
- Each synthetic learner has one day-35 snapshot; weekly trajectories are not supported.
- The feature set excludes demographics, text, course content, and many contextual factors.
- Actual event coverage, grade availability, administrative withdrawal mapping, and population differences require validation with the institution's source systems.
- The 0.15 threshold and release gates are assignment settings, not approved student-risk policies.
- Hosted GitHub Actions results are unverified until the repository is connected and the workflow runs.
