# Synthetic dataset guide

All records are fictional. The generator encodes assumed relationships between activity and withdrawal, with random noise. Results measure whether the MLOps implementation works on this simulated problem. They do not establish real student risk, real accuracy, causal explanations or suitability for deployment.

## Files

- `data/raw/lms_snapshots.csv`: 3,600 day-35 snapshots, 25 columns, six courses and six cohorts.
- `data/raw/student_outcomes.csv`: 3,600 matching outcomes, stored separately from features.
- `data/examples/scoring_input.csv`: 12 unlabelled scoring examples drawn from the final cohort. These demonstrate the interface and are not new independent observations.
- `data/fixtures/invalid_snapshot.csv`: one row with an impossible active-day count of 19. Expected result: validation failure.
- `data/fixtures/drifted_scoring_input.csv`: 600 final-cohort records with sharply reduced recent engagement. Expected result: a distribution-change alert. There are deliberately no attached changed outcomes; do not claim measured performance degradation from this fixture.
- `data/manifest.json`: seed, cohorts, temporal split, label counts and SHA-256 checksums.
- `scripts/generate_synthetic_data.py`: repeatable standard-library generator, seed 903. Run from project root with `python scripts/generate_synthetic_data.py`.

## Split and model inputs

Train: 2023_S1, 2023_S2, 2024_S1, 2024_S2 (2,400 rows). Validation: 2025_S1 (600). Final test: 2025_S2 (600). No identifier is shared across synthetic cohorts. Real-data validation must report repeat-student overlap and preferably evaluate a separate previously unseen-student subset.

Use exactly these inputs in the first model: active_days_14d, active_days_previous_14d, interactions_14d, interactions_previous_14d, resource_views_14d, forum_posts_14d, late_submissions_to_date, assignments_due_to_date, scores_available_count, mean_score_pct_to_date, plus the derived features below.

Derive missed_assignments = assignments_due_to_date - assignments_submitted_to_date; activity_change = active_days_14d - active_days_previous_14d; days_since_last_activity = elapsed days between cutoff and last_activity_at, capped at 35; no_activity_history = 1 iff last_activity_at blank. For blank last_activity_at set days_since_last_activity=35 with the indicator retained. Do not feed raw identifiers, dates, cohort, source system, outcome fields or metadata into the estimator.

Use a scikit-learn Pipeline: median imputation with missingness indicators, StandardScaler, LogisticRegression with fixed seed and no hyperparameter search. Fit all learned preprocessing on training only. Handle the blank score as missing. Default class weighting can remain unset; choose the alert threshold on validation data. Do not add calibration machinery unless evaluation exposes a reason; still report Brier score and a reliability plot.

## Limitations and demonstration use

Each learner has exactly one snapshot, all at day 35. This dataset does not support a claim of validated weekly predictions. It omits demographics, free text, LLMs, resource contents and genuine university records. Future ongoing cohorts may be unlabelled; score them but never assume retention until outcomes mature.

Use invalid and drift fixtures for failure and monitoring demonstrations only. Never append them to the training data. Generate a fresh deterministic version with an explicitly changed seed only when demonstrating DVC version recovery; restore seed 903 and the original checksum afterwards. Keep the generator version and input seed in experiment metadata.
