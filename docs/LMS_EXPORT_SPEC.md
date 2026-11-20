# LMS export specification

Please export two UTF-8 CSV files for a student course-withdrawal prediction prototype. The ML component will calculate an estimated probability of withdrawal after day 35 and before course end. This is course withdrawal, not departure from the university. Failure without withdrawal is a negative example for this target.

## Export scope and timing

One snapshot row represents one active student–course enrolment at the end of day 35, exactly `course_start_at + 35 days`. All timestamps are UTC ISO 8601 with `Z`. Include only students enrolled before the cutoff and not withdrawn or completed by that cutoff. Start with courses lasting longer than 35 days and without interrupted enrolments. Use several completed cohorts if available. Do not manufacture historical snapshots from current cumulative totals: reconstruct what was available at each historical cutoff.

The join key in BOTH files is `(tenant_id, student_key, course_id, cohort_id, snapshot_at)`. Keys must be unique. Student pseudonyms must remain stable across courses/cohorts; keep the identity mapping inside the application. The synthetic generator uses distinct students across cohorts to simplify the demonstration.

Counts use events with `event_at < snapshot_at`. Recent activity is `[snapshot_at - 14 days, snapshot_at)` and previous activity is `[snapshot_at - 28 days, snapshot_at - 14 days)`. An active day is a UTC calendar day with at least one included learner interaction. Exclude staff, bots, system events, duplicated log events and authentication-only events. Agree a versioned Moodle event-name mapping before extraction; a click count is not a session duration or proof of learning.

## File 1 lms_snapshots.csv

All columns below are required in the file. Blank values are allowed only where explicitly stated. Metadata supports validation and joining; only fields marked Feature or Derived become model inputs.

| Field | Type | Meaning and validation | Use |
|---|---|---|---|
| tenant_id | string | Institution/tenant key; never mix tenant results | Metadata |
| student_key | string | Stable pseudonymous student key; no name, email or original student number | Metadata |
| course_id | string | Course/module key, interpreted with cohort_id | Metadata |
| cohort_id | string | Sortable/mapped teaching period, e.g. 2025_S1 | Split metadata |
| snapshot_at | timestamp | Historical day-35 cutoff | Metadata |
| course_start_at | timestamp | Start of this course offering | Metadata |
| course_end_at | timestamp | Scheduled end, after cutoff | Metadata |
| enrolled_at | timestamp | Enrolment starts before cutoff; late enrolment allowed | Metadata |
| enrolment_status_at_snapshot | string | Must be active at cutoff, not today's status | Eligibility |
| activity_coverage_start_at | timestamp | Start of continuous complete log coverage; must cover the 28-day lookback | Quality |
| activity_complete_through_at | timestamp | Exclusive event watermark through which all logs are included; >= snapshot_at | Quality |
| active_days_14d | integer | Distinct active UTC dates in recent interval; 0–14 | Feature |
| active_days_previous_14d | integer | Same count in previous interval; 0–14 | Feature |
| interactions_14d | integer | Included learner events in recent interval; >= 0 | Feature |
| interactions_previous_14d | integer | Same count in previous interval; >= 0 | Feature |
| resource_views_14d | integer | Resource-view subset of recent interactions; 0 to interactions_14d | Feature |
| forum_posts_14d | integer | Forum-post subset of recent interactions; 0 to interactions_14d | Feature |
| last_activity_at | timestamp or blank | Latest included interaction before cutoff; blank only if no interaction since enrolment | Derived |
| assignments_due_to_date | integer | Required assignments with effective due date before cutoff, accounting for extensions | Feature |
| assignments_submitted_to_date | integer | Number of those due assignments with a valid submission before cutoff; 0 to due | Derived |
| late_submissions_to_date | integer | Submitted due assignments received after effective deadline; 0 to submitted | Feature |
| mean_score_pct_to_date | float or blank | Mean of available normalised 0–100 marks for those submitted due assignments; only grades released before cutoff | Feature |
| scores_available_count | integer | Count of those assignments contributing to mean; 0 to submitted; mean blank iff count=0 | Feature |
| source_system | string | e.g. moodle or synthetic_moodle | Provenance |
| schema_version | string | 1.0 for this specification | Provenance |

Resource-view and forum-post categories must not overlap. Their sum must not exceed interactions. Zero activity is valid only when log coverage is complete. A missing required activity count is an export error, not zero. Blank grades mean unavailable, not a mark of zero. If grades cannot be exported, report the limitation and agree a new schema/model version rather than invent values.

## File 2 student_outcomes.csv

Historical training/evaluation only. Never pass this file into the prediction service or use its columns as features. The snapshot file must be usable alone for scoring.

| Field | Type | Meaning |
|---|---|---|
| tenant_id, student_key, course_id, cohort_id, snapshot_at | same as snapshot | Exact join key |
| final_outcome | enum | withdrawn, completed, failed, or unknown |
| withdrawn_at | timestamp or blank | Required for withdrawn; strictly after cutoff and no later than course end; blank otherwise |
| outcome_observed_at | timestamp | When the final outcome was confirmed; completed/failed labels require course end and normal outcome-recording delay to have passed |

The training pipeline derives `dropout_label = 1` for confirmed withdrawal in the prediction window; `0` for completed or failed without withdrawal. Exclude unknown, unfinished, contradictory and unmatched outcomes from supervised training, and report the excluded counts. Never convert unknown outcomes to zero. Institution-specific administrative withdrawal codes must be mapped and documented before export; ambiguous deferrals/transfers require an agreed policy.

## Export manifest

Alongside CSVs provide a small JSON/text manifest: export time, schema version, timezone, included cohorts, row counts, extraction query/code version, included Moodle event categories, coverage limitations, and file checksums. For real exports, cohort completion and watermark assertions must be checked against source systems, not merely entered as constant values.

## Integration boundary

Backend colleague: retrieves Moodle/enrolment/outcome records, creates the two exports, maintains identity mapping and permission checks, and stores/displays returned predictions. ML owner: validates exports, derives features, trains, versions and runs a Docker batch scorer. Backend should only expose results within the matching tenant/course permissions.

Prediction output: join key, predicted_at, dropout_probability, risk_band, prediction_status, model_version, feature_schema_version, training_data_hash and batch_id. Rejected input returns a useful validation report with nonzero job exit; do not quietly score partial bad batches. No automatic messages or punitive actions follow a probability.

## Optional future raw export

The aggregated snapshot is sufficient for this assignment. If aggregation is difficult, agree a separate raw format first: enrolments with effective status history; activity events with event_id, event_at, ingestion time and event_type; assignments with effective due dates/extensions; submissions with submitted_at; and grades with released_at. These are not additional mandatory MVP exports.
