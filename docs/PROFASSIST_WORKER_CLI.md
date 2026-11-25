# ProfAssist private worker CLI

The stable entry point is `python -m profassist_worker OPERATION --input INPUT.json --output OUTPUT.json --work-dir JOB_DIR`.
The trusted scheduler authorizes the request and creates a private job directory before invoking the CLI. The input envelope is strict JSON:

```json
{"schema_version":1,"operation":"dropout-score","context":{"tracking_uri":"http://mlflow","model_name":"dropout","version":"7"},"payload":{"files":{"snapshots.csv":{"encoding":"base64","content":"..."}}}}
```

Input files may also be placed directly in the private job directory by the host shim. When `payload.files` is used, supported filenames are `snapshots.csv`, `outcomes.csv`, `manifest.json`, `raw/lms_snapshots.csv`, and `raw/student_outcomes.csv`; each entry is `{ "encoding": "base64", "content": "..." }`. The aggregate decoded file limit is 128 MiB. No path in the JSON envelope is opened.

Success writes an immutable output envelope and emits a short completion record on stdout:

```json
{"schema_version":1,"operation":"dropout-score","status":"completed","result":{},"artifacts":{},"input_sha256":"...","result_sha256":"..."}
```

Failures return exit code 2 and do not publish the output envelope. Error messages identify exception type and message; detailed model logs remain in the private worker logs.

## Operations

| Operation | Context | Payload / private files | Result |
|---|---|---|---|
| `plagiarism-score` | `tracking_uri`, `model_name`, `version`; optional `encoder_dir`, `fixture` | Existing worker payload `{scope:{institution_id,course_id,assignment_id},reports:[{report_id,version_id,text}],source_reports?:[...]}` | Score report, model and index versions, plus `display_flags` with immutable report/version IDs, raw offsets, passage text, and review score in `[0,1]`. The scope-specific index is job-local. |
| `plagiarism-snapshot` | `scope` (also appears in payload) | `{scope,reports,reviews}` matching `platform_worker` exactly | Dataset and manifest checksums. Snapshot files are persisted under `PROFASSIST_WORKER_SCOPE_DIR/plagiarism-snapshots/<scope-sha256>/` for later training jobs. |
| `plagiarism-train` | `tracking_uri`, `model_name`, `version`, `scope`; optional `config_path`, `encoder_dir` | `{snapshot_manifest_sha256}` | Existing gated Metaflow result and job-local report artifacts. The selected release and snapshot manifest are checked by the training flow. |
| `plagiarism-verify-release` | `tracking_uri`, `model_name`, `version` | `{scope}` | Verified concrete bundle and gate evidence for the exact assignment scope. |
| `dropout-score` | `tracking_uri`, `model_name`, `version`, `scope:{tenant_id}`; optional trusted `config_path`, `fixture:true` for fixture namespace only | `snapshots.csv` or `raw/lms_snapshots.csv` | Complete batch summary, prediction records, and the `predictions.csv` artifact hash. Invalid input rejects the entire batch. |
| `dropout-train` | `tracking_uri`, `model_name`, `scope:{tenant_id}`; optional `version` (the concrete active release), trusted `config_path`, `experiment_name`, `fixture:true` for fixture namespace | `train_cohorts`, `validation_cohorts`, optional `test_cohorts`; files `raw/lms_snapshots.csv`, `raw/student_outcomes.csv`, `manifest.json` | Gated candidate result and job-local report hashes. Requires an immutable non-synthetic `platform_export` manifest, exact tenant scope, verified CSV checksums, and explicit chronological cohorts. Without an active version the flow evaluates a bootstrap candidate against its locked prevalence/rule gates; it does not compare another tenant's registry alias. |
| `dropout-verify-release` | `tracking_uri`, `model_name`, `version` | `{scope:{tenant_id}}` | Verified concrete bundle, passing gate, and matching tenant lineage. |
| `dropout-monitor` | `tracking_uri`, `model_name`, `version`, `scope:{tenant_id}`; optional trusted `config_path`, `fixture:true` for fixture namespace only | `snapshots.csv` or `raw/lms_snapshots.csv`; optional `outcomes.csv` or `raw/student_outcomes.csv` | Monitoring report and artifact hash. |

The combined worker image expects the frozen MiniLM model directory to be mounted read-only at `/app/models/minilm`; the worker runs with offline Hugging Face settings. `PROFASSIST_WORKER_SCOPE_DIR` must be a private persistent shared volume for scoped plagiarism snapshots. `PROFASSIST_WORK_DIR` is set by the CLI for gated dropout training so result and report files stay inside that job's directory.

Dropout platform manifests must declare `synthetic:false`, `provenance:"platform_export"`, `schema_version:"1.0"`, an `export_id`, UTC export metadata, cohort list and chronological train/validation/test split, exact `platform_scope`, coverage limitations, row counts, and SHA-256 checksums for exactly the two exports. Manifest paths are restricted to the two private relative export paths and validated against parsed cohort dates.

For a training rehearsal, `data-mode=platform-fixture` accepts only an explicitly `synthetic:true`, `fixture_only:true`, `provenance:"synthetic_platform_fixture"` manifest with the same export metadata and a tenant scope. It also requires an isolated registry model name containing `fixture`. Fixture candidates cannot pass `dropout-verify-release` or normal scoring/monitoring; a scoring rehearsal must set `context.fixture:true`. The demo DVC flow remains the default.
