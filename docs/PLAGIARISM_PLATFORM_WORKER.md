# Platform worker boundary

`plagiarism.platform_worker` is a private batch-worker entry point for a
trusted backend queue. The platform checkout and its HTTP/auth/job services are
not part of this repository. The worker does not decide who may access an
assignment, resolve IDs, manage job state, activate a model, or roll back a
release. The backend must resolve and authorize inputs before creating a
job-owned payload and paths.

The backend supplies paths and registry configuration as trusted command-line
arguments. JSON payloads contain only scoped IDs, canonical text, and review
data. Payload fields cannot select files, tracking servers, registry namespaces,
shell commands, or output locations. Use a separate private output/index/snapshot
location per job. Keep input, index, snapshot, and result artifacts behind the
platform's assignment-scoped access controls; score results intentionally
contain excerpts and offsets.

## Score reports

Run from the repository/runtime with the source package installed or on
`PYTHONPATH`:

```sh
python -m plagiarism.platform_worker score \
  --input /private/job-123/input.json \
  --output /private/job-123/result.json \
  --index /private/job-123/index.npz \
  --index-manifest /private/job-123/index-manifest.json \
  --tracking-uri http://mlflow.internal:5000 \
  --model-name passage-reuse --version 12 \
  --encoder-dir /models/minilm
```

The backend supplies a concrete registry version. The worker calls
`load_verified_model`, builds one immutable assignment index with
`build_assignment_index`, and scores through `score_assignment_reports`. A
legacy seed model without `lineage.platform_scope` can be used only when trusted
backend policy explicitly chooses it. When a model has a platform scope, it
must match the input assignment. A scoped candidate must not be treated as a
legacy seed.

Input JSON has this shape; `source_reports` is optional and may contain approved
same-assignment reference versions. Query reports and source reports both
contribute to the shared index, while each query's own report versions are
excluded from its source candidates.

```json
{
  "scope": {
    "institution_id": "institution-opaque-id",
    "course_id": "course-opaque-id",
    "assignment_id": "assignment-opaque-id"
  },
  "reports": [
    {"report_id": "report-opaque-id", "version_id": "version-opaque-id", "text": "Canonical extracted text"}
  ],
  "source_reports": [
    {"report_id": "reference-opaque-id", "version_id": "version-opaque-id", "text": "Approved reference text"}
  ]
}
```

The assignment scope is mandatory on every report through the top-level scope;
all IDs are treated as opaque strings. Unknown fields are rejected. Result JSON
contains the scope, concrete model and bundle checksum, index version and
manifest checksum, aggregate metrics, per-report evidence, assignment-level
mirrored-passage `display_flags`, and a result checksum. A display flag groups
the unordered report/version pair and exact offsets, with evidence IDs from
either query direction; per-report evidence remains directional. The output is atomically published. Re-running identical
inputs to the same paths is idempotent; a conflicting result or changed input
at an already-published immutable index path fails closed. Stdout contains only
a completion record, and failures avoid printing exception text that may
contain report excerpts or paths.

## Freeze reviewed examples

```sh
python -m plagiarism.platform_worker snapshot \
  --input /private/job-456/reviews.json \
  --snapshot-dir /private/job-456/snapshot \
  --output /private/job-456/result.json
```

The payload contains the canonical report versions referenced by reviews. Optional trusted CLI parameters `--validation-fraction`, `--test-fraction`, and `--seed` control the corpus split (defaults are `0.2`, `0.0`, and `903`).
Review `left` and `right` fields refer to those entries by report and version
IDs. Each review has `review_id`, positive integer `revision`, `coverage`
(`full`, `partial`, or `uncertain`), nullable/boolean `reused`, and `spans`.
Each positive span has integer `left_start`, `left_end`, `right_start`, and
`right_end` offsets into the corresponding canonical strings. Only full,
unambiguous pair judgments enter the deterministic grouped corpus. The worker
uses `build_platform_corpus` and `write_platform_snapshot`, then writes a small
result with assignment scope, dataset checksum, exact manifest checksum, and
input checksum. This result is safe for the job scheduler; the snapshot itself
contains training text and must remain private.

```json
{
  "scope": {"institution_id": "i", "course_id": "c", "assignment_id": "a"},
  "reports": [
    {"report_id": "r1", "version_id": "v1", "text": "..."},
    {"report_id": "r2", "version_id": "v1", "text": "..."}
  ],
  "reviews": [
    {"review_id": "review-1", "revision": 3,
     "left": {"report_id": "r1", "version_id": "v1"},
     "right": {"report_id": "r2", "version_id": "v1"},
     "coverage": "full", "reused": true,
     "spans": [{"left_start": 0, "left_end": 12, "right_start": 4, "right_end": 16}]}
  ]
}
```

## Verify a candidate before platform activation

`verify-release` is read-only. It resolves one concrete candidate through
`load_verified_model`, requires exact matching `lineage.platform_scope`, and
requires the passing gate report to carry the same exact assignment scope and
the same snapshot checksum as `lineage.training_corpus_sha256`. It returns the
verified bundle hash, gate-report artifact hash, and training snapshot hash for
the backend's approval screen or release check. It does not activate, approve,
rollback, or write an audit ledger.

```sh
python -m plagiarism.platform_worker verify-release \
  --input /private/job-789/scope.json --output /private/job-789/release-check.json \
  --tracking-uri http://mlflow.internal:5000 \
  --model-name passage-reuse-platform-abc123 --version 13 \
  --encoder-dir /models/minilm
```

The input contains only `{"scope": {"institution_id": "...", "course_id": "...", "assignment_id": "..."}}`.
The backend must bind the returned evidence to the requested candidate and
scope, show it to an authorized approver, and own activation serialization,
approval identity/reason, audit history, and rollback. Never send an alias such
as `champion`; the CLI requires a decimal concrete version.

## Training handoff

The backend can freeze the snapshot first, verify its returned dataset and
manifest hashes, and then invoke the existing Metaflow entry point with
job-owned paths using an argument vector (not a shell string):

```sh
python flows/train_plagiarism_flow.py run \
  --data-dir /private/job-456/snapshot \
  --config-path /trusted/config/plagiarism.yaml \
  --tracking-uri http://mlflow.internal:5000 \
  --cache-dir /private/job-456/cache \
  --result-path /private/job-456/training-result.json \
  --platform-snapshot-sha256 <verified-manifest-sha256> \
  --encoder-dir /models/minilm
```

`tracking-uri`, config, cache/result paths, encoder path, optional active model
version, and snapshot digest must come from trusted backend job records. The
platform payload must never provide any command or path argument. Reconcile the
training result JSON and candidate against the same frozen snapshot and scope.
Training result handoff does not grant release eligibility by itself; run
`verify-release` after candidate completion. For workflow flags and gate
semantics see the [training flow](../flows/train_plagiarism_flow.py) and
[plagiarism runbook](PLAGIARISM_RUNBOOK.md).

## Failure and retry behavior

The process returns status 0 and a small `{status, operation}` JSON line on
success. It returns status 2 with a generic exception type on failure. Detailed
diagnostics, if captured by the worker runtime, belong in restricted backend
logs. Do not expose those logs as professor-facing failure text without
redaction. The scheduler should preserve the same input version IDs, payload,
model version, and job-owned paths for an idempotent retry; use fresh paths for
changed inputs or versions.
