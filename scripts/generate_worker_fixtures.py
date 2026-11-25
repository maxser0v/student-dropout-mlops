"""Write tiny deterministic synthetic Moodle and plagiarism CLI inputs."""
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "fixtures" / "profassist"
FIELDS = ["tenant_id", "student_key", "course_id", "cohort_id", "snapshot_at", "course_start_at",
          "course_end_at", "enrolled_at", "enrolment_status_at_snapshot", "activity_coverage_start_at",
          "activity_complete_through_at", "active_days_14d", "active_days_previous_14d", "interactions_14d",
          "interactions_previous_14d", "resource_views_14d", "forum_posts_14d", "last_activity_at",
          "assignments_due_to_date", "assignments_submitted_to_date", "late_submissions_to_date",
          "mean_score_pct_to_date", "scores_available_count", "source_system", "schema_version"]


def stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_snapshots() -> bytes:
    start = datetime(2025, 9, 1, tzinfo=timezone.utc)
    cutoff = start + timedelta(days=35)
    rows = []
    for index, values in enumerate(((9, 7, 46, 32, 28, 5, 72.5), (4, 10, 14, 37, 8, 1, 55.0), (2, 0, 3, 0, 2, 0, "")), 1):
        recent_days, prior_days, recent_events, prior_events, views, posts, score = values
        last = stamp(cutoff - timedelta(days=2)) if recent_days else stamp(cutoff - timedelta(days=18)) if prior_days else ""
        row = ["demo_university", f"SYN_WORKER_{index:03d}", "COURSE_01", "2025_S2", stamp(cutoff),
               stamp(start), stamp(start + timedelta(days=112)), stamp(start - timedelta(days=1)), "active",
               stamp(start), stamp(cutoff), recent_days, prior_days, recent_events, prior_events, views, posts,
               last, 2, 2 if score != "" else 1, 0, score, 1 if score != "" else 0, "synthetic_moodle", "1.0"]
        rows.append(row)
    from io import StringIO
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def write_training_fixture() -> None:
    raw = DEST / "dropout-training" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    snapshot_rows, outcome_rows = [], []
    cohorts = [("2024_S1", datetime(2024, 2, 5, tzinfo=timezone.utc)),
               ("2024_S2", datetime(2024, 9, 2, tzinfo=timezone.utc)),
               ("2025_S1", datetime(2025, 2, 3, tzinfo=timezone.utc))]
    rng = random.Random(903)
    for cohort, start in cohorts:
        cutoff, end = start + timedelta(days=35), start + timedelta(days=112)
        for index in range(40):
            withdrawn = index % 2 == 0
            recent = 2 + (index % 4) if withdrawn else 7 + (index % 7)
            previous = recent + (index % 3) if withdrawn else max(0, recent - 2)
            interactions = recent * (2 + index % 4)
            previous_interactions = previous * (2 + index % 3)
            row = {
                "tenant_id": "fixture-tenant", "student_key": f"SYN_{cohort}_{index:03d}",
                "course_id": "COURSE_01", "cohort_id": cohort, "snapshot_at": stamp(cutoff),
                "course_start_at": stamp(start), "course_end_at": stamp(end),
                "enrolled_at": stamp(start - timedelta(days=2)), "enrolment_status_at_snapshot": "active",
                "activity_coverage_start_at": stamp(start), "activity_complete_through_at": stamp(cutoff),
                "active_days_14d": recent, "active_days_previous_14d": previous,
                "interactions_14d": interactions, "interactions_previous_14d": previous_interactions,
                "resource_views_14d": min(interactions, interactions // 2),
                "forum_posts_14d": min(interactions - min(interactions, interactions // 2), index % 3),
                "last_activity_at": stamp(cutoff - timedelta(days=1 + index % 10)),
                "assignments_due_to_date": 2, "assignments_submitted_to_date": 1 + index % 2,
                "late_submissions_to_date": int(withdrawn),
                "mean_score_pct_to_date": str(45 + index % 20) if index % 5 else "",
                "scores_available_count": 1 if index % 5 else 0,
                "source_system": "synthetic_moodle", "schema_version": "1.0",
            }
            snapshot_rows.append(row)
            outcome = {key: row[key] for key in FIELDS[:5]}
            outcome.update({
                "final_outcome": "withdrawn" if withdrawn else ("failed" if index % 7 == 1 else "completed"),
                "withdrawn_at": stamp(cutoff + timedelta(days=20 + index % 35)) if withdrawn else "",
                "outcome_observed_at": stamp(end + timedelta(days=14)),
            })
            outcome_rows.append(outcome)
    snapshot_path, outcome_path = raw / "lms_snapshots.csv", raw / "student_outcomes.csv"
    for path, fields, rows in ((snapshot_path, FIELDS, snapshot_rows), (outcome_path, FIELDS[:5] + ["final_outcome", "withdrawn_at", "outcome_observed_at"], outcome_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "synthetic": True, "provenance": "synthetic_platform_fixture", "fixture_only": True,
        "export_id": "fixture-dropout-903", "schema_version": "1.0",
        "exported_at": "2026-11-25T00:00:00Z", "timezone": "UTC",
        "extraction_version": "generate_worker_fixtures.py:1", "included_event_categories": ["course_view", "forum_post"],
        "coverage_limitations": ["Fictional learner activity and outcomes; training fixture only."],
        "platform_scope": {"tenant_id": "fixture-tenant"},
        "cohorts": [cohort for cohort, _ in cohorts],
        "split": {"train": ["2024_S1"], "validation": ["2024_S2"], "test": ["2025_S1"]},
        "row_counts": {"lms_snapshots": len(snapshot_rows), "student_outcomes": len(outcome_rows)},
        "files": {},
    }
    for path in (snapshot_path, outcome_path):
        manifest["files"][path.relative_to(DEST / "dropout-training").as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    (DEST / "dropout-training" / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot = write_snapshots()
    (DEST / "dropout-snapshots.csv").write_bytes(snapshot)
    topic_a = ("Water conservation in dry regions depends on repairing distribution leaks, measuring household demand, "
               "and adapting irrigation schedules to local soil conditions. Public agencies can compare seasonal "
               "measurements to identify where conservation programs are useful.")
    topic_b = ("A library digitization project should preserve source metadata, use redundant storage, and schedule "
               "checksum audits. Curators can then track file migrations without confusing a preservation copy with "
               "a public access copy.")
    plagiarism = {
        "scope": {"institution_id": "fixture-university", "course_id": "fixture-course", "assignment_id": "fixture-assignment"},
        "reports": [
            {"report_id": "report-copy-source", "version_id": "v1", "text": topic_a + " " + topic_b},
            {"report_id": "report-copy-submission", "version_id": "v1", "text": "Field notes: " + topic_a},
            {"report_id": "report-independent", "version_id": "v1", "text": "Independent notes: " +
             "A field survey of coastal bird migration records wind, tide, and nesting observations for each route. " +
             "Researchers can compare seasonal arrival patterns across protected wetlands."},
        ],
    }
    # Strip the fixture-only annotation before use as a worker payload; strict worker envelopes accept only payload fields.
    (DEST / "plagiarism-score-payload.json").write_text(json.dumps(plagiarism, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    manifest = {"fixture_only": True, "seed": 903, "files": {}}
    for name in ("dropout-snapshots.csv", "plagiarism-score-payload.json"):
        manifest["files"][name] = hashlib.sha256((DEST / name).read_bytes()).hexdigest()
    (DEST / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    write_training_fixture()


if __name__ == "__main__":
    main()
