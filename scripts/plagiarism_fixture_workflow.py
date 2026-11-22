"""Exercise the complete local fixture release path with real XGBoost training.

The fixture is authored synthetic data, so the output validates software
plumbing only. It is not a benchmark or a performance claim.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dropout.registry import configure_client
from plagiarism.score import parse_submissions
from promote_model import promote
from rollback_model import rollback


def _normalise_tracking_uri(uri: str) -> str:
    if not uri.startswith("sqlite:///"):
        return uri
    suffix = uri[len("sqlite:///"):]
    database = Path(suffix)
    if not database.is_absolute():
        database = ROOT / database
    return "sqlite:///" + database.resolve().as_posix()


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _train(*, config: Path, result: Path, tracking_uri: str, cache: Path, env: dict[str, str]) -> dict:
    _run([
        sys.executable, "flows/train_plagiarism_flow.py", "run",
        "--data-dir", str(ROOT / "fixtures/plagiarism"),
        "--config-path", str(config), "--tracking-uri", tracking_uri,
        "--cache-dir", str(cache), "--result-path", str(result), "--fixture", "true",
    ], env=env)
    return _read_json(result)


def _score(*, tracking_uri: str, version: str, input_path: Path, output_path: Path, env: dict[str, str]) -> None:
    _run([
        sys.executable, "-m", "plagiarism.score",
        "--input", str(input_path), "--output", str(output_path),
        "--tracking-uri", tracking_uri,
        "--model-name", "passage-reuse-fixture", "--version", version, "--fixture",
    ], env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///reports/plagiarism/fixture-workflow.db")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/plagiarism/fixture-workflow")
    args = parser.parse_args()
    os.chdir(ROOT)
    args.tracking_uri = _normalise_tracking_uri(args.tracking_uri)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    release_log = output_dir / "releases.json"
    if release_log.exists():
        raise SystemExit(f"release log already exists; use a fresh output directory: {release_log}")
    cache_dir = ROOT / "artifacts/plagiarism/fixture-workflow-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", args.tracking_uri)
    env.setdefault("METAFLOW_USER", "student-mlops")
    env.setdefault("METAFLOW_DATASTORE_SYSROOT_LOCAL", str(ROOT / "artifacts/plagiarism/metaflow"))
    env.setdefault("METAFLOW_CARD_LOCALROOT", str(ROOT / "artifacts/plagiarism/mf.cards"))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH", "")]))

    config = ROOT / "config/plagiarism-fixture.yaml"
    base = yaml.safe_load(config.read_text(encoding="utf-8"))
    # Two fewer alerts than the MiniLM cosine baseline is impossible here:
    # validation contains only one negative example. This remains a valid,
    # explicit release policy and exercises the production gate code.
    failed_config_data = json.loads(json.dumps(base))
    failed_config_data["config_version"] = "1.0-fixture-strict-alert-policy"
    failed_config_data["gates"]["minimum_false_alert_reduction"] = 2
    failed_config = output_dir / "plagiarism-fixture-failed-gate.yaml"
    failed_config.write_text(yaml.safe_dump(failed_config_data, sort_keys=False), encoding="utf-8")

    first = _train(config=config, result=output_dir / "train-1.json", tracking_uri=args.tracking_uri, cache=cache_dir, env=env)
    second = _train(config=config, result=output_dir / "train-2.json", tracking_uri=args.tracking_uri, cache=cache_dir, env=env)
    if not first.get("gate_passed") or not second.get("gate_passed"):
        raise SystemExit("authored fixture candidate failed its configured gate; refusing to claim release/rollback success")
    first_version, second_version = str(first["candidate_version"]), str(second["candidate_version"])
    if first_version == second_version or not first_version.isdecimal() or not second_version.isdecimal():
        raise SystemExit("the two passing fixture runs did not create distinct concrete registered versions")

    client = configure_client(args.tracking_uri)
    versions_before_failure = sorted(str(v.version) for v in client.search_model_versions("name='passage-reuse-fixture'"))
    failed = _train(config=failed_config, result=output_dir / "failed-gate.json", tracking_uri=args.tracking_uri, cache=cache_dir, env=env)
    if failed.get("gate_passed") or failed.get("candidate_version") is not None:
        raise SystemExit("the deliberately strict validation gate did not fail as expected")
    failed_report = Path(str(failed["report_dir"])) / "gate_report.json"
    failed_gate = _read_json(failed_report)
    if not any(not check["passed"] for check in failed_gate["checks"]):
        raise SystemExit("failed flow has no failed gate check")
    versions_after_failure = sorted(str(v.version) for v in client.search_model_versions("name='passage-reuse-fixture'"))
    if versions_after_failure != versions_before_failure:
        raise SystemExit("failed-gate candidate unexpectedly added a registered model version")

    promote(first_version, tracking_uri=args.tracking_uri, config_path=config, release_log=release_log)
    champion = client.get_model_version_by_alias("passage-reuse-fixture", "champion")
    if str(champion.version) != first_version:
        raise SystemExit("first explicit promotion did not set the fixture champion")

    input_path = output_dir / "submissions.jsonl"
    fixture = _read_json(ROOT / "fixtures/plagiarism/fixture.json")
    submission = next(row for row in fixture["documents"] if row["role"] == "submission" and "train" in row["id"])
    input_path.write_text(json.dumps({"submission_id": "workflow-query", "text": submission["text"]}) + "\n", encoding="utf-8")
    before = output_dir / "score-before-second-promotion.jsonl"
    _score(tracking_uri=args.tracking_uri, version="champion", input_path=input_path, output_path=before, env=env)

    promote(second_version, tracking_uri=args.tracking_uri, config_path=config, release_log=release_log)
    after_second = output_dir / "score-second-version.jsonl"
    _score(tracking_uri=args.tracking_uri, version="champion", input_path=input_path, output_path=after_second, env=env)
    rollback(tracking_uri=args.tracking_uri, config_path=config, release_log=release_log)
    restored = client.get_model_version_by_alias("passage-reuse-fixture", "champion")
    if str(restored.version) != first_version:
        raise SystemExit(f"rollback restored {restored.version}; expected {first_version}")
    after_rollback = output_dir / "score-after-rollback.jsonl"
    _score(tracking_uri=args.tracking_uri, version="champion", input_path=input_path, output_path=after_rollback, env=env)
    if before.read_bytes() != after_rollback.read_bytes():
        raise SystemExit("scored JSONL changed after rollback to the original version")
    # Each record reports the exact concrete model version used.
    records = parse_submissions(input_path.read_text(encoding="utf-8").splitlines())
    restored_record = _read_json_line(after_rollback)
    if len(records) != 1 or restored_record["model_version"] != first_version:
        raise SystemExit("rollback score did not identify the restored concrete version")
    fixture_annotation = next(row for row in fixture["annotations"] if row["submission_id"] == submission["id"])
    matching_flag = any(
        flag["source_id"] == fixture_annotation["source_id"]
        and flag["submission_start"] < fixture_annotation["submission_end"]
        and fixture_annotation["submission_start"] < flag["submission_end"]
        and flag["source_start"] < fixture_annotation["source_end"]
        and fixture_annotation["source_start"] < flag["source_end"]
        for flag in restored_record["flags"]
    )
    if not matching_flag:
        raise SystemExit("fixture score did not return a flagged passage overlapping its authored annotation")
    first_gate = _read_json(Path(str(first["report_dir"])) / "gate_report.json")
    candidate_metrics = first_gate["candidate"]
    summary = {"status": "fixture_workflow_passed", "synthetic_fixture": True,
                      "failed_gate": True, "promoted_versions": [first_version, second_version],
                      "rollback_version": first_version, "score_jsonl_equal_after_rollback": True,
                      "annotated_overlap_flag_found": True,
                      "validation_comparison": {
                          "minilm_cosine_false_alerts": candidate_metrics.get("minilm_cosine_false_alerts"),
                          "xgboost_false_alerts": candidate_metrics.get("false_alerts"),
                          "minilm_cosine_recall": candidate_metrics.get("minilm_cosine_recall"),
                          "xgboost_recall": candidate_metrics.get("recall"),
                      }, "output_dir": str(output_dir)}
    (output_dir / "workflow-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def _read_json_line(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise SystemExit(f"expected one score JSONL record, found {len(lines)}")
    return json.loads(lines[0])


if __name__ == "__main__":
    raise SystemExit(main())
