"""Pure, fail-closed candidate release gate calculations."""

from __future__ import annotations

from typing import Any


def evaluate_release_gate(
    candidate: dict[str, Any],
    settings: dict[str, Any],
    *,
    lineage_passed: bool,
    champion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe gate report; a missing metric or incompatible champion fails closed."""
    gates = settings["gates"]
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, comparator: str, expected: Any, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "actual": actual, "comparator": comparator, "expected": expected, "passed": bool(passed), "detail": detail})

    add("input_lineage", lineage_passed, "is", True, lineage_passed)
    ap = candidate.get("average_precision")
    prevalence = candidate.get("prevalence")
    ap_floor = None if prevalence is None else float(prevalence) + float(gates["average_precision_above_prevalence"])
    add("average_precision", ap, ">= prevalence + margin", ap_floor, ap is not None and ap_floor is not None and ap >= ap_floor)
    recall = candidate.get("recall")
    recall_floor = float(gates["minimum_recall"])
    add("recall", recall, ">=", recall_floor, recall is not None and recall >= recall_floor)
    precision = candidate.get("precision")
    precision_floor = float(gates["minimum_precision"])
    add("precision", precision, ">=", precision_floor, precision is not None and precision >= precision_floor)
    brier = candidate.get("brier_score")
    baseline_brier = candidate.get("train_prevalence_brier_score")
    add("brier_score", brier, "<= train-prevalence baseline", baseline_brier, brier is not None and baseline_brier is not None and brier <= baseline_brier)

    champion_comparison: dict[str, Any] | None = None
    if champion is not None:
        compatible = (
            candidate.get("target_version") == champion.get("target_version")
            and candidate.get("feature_schema_version") == champion.get("feature_schema_version")
        )
        if not compatible:
            champion_comparison = {
                "status": "incompatible",
                "reason": "target or feature schema differs; separate migration required",
                "candidate_target_version": candidate.get("target_version"),
                "champion_target_version": champion.get("target_version"),
                "candidate_feature_schema_version": candidate.get("feature_schema_version"),
                "champion_feature_schema_version": champion.get("feature_schema_version"),
            }
            add("champion_compatibility", False, "is", True, False, champion_comparison["reason"])
        else:
            champion_comparison = {"status": "compared", "metrics": {}}
            recall_limit = float(gates["maximum_recall_drop"])
            precision_limit = float(gates["maximum_precision_drop"])
            brier_limit = float(gates["maximum_brier_increase"])
            for name, limit, direction in (
                ("recall", recall_limit, "drop"),
                ("precision", precision_limit, "drop"),
            ):
                new_value, old_value = candidate.get(name), champion.get(name)
                passed = new_value is not None and old_value is not None and new_value >= old_value - limit
                add(f"champion_{name}", new_value, f">= champion - {direction} limit", None if old_value is None else old_value - limit, passed)
                champion_comparison["metrics"][name] = {"candidate": new_value, "champion": old_value, "maximum_drop": limit}
            new_brier, old_brier = candidate.get("brier_score"), champion.get("brier_score")
            passed = new_brier is not None and old_brier is not None and new_brier <= old_brier + brier_limit
            add("champion_brier_score", new_brier, "<= champion + increase limit", None if old_brier is None else old_brier + brier_limit, passed)
            champion_comparison["metrics"]["brier_score"] = {"candidate": new_brier, "champion": old_brier, "maximum_increase": brier_limit}

    return {
        "report_version": "1.0",
        "decision": "passed" if all(check["passed"] for check in checks) else "failed",
        "passed": all(check["passed"] for check in checks),
        "candidate": candidate,
        "settings": gates,
        "checks": checks,
        "champion_comparison": champion_comparison,
    }
