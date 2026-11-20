from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_final_test import final_test_metrics


def test_final_test_metrics_apply_frozen_threshold_without_search() -> None:
    report = final_test_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.10, 0.20, 0.65, 0.90]),
        0.65,
        0.25,
        model_version="1",
        target_version="target-v1",
        feature_schema_version="1.0",
        training_manifest_sha256="a" * 64,
        frozen_bundle_sha256="b" * 64,
    )
    assert report["evaluation_split"] == "final_test"
    assert report["threshold"] == 0.65
    assert report["support"] == 4
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["confusion_matrix"] == [[2, 0], [0, 2]]
    assert report["synthetic"] is True


@pytest.mark.parametrize(
    "labels, probabilities, message",
    [
        ([0, 1], [0.2], "same non-zero length"),
        ([0, 2], [0.2, 0.8], "binary"),
        ([0, 1], [0.2, float("nan")], "finite"),
    ],
)
def test_final_test_metrics_reject_bad_inputs(labels, probabilities, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        final_test_metrics(
            np.asarray(labels),
            np.asarray(probabilities),
            0.5,
            0.25,
            model_version="1",
            target_version="target-v1",
            feature_schema_version="1.0",
            training_manifest_sha256="a" * 64,
            frozen_bundle_sha256="b" * 64,
        )
