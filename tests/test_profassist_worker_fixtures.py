import json
from pathlib import Path
import shutil

import pytest

from dropout.validation import DataContractError, load_training_data, verify_manifest


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "profassist" / "dropout-training"
FIXTURE_SCOPE = {"tenant_id": "fixture-tenant"}


def test_synthetic_platform_fixture_is_marked_and_cohort_splits_are_chronological():
    manifest = verify_manifest(FIXTURE_ROOT, mode="platform-fixture", expected_scope=FIXTURE_SCOPE)
    training = load_training_data(
        FIXTURE_ROOT,
        train_cohorts=tuple(manifest["split"]["train"]),
        validation_cohorts=tuple(manifest["split"]["validation"]),
        test_cohorts=tuple(manifest["split"]["test"]),
    )
    assert manifest["synthetic"] is True
    assert manifest["fixture_only"] is True
    assert training.split_counts == {"train": 40, "validation": 40, "test": 40}
    assert training.exclusions == {"unknown_outcome": 0, "immature_outcome": 0}


def test_platform_manifest_cannot_be_mislabelled_as_real_or_open_arbitrary_paths(tmp_path):
    destination = tmp_path / "export"
    shutil.copytree(FIXTURE_ROOT, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    with pytest.raises(DataContractError, match="provenance does not match"):
        verify_manifest(destination, mode="platform", expected_scope=FIXTURE_SCOPE)

    manifest["files"]["../../outside.csv"] = {"sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataContractError, match="exactly the snapshot and outcome exports"):
        verify_manifest(destination, mode="platform-fixture", expected_scope=FIXTURE_SCOPE)
