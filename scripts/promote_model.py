"""Explicitly promote one gate-passing immutable MLflow model version."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from dropout.registry import (
    configure_client,
    current_champion,
    read_release_log,
    release_entry,
    release_lock,
    utc_now,
    version_bundle,
    write_release_log,
)


def promote(version: str, *, tracking_uri: str, config_path: str | Path, release_log: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_name = str(config["model_name"])
    with release_lock(release_log):
        # Validate the existing audit before any mutable registry operation.
        events = read_release_log(release_log)
        client = configure_client(tracking_uri)
        candidate_version, candidate_bundle = version_bundle(client, model_name, str(version))
        old_version, old_bundle = current_champion(client, model_name)
        if old_version == str(version):
            return {"action": "already_champion", "model_name": model_name, "model_version": str(version)}

        client.set_registered_model_alias(model_name, "champion", str(version))
        event = {
            "event": "promote",
            "at_utc": utc_now(),
            "model_name": model_name,
            "alias": "champion",
            "previous": release_entry(old_version, old_bundle),
            "promoted": release_entry(str(version), candidate_bundle),
        }
        events.append(event)
        try:
            write_release_log(release_log, events)
        except BaseException as audit_error:
            try:
                if old_version is None:
                    client.delete_registered_model_alias(model_name, "champion")
                else:
                    client.set_registered_model_alias(model_name, "champion", old_version)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"release audit write failed and the champion alias could not be restored: {rollback_error}"
                ) from audit_error
            raise
        return {"action": "promoted", **event}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Registered model version that passed the release gate")
    parser.add_argument("--tracking-uri", default="sqlite:///reports/mlflow/mlflow.db")
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--release-log", default="reports/releases.json")
    args = parser.parse_args()
    result = promote(args.version, tracking_uri=args.tracking_uri, config_path=args.config, release_log=args.release_log)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
