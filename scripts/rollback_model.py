"""Restore the exact concrete version and bundle from the last promotion."""

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


def rollback(*, tracking_uri: str, config_path: str | Path, release_log: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_name = str(config["model_name"])
    with release_lock(release_log):
        events = read_release_log(release_log)
        promotions = [event for event in events if event.get("event") == "promote"]
        if not promotions:
            raise ValueError("there is no promotion in the release log to roll back")
        latest = promotions[-1]
        previous = latest.get("previous")
        promoted = latest.get("promoted")
        if not previous or not previous.get("model_version"):
            raise ValueError("the latest promotion has no previous champion version to restore")

        client = configure_client(tracking_uri)
        current_version, _ = current_champion(client, model_name)
        previous_version = str(previous["model_version"])
        promoted_version = str(promoted["model_version"])
        if current_version == previous_version:
            return {"action": "already_rolled_back", "model_name": model_name, "model_version": previous_version}
        if current_version != promoted_version:
            raise ValueError(
                f"champion is {current_version!r}; expected latest promoted version {promoted_version!r}; refusing to overwrite an unrelated alias change"
            )

        restored_version, restored_bundle = version_bundle(client, model_name, previous_version)
        client.set_registered_model_alias(model_name, "champion", previous_version)
        event = {
            "event": "rollback",
            "at_utc": utc_now(),
            "model_name": model_name,
            "alias": "champion",
            "from": release_entry(promoted_version, promoted.get("_bundle", {})) if promoted.get("_bundle") else promoted,
            "restored": release_entry(previous_version, restored_bundle),
        }
        events.append(event)
        try:
            write_release_log(release_log, events)
        except BaseException as audit_error:
            try:
                client.set_registered_model_alias(model_name, "champion", promoted_version)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"rollback audit write failed and the champion alias could not be restored: {rollback_error}"
                ) from audit_error
            raise
        return {"action": "rolled_back", **event}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///reports/mlflow/mlflow.db")
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--release-log", default="reports/releases.json")
    args = parser.parse_args()
    result = rollback(tracking_uri=args.tracking_uri, config_path=args.config, release_log=args.release_log)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
