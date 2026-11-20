"""Build and register a train-only reference profile for one gated model version."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from dropout.monitor import create_and_register_reference_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="DVC-restored snapshot export directory")
    parser.add_argument("--model-version", required=True, help="Concrete gate-passed MLflow registered version")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--output", help="Local copy path; defaults to reports/monitoring/reference-profile-v<version>.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_name = str(config["model_name"])
    output = args.output or f"reports/monitoring/reference-profile-v{args.model_version}.json"
    result = create_and_register_reference_profile(
        args.data_dir,
        args.model_version,
        tracking_uri=args.tracking_uri,
        model_name=model_name,
        output_path=output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
