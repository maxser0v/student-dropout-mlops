"""Run one isolated, traceable container training-and-scoring demonstration."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_provenance import git_provenance
from dropout.lineage import verify_dvc_directory

SETTING_KEYS = {
    "DEMO_MLFLOW_PORT",
    "DOCKER_PLATFORM",
    "RUNTIME_UID",
    "RUNTIME_GID",
    "METAFLOW_USER",
}
SIMPLE_VALUE = re.compile(r"^[A-Za-z0-9_./:@+-]+$")


class DemoError(RuntimeError):
    """A prerequisite or integrity check prevented the demo from running."""


def _parse_plain_env(path: Path) -> dict[str, str]:
    """Parse the intentionally small dotenv subset used by this project."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise DemoError(f"{path.name}:{line_no}: expected KEY=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in SETTING_KEYS:
            raise DemoError(f"{path.name}:{line_no}: unknown setting {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not SIMPLE_VALUE.fullmatch(value):
            raise DemoError(
                f"{path.name}:{line_no}: setting values must be plain text; shell syntax and whitespace are not supported"
            )
        values[key] = value
    return values


def load_settings(root: Path = ROOT, environ: dict[str, str] | None = None) -> dict[str, str]:
    settings = _parse_plain_env(root / ".env.example")
    settings.update(_parse_plain_env(root / ".env"))
    env = os.environ if environ is None else environ
    for key in SETTING_KEYS:
        if key in env:
            value = env[key]
            if not SIMPLE_VALUE.fullmatch(value):
                raise DemoError(f"environment setting {key} must be plain text")
            settings[key] = value
    settings.setdefault("DEMO_MLFLOW_PORT", "5000")
    settings.setdefault("DOCKER_PLATFORM", "linux/arm64")
    settings.setdefault("RUNTIME_UID", "auto")
    settings.setdefault("RUNTIME_GID", "auto")
    settings.setdefault("METAFLOW_USER", "student-mlops")
    return settings


def runtime_identity(settings: dict[str, str], *, uid: int | None = None, gid: int | None = None) -> tuple[int, int]:
    host_uid = os.getuid() if uid is None else uid
    host_gid = os.getgid() if gid is None else gid
    if host_uid == 0:
        raise DemoError("run `make demo` as your normal non-root user so bind-mounted files remain writable")
    resolved_uid = host_uid if settings["RUNTIME_UID"] == "auto" else int(settings["RUNTIME_UID"])
    resolved_gid = host_gid if settings["RUNTIME_GID"] == "auto" else int(settings["RUNTIME_GID"])
    if resolved_uid <= 0 or resolved_gid < 0:
        raise DemoError("application services require a nonzero RUNTIME_UID and nonnegative RUNTIME_GID")
    return resolved_uid, resolved_gid


def ensure_data(root: Path, run=subprocess.run) -> dict[str, Any]:
    data_dir = root / "data"
    if not data_dir.exists():
        print("data/ is absent; generating the deterministic fictional dataset.", flush=True)
        run([sys.executable, "scripts/generate_synthetic_data.py"], cwd=root, check=True)
    if not data_dir.is_dir():
        raise DemoError("data exists but is not a directory; refusing to replace it")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DemoError("data/ exists without manifest.json; refusing to overwrite or regenerate it")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoError(f"data/manifest.json is unreadable: {exc}") from exc
    if manifest.get("synthetic") is not True:
        raise DemoError("the one-command demo accepts only the bundled fictional dataset; existing non-synthetic data was left untouched")

    expected_files = {"manifest.json"}
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise DemoError("synthetic manifest is missing its file checksum map")
    for relative in declared:
        normalized = Path(relative)
        if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts or normalized.parts[0] != "data":
            raise DemoError(f"invalid dataset manifest path {relative!r}")
        expected_files.add(normalized.relative_to("data").as_posix())
    pointer = root / "data.dvc"
    if not pointer.is_file():
        raise DemoError("data.dvc is missing; cannot verify the dataset-to-DVC binding")
    binding = verify_dvc_directory(
        pointer_file=pointer,
        data_dir=data_dir,
        expected_path="data",
        expected_files=expected_files,
    )
    if not isinstance(binding, dict) or binding.get("passed") is not True:
        reason = binding.get("reason") if isinstance(binding, dict) else "verifier returned an invalid result"
        raise DemoError(f"DVC data binding check failed: {reason}")
    run([sys.executable, "scripts/verify_synthetic_data.py"], cwd=root, check=True)
    return binding


def _run(command: list[str], *, cwd: Path, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=check, text=True)


def _free_loopback_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])


def _assert_command(command: list[str], *, cwd: Path, env: dict[str, str], label: str) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise DemoError(f"{label} is required and must be working. Command output:\n{result.stdout.strip()}")


def _wait_http(url: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(2)
    raise DemoError(f"MLflow did not become healthy at {url}: {last_error}")


def _write_compose_env(path: Path, values: dict[str, str]) -> None:
    # Docker Compose reads this file as dotenv, never through a shell.
    def quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    path.write_text("".join(f"{key}={quote(value)}\n" for key, value in sorted(values.items())), encoding="utf-8")


def _safe_project_mkdir(root: Path, path: Path) -> None:
    """Create a project output directory only through real in-project parents."""
    resolved_root = root.resolve()
    candidate = root / path if not path.is_absolute() else path
    relative = candidate.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise DemoError(f"refusing to write through project output symlink: {cursor.relative_to(root)}")
        if cursor.exists():
            if not cursor.is_dir():
                raise DemoError(f"project output path component is not a directory: {cursor.relative_to(root)}")
        else:
            cursor.mkdir()
        if not cursor.resolve().is_relative_to(resolved_root):
            raise DemoError(f"project output path escapes repository root: {cursor}")


def compose_environment(base: dict[str, str], values: dict[str, str]) -> dict[str, str]:
    """Ensure validated, computed values win over inherited shell settings."""
    result = base.copy()
    for control in ("COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME", "COMPOSE_ENV_FILES"):
        result.pop(control, None)
    result.update(values)
    return result


def compose_command(docker: str, root: Path, env_file: Path, project: str) -> list[str]:
    """Pin Compose inputs so ambient COMPOSE_FILE cannot switch the project."""
    return [docker, "compose", "--file", str(root / "compose.yaml"), "--env-file", str(env_file), "--project-name", project]


def cleanup_project(compose: list[str], *, cwd: Path, env: dict[str, str], run=subprocess.run):
    """Stop this isolated project while retaining its named data volume."""
    return run(compose + ["down"], cwd=cwd, env=env, check=False)


def _read_training_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoError(f"training did not write its machine-readable result at {path}: {exc}") from exc
    required = {"status", "gate_passed", "candidate_version", "mlflow_run_id", "metaflow_run_id"}
    missing = required - set(result) if isinstance(result, dict) else required
    if missing:
        raise DemoError(f"training result is missing fields: {', '.join(sorted(missing))}")
    if result["status"] != "completed" or result["gate_passed"] is not True:
        raise DemoError("training completed without a gate-passing candidate; scoring was skipped")
    version = str(result["candidate_version"])
    if not version.isdigit() or int(version) < 1:
        raise DemoError("training result contains an invalid concrete candidate version")
    for key in ("mlflow_run_id", "metaflow_run_id"):
        if not isinstance(result[key], str) or not result[key]:
            raise DemoError(f"training result contains an invalid {key}")
    return result


def run_demo(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    settings = load_settings(root)
    if settings["DOCKER_PLATFORM"] != "linux/arm64":
        raise DemoError("the pinned base-image digest is verified for linux/arm64; DOCKER_PLATFORM must remain linux/arm64")
    uid, gid = runtime_identity(settings)
    docker = shutil.which("docker")
    if not docker:
        raise DemoError("Docker CLI was not found; install Docker Desktop or Docker Engine first")
    command_env = os.environ.copy()
    command_env.update(settings)
    _assert_command([docker, "compose", "version"], cwd=root, env=command_env, label="Docker Compose")
    _assert_command([docker, "info", "--format", "{{.ServerVersion}}"], cwd=root, env=command_env, label="a reachable Docker daemon")
    for required in ("data.dvc", "Dockerfile", "compose.yaml", "config/model.yaml"):
        if not (root / required).is_file():
            raise DemoError(f"required project file is missing: {required}")

    binding = ensure_data(root)
    provenance = git_provenance(root)
    demo_id = uuid.uuid4().hex
    compose_project = f"finalassignment-demo-{demo_id[:12]}"
    image = f"student-dropout:demo-{demo_id[:12]}"
    _safe_project_mkdir(root, root / "reports")
    _safe_project_mkdir(root, root / "outputs")
    _safe_project_mkdir(root, root / ".metaflow" / "mf.cards")
    report_dir = root / "reports" / "demo_runs" / demo_id
    output_dir = root / "outputs" / "demo_runs" / demo_id
    _safe_project_mkdir(root, report_dir)
    _safe_project_mkdir(root, output_dir)
    result_path = f"/app/reports/demo_runs/{demo_id}/training-result.json"
    score_path = f"/app/outputs/demo_runs/{demo_id}/predictions.csv"
    requested_port = int(settings["DEMO_MLFLOW_PORT"])
    if not 0 < requested_port < 65536:
        raise DemoError("DEMO_MLFLOW_PORT must be between 1 and 65535")
    port = _free_loopback_port(requested_port)
    if port != requested_port:
        print(f"Port {requested_port} is busy; using available loopback port {port}.", flush=True)

    compose_values = {
        "DEMO_MLFLOW_PORT": str(port),
        "DEMO_IMAGE": image,
        "DOCKER_PLATFORM": settings["DOCKER_PLATFORM"],
        "PROJECT_ROOT": str(root),
        "RUNTIME_UID": str(uid),
        "RUNTIME_GID": str(gid),
        "METAFLOW_USER": settings["METAFLOW_USER"],
        "GIT_COMMIT": provenance["git_commit"],
        "GIT_DIRTY": str(provenance["git_dirty"]).lower(),
        "SOURCE_TREE_SHA256": provenance["source_tree_sha256"],
        "SOURCE_REVISION_SOURCE": provenance["source_revision_source"],
        "SOURCE_TREE_HASH_ALGORITHM": provenance["source_tree_hash_algorithm"],
        "DockerImageID": "unknown",
        "TRAIN_RESULT_PATH": result_path,
    }
    env_file = report_dir / "compose.env"
    _write_compose_env(env_file, compose_values)
    compose = compose_command(docker, root, env_file, compose_project)
    compose_env = compose_environment(command_env, compose_values)

    _run(compose + ["build", "mlflow"], cwd=root, env=compose_env)
    inspected = subprocess.run(
        [docker, "image", "inspect", image, "--format", "{{.Id}}"],
        cwd=root,
        env=command_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", inspected):
        raise DemoError(f"Docker returned an invalid image ID: {inspected!r}")
    labels_raw = subprocess.run(
        [docker, "image", "inspect", image, "--format", "{{json .Config.Labels}}"],
        cwd=root,
        env=command_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    labels = json.loads(labels_raw) if labels_raw and labels_raw != "<no value>" else {}
    expected_labels = {
        "org.opencontainers.image.revision": provenance["git_commit"],
        "org.opencontainers.image.revision.dirty": str(provenance["git_dirty"]).lower(),
        "org.opencontainers.image.source-tree.sha256": provenance["source_tree_sha256"],
        "org.opencontainers.image.revision.source": provenance["source_revision_source"],
        "org.opencontainers.image.source-tree.hash-algorithm": provenance["source_tree_hash_algorithm"],
    }
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected_labels.items()):
        raise DemoError("built image OCI provenance labels do not match the inspected host provenance")
    compose_values["DockerImageID"] = inspected
    _write_compose_env(env_file, compose_values)
    compose_env = compose_environment(command_env, compose_values)

    # The project name and therefore named volume are unique for this demo.
    # This root-only initializer cannot race the existing dashboard or reuse its volume.
    _run(compose + ["--profile", "init", "run", "--rm", "volume-init"], cwd=root, env=compose_env)
    _run(compose + ["up", "-d", "mlflow"], cwd=root, env=compose_env)
    endpoint = f"http://127.0.0.1:{port}"
    try:
        _wait_http(f"{endpoint}/health")

        runtime_env = compose_environment(command_env, compose_values)
        # Check the real application identity in all three application services.
        for service in ("mlflow", "train", "score"):
            result = subprocess.run(
                compose + (["exec", "-T", service, "id", "-u"] if service == "mlflow" else ["--profile", "jobs", "run", "--rm", "--no-deps", "--entrypoint", "id", service, "-u"]),
                cwd=root,
                env=runtime_env,
                text=True,
                check=True,
                stdout=subprocess.PIPE,
            )
            if not result.stdout.strip().isdigit() or int(result.stdout.strip()) == 0:
                raise DemoError(f"application service {service} did not run as a non-root user")

        _run(compose + ["--profile", "jobs", "run", "--rm", "train"], cwd=root, env=runtime_env)
        training = _read_training_result(report_dir / "training-result.json")
        _run(
            compose
            + [
                "--profile",
                "jobs",
                "run",
                "--rm",
                "score",
                "--input",
                "/app/data/examples/scoring_input.csv",
                "--output",
                score_path,
                "--model-version",
                str(training["candidate_version"]),
                "--tracking-uri",
                "http://mlflow:5000",
                "--config",
                "/app/config/model.yaml",
            ],
            cwd=root,
            env=runtime_env,
        )
        predictions = output_dir / "predictions.csv"
        if not predictions.is_file() or predictions.stat().st_size <= 0:
            raise DemoError("the scorer exited without a nonempty per-run prediction file")

        result = {
            "status": "completed",
            "demo_id": demo_id,
            "compose_project": compose_project,
            "image": image,
            "docker_image_id": inspected,
            "mlflow_endpoint": endpoint,
            "training": training,
            "reports_directory": str(report_dir),
            "predictions_file": str(predictions),
            "dvc_binding": binding,
            "provenance": provenance,
            "runtime_uid": uid,
            "runtime_gid": gid,
        }
        result_file = report_dir / "demo-result.json"
        result_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(
            f"\nMLflow is running at {endpoint}. Stop this isolated demo without deleting its volume with:\n"
            f"  {shlex.join(compose + ['down'])}\n",
            flush=True,
        )
        return result
    except BaseException:
        print(
            f"Demo failed after MLflow started. Stopping only {compose_project} and preserving its named volume; inspect {report_dir}.",
            file=sys.stderr,
            flush=True,
        )
        cleanup_project(compose, cwd=root, env=compose_env)
        raise


def main() -> int:
    try:
        run_demo()
    except (DemoError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
