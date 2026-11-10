"""Run identity, run manifests and the append-only experiment log (FR-33, NFR-06).

Every command that produces results opens a :class:`Run`. It assigns a ``run_id``, writes a
manifest (config, seed, git hash, hardware stamp) to ``results/manifests/<run_id>.json`` and
appends one row to ``results/experiment_log.csv``. The log is the honest trial count.
"""

from __future__ import annotations

import csv
import json
import secrets
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpre.config import REPO_ROOT

EXPERIMENT_LOG_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "git_hash",
    "command",
    "config_path",
    "config_hash",
    "seed",
    "backend",
    "dtype",
    "gpu_name",
    "driver_version",
    "cuda_version",
    "key_metric_name",
    "key_metric_value",
    "output_path",
    "notes",
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_stamp(when: datetime | None = None) -> str:
    """ISO-8601 UTC with a trailing Z and whole seconds."""
    return (when or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(when: datetime | None = None) -> str:
    """Sortable, collision-resistant run id such as ``20261109T231500Z-3fa2c1``."""
    return f"{(when or utc_now()).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def git_hash(repo: Path = REPO_ROOT) -> str:
    """Short commit hash, ``-dirty`` if tracked files changed; ``unknown`` outside git."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def append_experiment_log(log_path: Path, row: dict[str, Any]) -> None:
    """Append one row; writes the header when the file is new. Unknown keys are rejected."""
    unknown = set(row) - set(EXPERIMENT_LOG_COLUMNS)
    if unknown:
        raise ValueError(f"unknown experiment-log columns: {sorted(unknown)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_LOG_COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({k: _cell(row.get(k)) for k in EXPERIMENT_LOG_COLUMNS})


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


@dataclass
class Run:
    """One results-producing invocation.

    Args:
        command: the CLI command line, e.g. ``gpre ingest fred``.
        results_dir: the committed results root.
        config: the config (as a mapping) this run used.
        config_path: where the config came from, if a file.
        seed: master seed, if the run is stochastic.
        hardware: hardware stamp from :func:`gpre.env.collect`, if relevant.
    """

    command: str
    results_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    config_path: str = ""
    seed: int | None = None
    backend: str = ""
    dtype: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=new_run_id)
    started: datetime = field(default_factory=utc_now)
    git: str = field(default_factory=git_hash)

    @property
    def manifest_path(self) -> Path:
        return self.results_dir / "manifests" / f"{self.run_id}.json"

    @property
    def log_path(self) -> Path:
        return self.results_dir / "experiment_log.csv"

    def finish(
        self,
        *,
        outputs: list[str],
        key_metric_name: str = "",
        key_metric_value: float | int | str | None = None,
        notes: str = "",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write the manifest and append the experiment-log row; return the manifest path."""
        from gpre.config import config_hash

        chash = config_hash(self.config) if self.config else ""
        manifest = {
            "run_id": self.run_id,
            "command": self.command,
            "started_utc": utc_stamp(self.started),
            "finished_utc": utc_stamp(),
            "git_hash": self.git,
            "config_path": self.config_path,
            "config_hash": chash,
            "config": self.config,
            "seed": self.seed,
            "backend": self.backend,
            "dtype": self.dtype,
            "hardware": self.hardware,
            "outputs": outputs,
            "key_metric": {"name": key_metric_name, "value": key_metric_value},
            "notes": notes,
            **({"extra": extra} if extra else {}),
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        append_experiment_log(
            self.log_path,
            {
                "run_id": self.run_id,
                "timestamp_utc": utc_stamp(self.started),
                "git_hash": self.git,
                "command": self.command,
                "config_path": self.config_path,
                "config_hash": chash,
                "seed": self.seed,
                "backend": self.backend,
                "dtype": self.dtype,
                "gpu_name": self.hardware.get("gpu_name", ""),
                "driver_version": self.hardware.get("driver_version", ""),
                "cuda_version": self.hardware.get("cuda_version", ""),
                "key_metric_name": key_metric_name,
                "key_metric_value": key_metric_value,
                "output_path": ";".join(outputs),
                "notes": notes,
            },
        )
        return self.manifest_path
