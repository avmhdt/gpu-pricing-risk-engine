"""Immutable raw files and ``data/raw/MANIFEST.json`` (DQ-09).

Raw files are written once under ``raw/<source>/<YYYY-MM-DD>/`` and never overwritten. The
manifest records path, source URL (never a key), download time and SHA-256, so a data check can
prove the raw tree is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpre.manifest import utc_stamp

MANIFEST_NAME = "MANIFEST.json"


class RawFileExistsError(FileExistsError):
    """A raw file already exists; raw files are immutable."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RawStore:
    """Writer for one data root's ``raw/`` tree."""

    raw_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / MANIFEST_NAME

    def load_manifest(self) -> dict[str, dict[str, Any]]:
        """Manifest entries keyed by path relative to ``raw/``."""
        if not self.manifest_path.exists():
            return {}
        loaded: dict[str, dict[str, Any]] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        return loaded

    def _save_manifest(self, entries: dict[str, dict[str, Any]]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def path_for(self, source: str, day: str, name: str) -> Path:
        return self.raw_dir / source / day / name

    def write(self, path: Path, data: bytes, *, url: str, **meta: Any) -> Path:
        """Write ``data`` to ``path`` (must not exist) and record it in the manifest."""
        if path.exists():
            raise RawFileExistsError(f"raw file exists and is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        self.record(path, url=url, sha256=sha256_bytes(data), **meta)
        return path

    def record(self, path: Path, *, url: str, sha256: str, **meta: Any) -> None:
        """Add or refresh one manifest entry (used for files read in place, such as Stooq)."""
        entries = self.load_manifest()
        rel = (
            path.relative_to(self.raw_dir).as_posix()
            if path.is_relative_to(self.raw_dir)
            else (path.as_posix())
        )
        entries[rel] = {"url": url, "sha256": sha256, "recorded_utc": utc_stamp(), **meta}
        self._save_manifest(entries)

    def verify(self) -> list[str]:
        """Return manifest entries under ``raw/`` whose file is missing or whose hash changed."""
        problems = []
        for rel, entry in self.load_manifest().items():
            path = Path(rel) if Path(rel).is_absolute() else self.raw_dir / rel
            if not Path(rel).is_absolute():
                if not path.exists():
                    problems.append(f"missing: {rel}")
                elif sha256_file(path) != entry["sha256"]:
                    problems.append(f"hash changed: {rel}")
        return problems
