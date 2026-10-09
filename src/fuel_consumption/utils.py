"""Shared filesystem and configuration helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/project.json") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = project_root() / config_path
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, obj: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                            prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
