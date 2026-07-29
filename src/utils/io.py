"""Small shared I/O helpers: JSON read/write, directory creation, checksums.

``sha256_of_file`` / ``sha256_of_dataframe`` back the ``corpus_checksum``
field every run record must carry, so a given metrics artefact can be tied
back to the exact corpus content that produced it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, sort_keys=False)


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sha256_of_dataframe(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_csv(path: str | Path, df: pd.DataFrame, **kwargs: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, **kwargs)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, returning [] if it doesn't exist yet.

    Missing-file-as-empty lets scrapers resume into the same file across
    runs without a separate existence check at every call site.
    """
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
