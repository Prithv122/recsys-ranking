"""MovieLens 1M loading and download. The data is never committed (see GUIDELINES.md)."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_ranking.types import Interactions

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# Recorded from the first download on 2026-10-23 (5,917,549 bytes, 1,000,209 ratings).
ML1M_SHA256: str | None = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"

_EXPECTED_BYTES = 5_900_000  # the archive is about 5.9 MB
_MIN_BYTES = 1_000_000
_MAX_BYTES = 50_000_000
_MAX_UNPACKED_BYTES = 200_000_000
_FOLDER = "ml-1m"
_HASH_FILE = "ml-1m.zip.sha256"


def load_ml1m(path: str | Path) -> Interactions:
    """Parse ``ratings.dat`` (``UserID::MovieID::Rating::Timestamp``), all ratings kept.

    ``path`` is the ratings file itself or the folder that contains it.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "ratings.dat"
    if not p.is_file():
        raise FileNotFoundError(f"ratings file not found: {p}")
    # "::" is not a single-character separator, so collapse it and use the fast C parser.
    raw = p.read_bytes().replace(b"::", b":")
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=":",
            header=None,
            names=["user", "item", "rating", "ts"],
            dtype={"user": np.int64, "item": np.int64, "rating": np.float32, "ts": np.int64},
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"{p} is not a valid ratings.dat: {exc}") from exc
    return Interactions(
        user=df["user"].to_numpy(dtype=np.int64),
        item=df["item"].to_numpy(dtype=np.int64),
        ts=df["ts"].to_numpy(dtype=np.int64),
        rating=df["rating"].to_numpy(dtype=np.float32),
    )


def positives(inter: Interactions, threshold: float = 4.0) -> Interactions:
    """Keep rows with ``rating >= threshold``; lower ratings are dropped, not negatives."""
    keep = inter.rating >= threshold
    return Interactions(
        user=inter.user[keep],
        item=inter.item[keep],
        ts=inter.ts[keep],
        rating=inter.rating[keep],
    )


def _is_safe_member(name: str) -> bool:
    """Reject absolute paths, drive letters and ``..`` components (zip-slip)."""
    if not name or name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
        return False
    return ".." not in name.replace("\\", "/").split("/")


def _extract(data: bytes, dest_dir: Path) -> None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("downloaded file is not a valid zip archive") from exc
    with zf:
        infos = zf.infolist()
        for info in infos:
            if not _is_safe_member(info.filename):
                raise ValueError(f"unsafe path in archive: {info.filename!r}")
        if sum(i.file_size for i in infos) > _MAX_UNPACKED_BYTES:
            raise ValueError("archive unpacks to an implausible size")
        root = dest_dir.resolve()
        for info in infos:
            target = (root / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe path in archive: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))


def _fetch(url: str) -> bytes:
    """Read the response in chunks and stop early on an implausibly large body."""
    chunks: list[bytes] = []
    total = 0
    with urllib.request.urlopen(url, timeout=60) as resp:
        while chunk := resp.read(1 << 20):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_BYTES:
                raise ValueError(f"download exceeds {_MAX_BYTES} bytes; refusing to continue")
    return b"".join(chunks)


def download(dest_dir: Path = Path("data")) -> Path:
    """Fetch ``ml-1m.zip``, verify it, extract it under ``dest_dir`` and return the folder.

    Skipped when the folder is already extracted and its recorded hash is still valid.
    """
    dest_dir = Path(dest_dir)
    folder = dest_dir / _FOLDER
    marker = dest_dir / _HASH_FILE
    if (folder / "ratings.dat").is_file() and marker.is_file():
        recorded = marker.read_text(encoding="ascii").strip()
        if ML1M_SHA256 is None or recorded == ML1M_SHA256:
            return folder

    data = _fetch(ML1M_URL)
    if not _MIN_BYTES <= len(data) <= _MAX_BYTES:
        raise ValueError(
            f"downloaded {len(data)} bytes, expected about {_EXPECTED_BYTES}; refusing to extract"
        )
    digest = hashlib.sha256(data).hexdigest()
    if ML1M_SHA256 is not None and digest != ML1M_SHA256:
        raise ValueError(f"sha256 mismatch for ml-1m.zip: got {digest}, expected {ML1M_SHA256}")
    if ML1M_SHA256 is None:
        print(f"ml-1m.zip sha256 = {digest}")
        print("ML1M_SHA256 is not set: record this value in recsys_ranking/data.py to pin it.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    _extract(data, dest_dir)
    if not (folder / "ratings.dat").is_file():
        raise ValueError(f"archive did not contain {_FOLDER}/ratings.dat")
    marker.write_text(digest + "\n", encoding="ascii")
    return folder
