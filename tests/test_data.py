"""Loading, thresholding and the download guards. No test touches the network."""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from recsys_ranking import data
from recsys_ranking.data import download, load_ml1m, positives

FIXTURE = Path(__file__).parent / "data" / "ratings_fixture.dat"
PAD = bytes(1_200_000)  # keeps the archive above the minimum plausible size


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def _good_zip() -> bytes:
    return _zip_bytes({"ml-1m/ratings.dat": FIXTURE.read_bytes(), "ml-1m/pad.bin": PAD})


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    """Replace urlopen with an in-memory response; returns the list of requested URLs."""
    calls: list[str] = []

    def fake_urlopen(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


# --- load_ml1m / positives ---------------------------------------------------------------


def test_load_fixture_file() -> None:
    inter = load_ml1m(FIXTURE)
    assert len(inter) == 52
    assert inter.user.dtype == np.int64
    assert inter.item.dtype == np.int64
    assert inter.ts.dtype == np.int64
    assert inter.rating.dtype == np.float32
    # First line of the file: user 1 rated movie 1 with 4 at 2000-01-01 + 20 days.
    assert (inter.user[0], inter.item[0], inter.rating[0], inter.ts[0]) == (1, 1, 4.0, 948_412_800)
    assert set(np.unique(inter.user)) == {1, 2, 3, 4, 5, 6, 7, 8, 9}
    assert inter.ts.min() >= 946_684_800  # 2000-01-01
    assert inter.ts.max() < 1_072_915_200  # 2004-01-01


def test_load_accepts_folder(tmp_path: Path) -> None:
    shutil.copy(FIXTURE, tmp_path / "ratings.dat")
    from_folder = load_ml1m(tmp_path)
    from_file = load_ml1m(FIXTURE)
    assert np.array_equal(from_folder.user, from_file.user)
    assert np.array_equal(from_folder.ts, from_file.ts)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ml1m(tmp_path)


def test_load_rejects_malformed_line(tmp_path: Path) -> None:
    bad = tmp_path / "ratings.dat"
    bad.write_text("1::2::4::946684800\n1::x::4::946684801\n", encoding="ascii")
    with pytest.raises(ValueError, match=r"not a valid ratings.dat"):
        load_ml1m(bad)


def test_positives_threshold_keeps_exactly_four() -> None:
    inter = load_ml1m(FIXTURE)
    pos = positives(inter)
    # 47 hand-written positives (ratings 4 and 5) and 5 lower ratings.
    assert len(pos) == 47
    assert pos.rating.min() == 4.0
    assert 9 not in set(pos.user)  # user 9 only has low ratings
    assert 12 not in set(pos.item)
    # Odd positions in the time order carry rating 5: 23 of them.
    assert int((pos.rating == 5.0).sum()) == 23
    assert len(positives(inter, threshold=5.0)) == 23
    assert len(positives(inter, threshold=1.0)) == 52


# --- download ----------------------------------------------------------------------------


def test_download_extracts_and_reports_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _good_zip()
    calls = _serve(monkeypatch, payload)
    monkeypatch.setattr(data, "ML1M_SHA256", None)

    folder = download(tmp_path / "data")

    assert calls == [data.ML1M_URL]
    assert folder == tmp_path / "data" / "ml-1m"
    assert len(load_ml1m(folder)) == 52
    out = capsys.readouterr().out
    assert hashlib.sha256(payload).hexdigest() in out
    assert "ML1M_SHA256" in out


def test_download_verifies_pinned_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _good_zip()
    _serve(monkeypatch, payload)
    monkeypatch.setattr(data, "ML1M_SHA256", hashlib.sha256(payload).hexdigest())
    folder = download(tmp_path)
    assert (folder / "ratings.dat").is_file()


def test_download_hash_mismatch_extracts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, _good_zip())
    monkeypatch.setattr(data, "ML1M_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        download(tmp_path)
    assert not (tmp_path / "ml-1m").exists()
    assert not (tmp_path / "ml-1m.zip.sha256").exists()


def test_download_skips_when_already_extracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, _good_zip())
    monkeypatch.setattr(data, "ML1M_SHA256", None)
    first = download(tmp_path)

    def must_not_be_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("second call should not download")

    monkeypatch.setattr("urllib.request.urlopen", must_not_be_called)
    assert download(tmp_path) == first


def test_download_refetches_when_recorded_hash_differs_from_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _good_zip()
    calls = _serve(monkeypatch, payload)
    monkeypatch.setattr(data, "ML1M_SHA256", None)
    download(tmp_path)
    monkeypatch.setattr(data, "ML1M_SHA256", hashlib.sha256(payload).hexdigest())
    (tmp_path / "ml-1m.zip.sha256").write_text("stale\n", encoding="ascii")
    download(tmp_path)
    assert len(calls) == 2


@pytest.mark.parametrize("bad_name", ["../evil.txt", "ml-1m/../../evil.txt", "/abs/evil.txt"])
def test_download_blocks_zip_slip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_name: str
) -> None:
    payload = _zip_bytes({"ml-1m/ratings.dat": b"1::1::5::1\n", bad_name: b"x", "pad.bin": PAD})
    _serve(monkeypatch, payload)
    monkeypatch.setattr(data, "ML1M_SHA256", None)
    dest = tmp_path / "inner" / "data"
    with pytest.raises(ValueError, match="unsafe path"):
        download(dest)
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "inner" / "evil.txt").exists()
    assert not (dest / "ml-1m" / "ratings.dat").exists()  # nothing was extracted


def test_download_rejects_tiny_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _zip_bytes({"ml-1m/ratings.dat": b"1::1::5::1\n"}))
    with pytest.raises(ValueError, match="expected about"):
        download(tmp_path)
    assert not (tmp_path / "ml-1m").exists()


def test_download_stops_on_oversized_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reads = 0

    class Endless:
        def __enter__(self) -> Endless:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, n: int) -> bytes:
            nonlocal reads
            reads += 1
            return b"\0" * n

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Endless())
    with pytest.raises(ValueError, match="exceeds"):
        download(tmp_path)
    assert reads < 100  # stopped early instead of reading forever


def test_download_rejects_non_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, b"<html>" + PAD)
    monkeypatch.setattr(data, "ML1M_SHA256", None)
    with pytest.raises(ValueError, match="not a valid zip"):
        download(tmp_path)


def test_download_requires_ratings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _zip_bytes({"ml-1m/readme": b"hi", "ml-1m/pad.bin": PAD}))
    monkeypatch.setattr(data, "ML1M_SHA256", None)
    with pytest.raises(ValueError, match=r"ratings.dat"):
        download(tmp_path)
    assert not (tmp_path / "ml-1m.zip.sha256").exists()
