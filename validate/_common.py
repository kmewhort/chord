"""Shared plumbing for the validation suite: data paths, download, skip-guards.

Kept dependency-light. ``requests`` is only imported inside the download helper so
that merely importing an adapter (to read already-present data) does not require
the ``[validate]`` extra beyond pandas/pyarrow.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

# validate/data — the committed (Git LFS) root for dataset slices.
DATA_ROOT = Path(__file__).resolve().parent / "data"


def dataset_dir(name: str, create: bool = False) -> Path:
    """Return ``validate/data/<name>``; optionally create it."""
    d = DATA_ROOT / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def is_present(name: str, *required: str) -> bool:
    """True iff every ``required`` path (relative to the dataset dir) exists.

    A Git LFS pointer that was never pulled is a small text file, not the real
    artifact; treat a suspiciously tiny file as absent so tests skip cleanly
    rather than choking on a pointer.
    """
    base = dataset_dir(name)
    for rel in required:
        p = base / rel
        if not p.exists():
            return False
        if _looks_like_lfs_pointer(p):
            return False
    return True


def _looks_like_lfs_pointer(p: Path) -> bool:
    try:
        if p.is_dir():
            return False
        if p.stat().st_size > 1024:
            return False
        with open(p, "rb") as fh:
            head = fh.read(64)
        return head.startswith(b"version https://git-lfs")
    except OSError:
        return False


def require(name: str, *required: str) -> Path:
    """Skip the calling test if the dataset is missing; else return its dir.

    Uses :func:`pytest.skip` so the suite degrades gracefully on a machine that
    has not run ``python -m validate.fetch`` (or ``git lfs pull``).
    """
    if not is_present(name, *required):
        import pytest

        hint = f"python -m validate.fetch {name}"
        pytest.skip(
            f"dataset {name!r} not present under {dataset_dir(name)} "
            f"(missing one of {required!r}). Fetch with: {hint}"
        )
    return dataset_dir(name)


def download(url: str, dest: Path, *, expect_min_bytes: int = 1, force: bool = False,
             timeout: int = 120) -> Path:
    """Stream ``url`` to ``dest`` (skips if already present and non-trivial)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= expect_min_bytes and not force:
        print(f"  ✓ have {dest.name} ({dest.stat().st_size:,} B)")
        return dest
    import requests

    print(f"  ↓ {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest)
    size = dest.stat().st_size
    if size < expect_min_bytes:
        raise IOError(f"{url} produced only {size} bytes (< {expect_min_bytes})")
    print(f"  ✓ {dest.name} ({size:,} B)")
    return dest


def sha256(p: Path, limit: Optional[int] = None) -> str:
    """SHA-256 of a file (optionally only the first ``limit`` bytes)."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        read = 0
        while True:
            n = 1 << 20 if limit is None else min(1 << 20, limit - read)
            if n <= 0:
                break
            b = fh.read(n)
            if not b:
                break
            h.update(b)
            read += len(b)
    return h.hexdigest()


def record_finding(message: str) -> None:
    """Mark the current test as a *documented negative result* (imperative xfail).

    The whole point of this suite (Appendix C) is honest validation: a whitepaper
    claim that does **not** hold on real data is a finding to iterate on, not a bug
    to hide. Calling this prints the measured finding and marks the test XFAIL, so:

    * ``pytest validate/`` still exits 0 (the finding is expected), and
    * ``pytest -rx`` lists every finding with its numbers, and
    * if a future change makes the claim hold, the test XPASSes — a loud signal
      that the fix worked and the assertion should be tightened back to a hard one.

    See ``validate/FINDINGS.md`` for the running list.
    """
    import pytest

    print(f"\nFINDING (whitepaper claim did not hold on real data):\n  {message}")
    pytest.xfail(message)


def eprint(*a) -> None:
    print(*a, file=sys.stderr)


# Allow an env override so CI can point at a shared cache without editing code.
if os.environ.get("CHORD_VALIDATE_DATA"):
    DATA_ROOT = Path(os.environ["CHORD_VALIDATE_DATA"]).resolve()
