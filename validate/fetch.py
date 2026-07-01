"""Download the public validation datasets into ``validate/data`` (Appendix C).

Usage::

    python -m validate.fetch all
    python -m validate.fetch coat polis movielens signed_nets community_notes
    python -m validate.fetch community_notes --cn-shards 2   # pull more CN ratings

Small datasets (Coat, Polis, MovieLens, SNAP) land directly under
``validate/data/<name>/`` and are committed via Git LFS. Community Notes is
GB-scale: its raw parquet shards download into ``validate/data/community_notes/raw``
(git-ignored) and are distilled into a compact, committed ``slice/`` (a few MB).
Re-running is idempotent — present files are skipped.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

from ._common import dataset_dir, download

ALL = ["coat", "polis", "movielens", "signed_nets", "community_notes"]


# --------------------------------------------------------------------- Coat
def fetch_coat() -> None:
    print("[coat]")
    d = dataset_dir("coat", create=True)
    z = download("https://www.cs.cornell.edu/~schnabts/mnar/coat.zip",
                 d / "coat.zip", expect_min_bytes=500_000)
    if not (d / "coat" / "train.ascii").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    print("  extracted coat/")


# -------------------------------------------------------------------- Polis
def fetch_polis() -> None:
    print("[polis]")
    from .datasets.polis import CONVERSATIONS

    base = "https://raw.githubusercontent.com/compdemocracy/openData/master"
    d = dataset_dir("polis", create=True)
    for slug in CONVERSATIONS:
        for fname in ("votes.csv", "comments.csv", "participants-votes.csv"):
            try:
                download(f"{base}/{slug}/{fname}", d / slug / fname,
                         expect_min_bytes=10)
            except Exception as e:  # a conversation may lack a file; keep going
                print(f"  ! {slug}/{fname}: {e}")


# ---------------------------------------------------------------- MovieLens
def fetch_movielens() -> None:
    print("[movielens]")
    d = dataset_dir("movielens", create=True)
    z = download("https://files.grouplens.org/datasets/movielens/ml-100k.zip",
                 d / "ml-100k.zip", expect_min_bytes=1_000_000)
    if not (d / "ml-100k" / "u.data").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    print("  extracted ml-100k/")


# -------------------------------------------------------------- signed nets
def fetch_signed_nets() -> None:
    print("[signed_nets]")
    d = dataset_dir("signed_nets", create=True)
    download("https://snap.stanford.edu/data/wiki-RfA.txt.gz",
             d / "wiki-RfA.txt.gz", expect_min_bytes=1_000_000)


# ---------------------------------------------------------- Community Notes
CN_HF = ("https://huggingface.co/datasets/deadbirds/"
         "x-community-notes-parquet-20250222/resolve/main")


def fetch_community_notes(cn_shards: int = 1) -> None:
    print("[community_notes]")
    d = dataset_dir("community_notes", create=True)
    raw = d / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    download(f"{CN_HF}/notes/notes-00000.parquet",
             raw / "notes-00000.parquet", expect_min_bytes=10_000_000)
    download(f"{CN_HF}/note_status_history/noteStatusHistory-00000.parquet",
             raw / "noteStatusHistory-00000.parquet", expect_min_bytes=10_000_000)
    for i in range(cn_shards):
        download(f"{CN_HF}/ratings/ratings-{i:05d}.parquet",
                 raw / f"ratings-{i:05d}.parquet", expect_min_bytes=10_000_000)

    from .datasets import community_notes as cn

    print("  distilling committed slice ...")
    cn.build_slice(raw, d / "slice")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+", help=f"one of: all, {', '.join(ALL)}")
    ap.add_argument("--cn-shards", type=int, default=1,
                    help="number of Community Notes ratings shards to download")
    args = ap.parse_args(argv)

    targets = ALL if "all" in args.datasets else args.datasets
    fns = {
        "coat": fetch_coat,
        "polis": fetch_polis,
        "movielens": fetch_movielens,
        "signed_nets": fetch_signed_nets,
        "community_notes": lambda: fetch_community_notes(args.cn_shards),
    }
    for name in targets:
        if name not in fns:
            print(f"unknown dataset {name!r}; choose from {ALL}", file=sys.stderr)
            return 2
        fns[name]()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
