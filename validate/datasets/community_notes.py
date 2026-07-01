"""X Community Notes — the deployed bridging-MF keystone (Appendix C.1/C.2, §4/§5).

Community Notes ranks notes on *bridged* helpfulness with a matrix-factorization
model (Wojcik et al. 2022) whose shape is the same as CHORD §4.1: a rater x note
matrix, a note intercept, and low-rank opinion embeddings. Its published
``currentStatus`` (CURRENTLY_RATED_HELPFUL vs NOT) is the output of that deployed
model — a strong *external baseline* to benchmark CHORD's ``B_LCB`` against, which
is exactly what C.2 step 1 calls for ("start here").

Ratings map to signed reactions (§4.1): HELPFUL → +1, SOMEWHAT_HELPFUL → 0,
NOT_HELPFUL → −1; a note is a post authored by its ``noteAuthorParticipantId``.

The full public dump is multi-GB. :func:`build_slice` distills the raw parquet
shards (git-ignored under ``raw/``) into a compact, well-connected, committed slice
(``slice/*.parquet`` via Git LFS): notes with a decided status and enough ratings,
and raters with enough ratings — the sub-population where the bridging estimate is
actually identifiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chord.types import Id, Post, Reaction

from .._common import dataset_dir

NAME = "community_notes"
SLICE_FILES = ("slice/ratings.parquet", "slice/notes.parquet")

# helpfulnessLevel -> signed reaction value (§4.1)
_HELPFULNESS = {"HELPFUL": 1.0, "SOMEWHAT_HELPFUL": 0.0, "NOT_HELPFUL": -1.0}
# currentStatus -> binary helpful label (ground truth); NMR/other dropped
_STATUS_BINARY = {
    "CURRENTLY_RATED_HELPFUL": 1,
    "CURRENTLY_RATED_NOT_HELPFUL": 0,
}


@dataclass
class CNSlice:
    ratings: "object"   # DataFrame: noteId, raterId, value
    notes: "object"     # DataFrame: noteId, author, statusBinary


def _first_col(schema_names, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in schema_names:
            return c
    return None


def _bipartite_kcore(ratings, min_note: int, min_rater: int):
    """Iteratively drop under-connected notes/raters until a stable k-core.

    A random sample of notes would leave the rater x note matrix almost empty
    (raters barely co-rate), so the factorization would be meaningless. The k-core
    keeps the dense, well-connected sub-population where the bridging estimate is
    actually identifiable — the region §4/§6 assume.
    """
    while True:
        n0 = len(ratings)
        nc = ratings["noteId"].value_counts()
        ratings = ratings[ratings["noteId"].isin(nc[nc >= min_note].index)]
        rc = ratings["raterId"].value_counts()
        ratings = ratings[ratings["raterId"].isin(rc[rc >= min_rater].index)]
        if len(ratings) == n0:
            return ratings


def build_slice(
    raw: Path,
    out: Path,
    *,
    max_notes: int = 4000,
    min_ratings_per_note: int = 30,
    min_ratings_per_rater: int = 30,
    seed: int = 0,
) -> None:
    """Distill the raw CN parquet shards into a compact committed slice.

    Reads only the handful of columns CHORD needs (via pyarrow column pushdown),
    keeps notes with a decided status, then extracts the dense bipartite k-core
    (notes with ``>= min_ratings_per_note`` ratings, raters with
    ``>= min_ratings_per_rater`` ratings — mutually, iterated to a fixed point) so
    the rater x note matrix is well-connected. If the core still exceeds
    ``max_notes``, the most-rated notes are kept and the core re-tightened. Writes
    ``slice/notes.parquet`` and ``slice/ratings.parquet``.
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- note status (ground-truth label) ---
    nsh_path = raw / "noteStatusHistory-00000.parquet"
    nsh_names = pq.ParquetFile(nsh_path).schema_arrow.names
    status_col = _first_col(nsh_names, "currentStatus")
    nsh = pq.read_table(nsh_path, columns=["noteId", status_col]).to_pandas()
    nsh = nsh.rename(columns={status_col: "status"})
    nsh["statusBinary"] = nsh["status"].map(_STATUS_BINARY)
    nsh = nsh.dropna(subset=["statusBinary"])
    nsh["statusBinary"] = nsh["statusBinary"].astype(int)

    # --- note authorship ---
    notes_path = raw / "notes-00000.parquet"
    notes_names = pq.ParquetFile(notes_path).schema_arrow.names
    author_col = _first_col(notes_names, "noteAuthorParticipantId", "participantId")
    notes = pq.read_table(notes_path, columns=["noteId", author_col]).to_pandas()
    notes = notes.rename(columns={author_col: "author"})

    note_meta = nsh.merge(notes, on="noteId", how="inner")

    # --- ratings (may span several shards) ---
    rating_shards = sorted(raw.glob("ratings-*.parquet"))
    if not rating_shards:
        raise FileNotFoundError(f"no ratings-*.parquet under {raw}")
    r_names = pq.ParquetFile(rating_shards[0]).schema_arrow.names
    rater_col = _first_col(r_names, "raterParticipantId", "participantId")
    level_col = _first_col(r_names, "helpfulnessLevel")

    keep_note_ids = set(note_meta["noteId"])
    frames = []
    for shard in rating_shards:
        cols = ["noteId", rater_col, level_col]
        t = pq.read_table(shard, columns=cols).to_pandas()
        t = t.rename(columns={rater_col: "raterId", level_col: "level"})
        t = t[t["noteId"].isin(keep_note_ids)]
        t["value"] = t["level"].map(_HELPFULNESS)
        t = t.dropna(subset=["value", "raterId"])
        frames.append(t[["noteId", "raterId", "value"]])
    ratings = pd.concat(frames, ignore_index=True)

    # --- extract the dense, well-connected bipartite k-core ---
    ratings = _bipartite_kcore(ratings, min_ratings_per_note, min_ratings_per_rater)

    # If still huge, keep the most-rated notes and re-tighten the core so the
    # cap does not re-sparsify the matrix.
    if ratings["noteId"].nunique() > max_notes:
        top = ratings["noteId"].value_counts().head(max_notes).index
        ratings = ratings[ratings["noteId"].isin(top)]
        ratings = _bipartite_kcore(ratings, min_ratings_per_note, min_ratings_per_rater)

    note_meta = note_meta[note_meta["noteId"].isin(set(ratings["noteId"]))]

    note_meta[["noteId", "author", "status", "statusBinary"]].to_parquet(
        out / "notes.parquet", index=False)
    ratings.reset_index(drop=True).to_parquet(out / "ratings.parquet", index=False)
    print(f"    slice: {len(note_meta):,} notes, {len(ratings):,} ratings, "
          f"{ratings['raterId'].nunique():,} raters")


def load_slice(base: Optional[Path] = None) -> CNSlice:
    import pandas as pd

    base = base or dataset_dir(NAME)
    ratings = pd.read_parquet(base / "slice" / "ratings.parquet")
    notes = pd.read_parquet(base / "slice" / "notes.parquet")
    return CNSlice(ratings=ratings, notes=notes)


def to_reactions(sl: CNSlice) -> Tuple[List[Reaction], Dict[Id, Post], Dict[Id, int]]:
    """Signed reactions + note posts + ground-truth helpful labels.

    Returns ``(reactions, posts, labels)`` where ``labels`` maps note id ->
    CN ``statusBinary`` (1 = CURRENTLY_RATED_HELPFUL, 0 = NOT).
    """
    author_of = dict(zip(sl.notes["noteId"], sl.notes["author"]))
    labels = {nid: int(b) for nid, b in zip(sl.notes["noteId"], sl.notes["statusBinary"])}
    posts: Dict[Id, Post] = {
        nid: Post(nid, author_id=(author_of.get(nid) or f"anon:{nid}"))
        for nid in labels
    }
    reactions: List[Reaction] = []
    for nid, rid, val in sl.ratings[["noteId", "raterId", "value"]].itertuples(index=False):
        if nid in posts:
            reactions.append(Reaction(rid, nid, float(val)))
    return reactions, posts, labels
