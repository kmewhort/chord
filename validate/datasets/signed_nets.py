"""SNAP Wikipedia RfA signed votes — trust / credibility propagation (C.1, §5).

West et al. 2014, "Exploiting Social Network Structure for Person-to-Person
Sentiment Analysis" (snap.stanford.edu/data/wiki-RfA.html). Requests-for-
adminship: editors cast **signed** votes (support +1 / neutral 0 / oppose -1) on
each other's candidacies. Signed per-user votes are exactly what §5's EigenTrust
consumes, and are rare in public data.

We model each candidacy as a post authored by the *target*; each vote is a signed
reaction by the *source*. CHORD then fits the opinion embedding, builds the
cross-divide trust matrix, and computes the rater-influence distribution
``lambda`` — trust flows toward candidates approved by raters who are far away in
opinion space (cross-divide support), and fresh/isolated accounts stay near the
teleport floor (Sybil starvation, §5).

Raw file format: blank-line-separated records of ``SRC:``, ``TGT:``, ``VOT:``,
``RES:``, ``YEA:``, ``DAT:``, ``TXT:`` lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chord.types import Id, Post, Reaction

from .._common import dataset_dir

NAME = "signed_nets"
REQUIRED = ("wiki-RfA.txt.gz",)


@dataclass
class SignedVote:
    src: str
    tgt: str
    vote: int
    result: int
    year: str


def _iter_records(path: Path):
    import gzip

    rec: Dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if rec:
                    yield rec
                    rec = {}
                continue
            key, _, val = line.partition(":")
            rec[key] = val
        if rec:
            yield rec


def load_votes(base: Optional[Path] = None, max_records: Optional[int] = None) -> List[SignedVote]:
    base = base or dataset_dir(NAME)
    path = base / "wiki-RfA.txt.gz"
    votes: List[SignedVote] = []
    for rec in _iter_records(path):
        src, tgt = rec.get("SRC", ""), rec.get("TGT", "")
        if not src or not tgt or src == tgt:
            continue
        try:
            vot = int(rec.get("VOT", "0"))
        except ValueError:
            continue
        votes.append(SignedVote(
            src=src, tgt=tgt, vote=vot,
            result=int(rec.get("RES", "0") or 0), year=rec.get("YEA", ""),
        ))
        if max_records is not None and len(votes) >= max_records:
            break
    return votes


def to_reactions(
    votes: List[SignedVote],
    min_src_votes: int = 5,
    min_tgt_votes: int = 15,
    include_neutral: bool = False,
) -> Tuple[List[Reaction], Dict[Id, Post], List[Id]]:
    """Signed reactions + candidacy posts, filtered to active editors.

    Each ``(target, year)`` is one candidacy post authored by the target; each
    non-neutral vote becomes a signed reaction. Activity filters drop editors too
    sparse to embed. Returns ``(reactions, posts, users)`` where ``users`` is the
    union of every source and target that survived filtering — the row set for the
    trust matrix.
    """
    from collections import Counter

    src_counts = Counter(v.src for v in votes)
    tgt_counts = Counter(v.tgt for v in votes)
    keep_src = {s for s, c in src_counts.items() if c >= min_src_votes}
    keep_tgt = {t for t, c in tgt_counts.items() if c >= min_tgt_votes}

    reactions: List[Reaction] = []
    posts: Dict[Id, Post] = {}
    users: set = set()
    for v in votes:
        if v.src not in keep_src or v.tgt not in keep_tgt:
            continue
        if v.vote == 0 and not include_neutral:
            continue
        pid = f"{v.tgt}@{v.year}"
        if pid not in posts:
            posts[pid] = Post(pid, author_id=v.tgt)
        reactions.append(Reaction(v.src, pid, float(v.vote)))
        users.add(v.src)
        users.add(v.tgt)
    return reactions, posts, sorted(users)
