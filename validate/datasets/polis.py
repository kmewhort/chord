"""Polis open conversation data — real deliberation / clusters (Appendix C.1, §4).

The Computational Democracy Project publishes exported Polis conversations
(https://github.com/compdemocracy/openData) as CSVs. Each conversation is a
participant x comment matrix of agree(+1) / disagree(-1) / pass(0) votes, plus
Polis's own *validated opinion groups* (``group-id``) computed by its PCA +
k-means pipeline. That makes it the natural real-world check for CHORD's §4.2
cluster reconstruction and B_LCB bridged-support ranking:

* ``votes.csv``            — timestamp, datetime, comment-id, voter-id, vote
* ``comments.csv``         — comment-id, author-id, agrees, disagrees, moderated, body
* ``participants-votes.csv`` — per-participant ``group-id`` (the ground-truth cluster)

We map an agree/disagree vote to a signed CHORD reaction on the comment (a "post"
authored by ``author-id``). Pass votes carry no directional signal and are dropped
from the factorization, matching Polis's own treatment of pass as non-informative
for the opinion embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from chord.types import Id, Post, Reaction

from .._common import dataset_dir

NAME = "polis"

# A few conversations of increasing size. The fetcher pulls these by default;
# tests iterate over whichever are present.
CONVERSATIONS = (
    "football-concussions",
    "brexit-consensus",
    "vtaiwan.uberx",
)


@dataclass
class PolisConversation:
    slug: str
    votes: pd.DataFrame          # columns: comment-id, voter-id, vote
    comments: pd.DataFrame       # columns: comment-id, author-id, ...
    groups: Dict[Id, int]        # participant id -> Polis group-id (ground truth)
    n_groups: int


def available(base: Optional[Path] = None) -> List[str]:
    base = base or dataset_dir(NAME)
    return [c for c in CONVERSATIONS if (base / c / "votes.csv").exists()]


def load_conversation(slug: str, base: Optional[Path] = None) -> PolisConversation:
    base = base or dataset_dir(NAME)
    root = base / slug
    votes = pd.read_csv(root / "votes.csv")
    comments = pd.read_csv(root / "comments.csv")
    pv = pd.read_csv(root / "participants-votes.csv")

    # group-id is the validated cluster; participants with no group are dropped.
    groups: Dict[Id, int] = {}
    if "group-id" in pv.columns:
        g = pv[["participant", "group-id"]].dropna().to_numpy()
        groups = {int(part): int(grp) for part, grp in g}
    n_groups = (max(groups.values()) + 1) if groups else 0
    return PolisConversation(
        slug=slug, votes=votes, comments=comments, groups=groups, n_groups=n_groups,
    )


def to_reactions(
    conv: PolisConversation, include_pass: bool = False,
) -> tuple[List[Reaction], Dict[Id, Post]]:
    """Signed reactions (agree=+1 / disagree=-1) + comment posts.

    Comment ids are ``c{comment-id}``, participant ids ``p{voter-id}``.
    """
    posts: Dict[Id, Post] = {}
    cc = _col(conv.comments, "comment-id")
    ca = _col(conv.comments, "author-id")
    for row in conv.comments.itertuples(index=False):
        pid = f"c{int(row[cc])}"
        author = row[ca]
        posts[pid] = Post(pid, author_id=f"auth{int(author)}" if pd.notna(author) else "auth?")

    reactions: List[Reaction] = []
    vv = conv.votes
    ci = _col(vv, "comment-id")
    vi = _col(vv, "voter-id")
    val = _col(vv, "vote")
    for row in vv.itertuples(index=False):
        vote = row[val]
        if pd.isna(vote):
            continue
        vote = float(vote)
        if vote == 0 and not include_pass:
            continue
        pid = f"c{int(row[ci])}"
        if pid not in posts:
            posts[pid] = Post(pid, author_id="auth?")
        reactions.append(Reaction(f"p{int(row[vi])}", pid, vote))
    return reactions, posts


def group_split(conv: PolisConversation) -> Dict[int, np.ndarray]:
    """Per-comment mean vote within each Polis group.

    Returns comment-id -> array of length ``n_groups`` giving the average vote
    (in [-1, 1]) of each group on that comment. A genuinely *bridging* comment
    has a high value in **every** group; a divisive one splits them.
    """
    vv = conv.votes
    ci = _col(vv, "comment-id")
    vi = _col(vv, "voter-id")
    val = _col(vv, "vote")
    out: Dict[int, np.ndarray] = {}
    if conv.n_groups == 0:
        return out
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, np.ndarray] = {}
    for row in vv.itertuples(index=False):
        vote = row[val]
        if pd.isna(vote):
            continue
        part = int(row[vi])
        g = conv.groups.get(part)
        if g is None:
            continue
        cid = int(row[ci])
        if cid not in sums:
            sums[cid] = np.zeros(conv.n_groups)
            counts[cid] = np.zeros(conv.n_groups)
        sums[cid][g] += float(vote)
        counts[cid][g] += 1.0
    for cid in sums:
        c = np.where(counts[cid] > 0, counts[cid], np.nan)
        out[cid] = sums[cid] / c
    return out


def _col(df: pd.DataFrame, name: str) -> int:
    """Positional index of a column, tolerant of hyphen/underscore variants."""
    cols = list(df.columns)
    for cand in (name, name.replace("-", "_"), name.replace("-", ".")):
        if cand in cols:
            return cols.index(cand)
    raise KeyError(f"{name!r} not in {cols}")
