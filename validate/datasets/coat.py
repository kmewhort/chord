"""Coat shopping dataset — the propensity / MNAR keystone (Appendix C.1, §6).

Schnabel et al. 2016, "Recommendations as Treatments". 290 users x 300 items of
integer ratings 1..5 (0 = unobserved), in two blocks:

* ``train.ascii`` — ratings on **self-selected** items. Missing-not-at-random:
  users rated what they chose to look at, so what is observed is confounded with
  preference — exactly the §6.1 in-group-over-exposure confound.
* ``test.ascii`` — ratings on **uniformly random** items. Missing-*completely*-at-
  random: the unconfounded holdout, the "randomly-exposed anchor" of §6.2 and the
  standard ground truth for unbiased NDCG/AUC (C.3).
* ``propensities.ascii`` — the paper's learned P(observed) per cell; the true
  logging propensity for IPW.

This is the C.3 semi-synthetic harness run on *real* data: fit on the MNAR train
block with and without IPW correction, score against the MAR test block.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from chord.types import Id, Post, Reaction

from .._common import dataset_dir

NAME = "coat"
REQUIRED = ("coat/train.ascii", "coat/test.ascii", "coat/propensities.ascii")


@dataclass
class CoatData:
    train: np.ndarray          # (290, 300) ratings 1..5, 0 = missing (MNAR)
    test: np.ndarray           # (290, 300) ratings 1..5, 0 = missing (MAR)
    propensities: np.ndarray   # (290, 300) P(observed) for the train block
    n_users: int
    n_items: int


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def load(base: Optional[Path] = None) -> CoatData:
    base = base or dataset_dir(NAME)
    root = base / "coat"
    train = _read_matrix(root / "train.ascii")
    test = _read_matrix(root / "test.ascii")
    prop = _read_matrix(root / "propensities.ascii")
    return CoatData(
        train=train, test=test, propensities=prop,
        n_users=train.shape[0], n_items=train.shape[1],
    )


def signed(rating: float) -> float:
    """Map a 1..5 star rating onto CHORD's signed [-1, 1] reaction scale (§4.1)."""
    return (rating - 3.0) / 2.0


def to_reactions(matrix: np.ndarray) -> tuple[List[Reaction], Dict[Id, Post]]:
    """Build signed reactions + posts from a rating matrix (0 entries dropped).

    User ids are ``u{i}``, item/post ids ``i{j}``, each item authored by a
    distinct synthetic author ``a{j}`` (Coat has no author structure — one item
    per author keeps the author-bias term inert).
    """
    reactions: List[Reaction] = []
    posts: Dict[Id, Post] = {}
    users, items = np.nonzero(matrix)
    for u, j in zip(users.tolist(), items.tolist()):
        pid = f"i{j}"
        if pid not in posts:
            posts[pid] = Post(pid, author_id=f"a{j}")
        reactions.append(Reaction(f"u{u}", pid, signed(float(matrix[u, j]))))
    return reactions, posts


def propensity_lookup(data: CoatData) -> Dict[tuple, float]:
    """(user_id, post_id) -> true logging propensity for the train block."""
    out: Dict[tuple, float] = {}
    users, items = np.nonzero(data.train)
    for u, j in zip(users.tolist(), items.tolist()):
        out[(f"u{u}", f"i{j}")] = float(data.propensities[u, j])
    return out
