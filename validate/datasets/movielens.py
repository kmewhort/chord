"""MovieLens 100K — dense source for the semi-synthetic propensity harness (C.3).

GroupLens' ml-100k: 100,000 ratings (1..5) from 943 users on 1682 movies. Unlike
Coat it has no random-exposure block, so we use it the way Appendix C.3 suggests —
"a dense MovieLens slice" as a ground-truth preference matrix — and *impose* a
synthetic MNAR logging policy ourselves (:mod:`validate.synthetic`). That gives us
control of ground truth on genuinely human preference structure, complementing
Coat's real MAR holdout.

``u.data`` is tab-separated: user_id, item_id, rating, timestamp (1-indexed ids).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .._common import dataset_dir

NAME = "movielens"
REQUIRED = ("ml-100k/u.data",)


@dataclass
class MovieLensData:
    ratings: pd.DataFrame        # columns: user, item, rating, ts


def load(base: Optional[Path] = None) -> MovieLensData:
    base = base or dataset_dir(NAME)
    df = pd.read_csv(
        base / "ml-100k" / "u.data",
        sep="\t",
        names=["user", "item", "rating", "ts"],
    )
    return MovieLensData(ratings=df)


def dense_slice(
    data: MovieLensData, n_users: int = 200, n_items: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Densest ``n_users x n_items`` submatrix, by most-active users/items.

    Returns ``(R, observed)`` where ``R`` holds signed ratings in [-1, 1]
    (``(rating - 3) / 2``) and ``observed`` is a boolean mask of which cells are
    real. Selecting the most-active rows/columns yields a submatrix dense enough
    to serve as a ground-truth preference block for the C.3 harness.
    """
    df = data.ratings
    top_users = df["user"].value_counts().head(n_users).index
    sub = df[df["user"].isin(top_users)]
    top_items = sub["item"].value_counts().head(n_items).index
    sub = sub[sub["item"].isin(top_items)]

    uidx = {u: i for i, u in enumerate(sorted(top_users))}
    iidx = {it: j for j, it in enumerate(sorted(top_items))}
    R = np.full((len(uidx), len(iidx)), np.nan)
    for u, it, r in sub[["user", "item", "rating"]].itertuples(index=False):
        R[uidx[u], iidx[it]] = (r - 3.0) / 2.0
    observed = ~np.isnan(R)
    return R, observed
