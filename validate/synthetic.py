"""Impose a synthetic MNAR logging policy on a real ground-truth matrix (C.3).

This is the semi-synthetic harness of :mod:`chord.eval.mnar_harness`, generalized
to run on an arbitrary dense preference matrix ``R`` (e.g. a MovieLens slice)
instead of a generated bipolar world. The point of C.3 is to *keep ground truth*
while inducing exactly the confound §6.1 warns about: items are over-exposed to the
users predicted to like them (in-group alignment), so what is logged is confounded
with preference — except for a small ``epsilon_anchor`` fraction exposed uniformly
at random (the §6.2 identifiability anchor) and a held-out MAR block for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from chord.types import Exposure, ExposureSource, Id, Post, Reaction


@dataclass
class MnarSplit:
    reactions: List[Reaction]                 # observed under the logging policy
    posts: Dict[Id, Post]
    exposures: List[Exposure]                 # carry source + propensity
    true_pi: Dict[Tuple[Id, Id], float]       # (user, post) -> logging propensity
    holdout: List[Tuple[Id, Id, float]]       # MAR ground-truth (user, post, signed rating)
    polarity_user: np.ndarray                 # recovered 1-D user polarity
    polarity_item: np.ndarray                 # recovered 1-D item polarity


def _polarity_axis(R: np.ndarray, observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rank-1 SVD of the mean-imputed, doubly-centered matrix → opinion axis."""
    filled = np.where(observed, R, np.nan)
    col_mean = np.nanmean(filled, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    filled = np.where(observed, R, col_mean[None, :])
    filled = filled - filled.mean(axis=0, keepdims=True)
    filled = filled - filled.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(filled, full_matrices=False)
    return U[:, 0], Vt[0, :]


def induce_mnar(
    R: np.ndarray,
    observed: np.ndarray,
    *,
    ingroup_bias: float = 3.0,
    exposure_rate: float = 0.5,
    epsilon_anchor: float = 0.1,
    holdout_rate: float = 0.25,
    seed: int = 0,
) -> MnarSplit:
    """Split observed cells into an MNAR training log and a MAR holdout.

    Parameters mirror :func:`chord.eval.mnar_harness.logging_policy_exposures`.
    Only *observed* cells participate (we cannot know unrated ground truth); the
    MNAR policy sub-samples them in proportion to predicted in-group alignment.
    """
    rng = np.random.default_rng(seed)
    upol, ipol = _polarity_axis(R, observed)
    # normalize so alignment ~ O(1)
    align = np.outer(upol, ipol)
    if align.std() > 0:
        align = align / align.std()

    logit = ingroup_bias * align
    p_org = 1.0 / (1.0 + np.exp(-logit))
    p_org *= exposure_rate / max(p_org.mean(), 1e-9)
    p_org = np.clip(p_org, 0.0, 0.98)

    reactions: List[Reaction] = []
    exposures: List[Exposure] = []
    true_pi: Dict[Tuple[Id, Id], float] = {}
    holdout: List[Tuple[Id, Id, float]] = []
    posts: Dict[Id, Post] = {}

    n_users, n_items = R.shape
    for j in range(n_items):
        posts[f"i{j}"] = Post(f"i{j}", author_id=f"a{j}")

    rows, cols = np.nonzero(observed)
    for u, j in zip(rows.tolist(), cols.tolist()):
        pid = f"i{j}"
        uid = f"u{u}"
        rating = float(R[u, j])
        if rng.random() < holdout_rate:
            holdout.append((uid, pid, rating))
            continue
        if rng.random() < epsilon_anchor:
            exposures.append(Exposure(uid, pid, source=ExposureSource.EXPLORATION,
                                      propensity=epsilon_anchor))
            true_pi[(uid, pid)] = epsilon_anchor
            reactions.append(Reaction(uid, pid, rating))
        elif rng.random() < p_org[u, j]:
            pi = float(p_org[u, j])
            exposures.append(Exposure(uid, pid, source=ExposureSource.ORGANIC,
                                      propensity=pi))
            true_pi[(uid, pid)] = pi
            reactions.append(Reaction(uid, pid, rating))

    return MnarSplit(
        reactions=reactions, posts=posts, exposures=exposures, true_pi=true_pi,
        holdout=holdout, polarity_user=upol, polarity_item=ipol,
    )
