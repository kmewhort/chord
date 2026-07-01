"""Recursive cross-divide credibility via damped, teleporting eigentrust (§5).

The modern form of YourView's "you are credible if trusted by credible people who
disagree with you", computed on the learned opinion geometry:

    lambda <- (1-delta)/n * 1 + delta * T^T lambda
    T_vu   ∝ sum_{p: a(p)=u} [r_vp]_+ * dist(x_v, x_u)

The teleport floor (delta < 1) makes this a contraction with a unique fixed
point, floors every rater's weight (no one is zeroed), and starves Sybils (fresh
accounts have no incoming cross-divide trust). Trust flows toward authors whose
posts earn *positive* reactions from raters who are *far away* in opinion space —
cross-divide approval, not in-group applause.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

from ..config import ChordConfig
from ..types import Id, Post, Reaction
from ..model.factorization import FactorizationResult


def build_trust_matrix(
    reactions: Sequence[Reaction],
    posts: Mapping[Id, Post],
    result: FactorizationResult,
    users: Sequence[Id],
) -> np.ndarray:
    """Column-stochastic trust matrix T (§5).

    ``T[v, u]`` accumulates rater v's positive reactions to author u's posts,
    weighted by the opinion-space distance between them (cross-divide trust
    counts more). Each **rater's outgoing** trust (row) is normalized to sum to 1,
    so a rater distributes a fixed unit of trust among the authors it approves —
    classic EigenTrust. The iteration ``T^T lambda`` then transports each rater's
    weight to the authors it trusts, so an honest author boosted by *many*
    cross-divide raters accrues far more than a Sybil boosted by one colluder.
    """
    idx = {uid: i for i, uid in enumerate(users)}
    n = len(users)
    T = np.zeros((n, n))
    for rx in reactions:
        if rx.value <= 0:
            continue  # only positive (approving) reactions build trust
        post = posts.get(rx.post_id)
        if post is None:
            continue
        v = idx.get(rx.user_id)
        u = idx.get(post.author_id)
        if v is None or u is None or v == u:
            continue
        xv = result.x_user.get(rx.user_id)
        xu = result.x_user.get(post.author_id)
        dist = 1.0
        if xv is not None and xu is not None:
            dist = float(np.linalg.norm(xv - xu))
        T[v, u] += rx.value * dist
    # row-normalize (each row = rater v's outgoing trust, sums to 1)
    row = T.sum(axis=1, keepdims=True)
    row_safe = np.where(row > 0, row, 1.0)
    T = T / row_safe
    return T


def eigentrust(
    T: np.ndarray,
    config: ChordConfig,
) -> np.ndarray:
    """Damped teleporting eigentrust fixed point (§5).

    lambda <- (1-delta)/n + delta * T^T lambda, iterated to convergence.
    Returns a probability-like weight vector (non-negative, sums to 1).
    """
    n = T.shape[0]
    if n == 0:
        return np.zeros(0)
    delta = config.eigentrust_delta
    teleport = (1.0 - delta) / n
    lam = np.full(n, 1.0 / n)
    Tt = T.T
    for _ in range(config.eigentrust_iters):
        new = teleport + delta * (Tt @ lam)
        s = new.sum()
        if s > 0:
            new = new / s
        if np.max(np.abs(new - lam)) < config.eigentrust_tol:
            lam = new
            break
        lam = new
    return lam


def compute_lambda(
    reactions: Sequence[Reaction],
    posts: Mapping[Id, Post],
    result: FactorizationResult,
    users: Sequence[Id],
    config: ChordConfig,
) -> Dict[Id, float]:
    """End-to-end rater influence lambda_u keyed by user id (§5)."""
    users = list(users)
    T = build_trust_matrix(reactions, posts, result, users)
    lam = eigentrust(T, config)
    return {uid: float(lam[i]) for i, uid in enumerate(users)}
