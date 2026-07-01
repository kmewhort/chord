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

from typing import Dict, List, Mapping, Optional, Sequence

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
    """Row-stochastic trust matrix T (§5).

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


def outgoing_diversity_weights(T: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Normalized Shannon entropy of each rater's outgoing trust row, in [floor, 1].

    A rater that spreads approval across many authors has entropy ≈ 1; a rater that
    approves a *single* author (out-degree 1 — the exact fingerprint of a collusion-
    ring puppet) has entropy 0. Used to down-weight the trust such a rater
    *transmits*, so the pooled teleport-floor mass of a K-account ring can no longer
    be harvested by its target (§5, Appendix C.5). ``floor`` keeps a genuine
    low-activity newcomer from being fully muted.
    """
    outdeg = (T > 0).sum(axis=1)
    logT = np.zeros_like(T)
    np.log(T, out=logT, where=T > 0)          # 0·log0 handled by the where-mask
    ent = -(T * logT).sum(axis=1)              # natural-log entropy per row
    mask = outdeg > 1
    denom = np.ones(T.shape[0], dtype=float)
    np.log(outdeg, out=denom, where=mask)      # log only where outdeg>1 (never log 0)
    w = np.where(mask, ent / denom, 0.0)
    return np.clip(floor + (1.0 - floor) * w, 0.0, 1.0)


def eigentrust(
    T: np.ndarray,
    config: ChordConfig,
    transmit_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Damped teleporting eigentrust fixed point (§5).

    lambda <- (1-delta)/n + delta * T^T (w ⊙ lambda), iterated to convergence,
    where ``w`` is the per-rater outgoing-diversity transmit weight (§5 ring
    defense). Returns a probability-like weight vector (non-negative, sums to 1).

    ``transmit_weights`` overrides the default; when None and
    ``config.sybil_out_diversity`` is set, ``w`` is derived from T's row entropy
    (``outgoing_diversity_weights``). Setting ``sybil_out_diversity=False`` recovers
    the plain iteration ``w = 1``.
    """
    n = T.shape[0]
    if n == 0:
        return np.zeros(0)
    delta = config.eigentrust_delta
    teleport = (1.0 - delta) / n
    if transmit_weights is None:
        transmit_weights = (
            outgoing_diversity_weights(T, config.out_diversity_floor)
            if config.sybil_out_diversity else np.ones(n)
        )
    lam = np.full(n, 1.0 / n)
    Tt = T.T
    for _ in range(config.eigentrust_iters):
        new = teleport + delta * (Tt @ (transmit_weights * lam))
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
