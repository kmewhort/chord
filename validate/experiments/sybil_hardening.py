"""Candidate fixes for the §5 Sybil-ring finding, evaluated on Wikipedia RfA.

The finding (validate/test_signed_nets_eigentrust.py): a ring of K fresh accounts
each casting one approval at a target lifts that target's influence percentile
linearly in K, because the uniform teleport floor `(1-δ)/n` gives every puppet
baseline mass that the target harvests via the row-stochastic transport `Tᵀλ`.

The research (both agents) converged on three cheap, numpy-only iteration tweaks,
each a near-one-liner on `λ ← (1-δ)/n + δ·Tᵀλ`:

* **out-diversity** — weight each rater's *transmitted* mass by the normalized
  entropy of its outgoing trust row. A rater that only ever approves one author
  (every ring puppet) has out-entropy 0 and transmits nothing.
  (Ziegler & Lausen 2005 conserved-out-budget; Guha et al. 2004.)
* **per-author clip** — cap any one author's harvested inflow `min(Tᵀλ, c)`, so a
  ring's lift is bounded by the constant c regardless of K.
  (Huber 1964 bounded influence; Dwork et al. 2006 sensitivity.)
* **seeded teleport** — replace the uniform floor with a distribution `p_seed`
  supported on structurally-trusted accounts (high cross-cluster out-diversity);
  fresh puppets get zero restart mass to launder.
  (Kamvar et al. 2003 pre-trusted peers; Gyöngyi et al. 2004 TrustRank;
  Cheng & Friedman 2005/2006 — symmetric/uniform teleport is provably not
  sybilproof, an asymmetric seeded restart is the escape.)

This module reuses `chord.rater.eigentrust.build_trust_matrix` to construct the
row-stochastic T over the augmented (real + ring) user set, then runs each
candidate λ-iteration on that same T so the comparison is apples-to-apples.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from chord.config import ChordConfig
from chord.rater.eigentrust import build_trust_matrix
from chord.types import Post, Reaction


def row_entropy_weights(T: np.ndarray) -> np.ndarray:
    """Normalized Shannon entropy of each rater's outgoing trust row, in [0,1].

    Out-degree ≤ 1 → weight 0 (a single-target rater transmits nothing). This is
    the exact fingerprint of a ring puppet.
    """
    outdeg = (T > 0).sum(axis=1)
    logT = np.zeros_like(T)
    np.log(T, out=logT, where=T > 0)          # 0·log0 handled by the where-mask
    ent = -(T * logT).sum(axis=1)              # natural-log entropy per row
    mask = outdeg > 1
    denom = np.ones(T.shape[0], dtype=float)
    np.log(outdeg, out=denom, where=mask)     # log only where outdeg>1 (no log 0)
    w = np.where(mask, ent / denom, 0.0)
    return np.clip(w, 0.0, 1.0)


def eigentrust_variant(
    T: np.ndarray,
    cfg: ChordConfig,
    *,
    teleport: Optional[np.ndarray] = None,
    transmit_w: Optional[np.ndarray] = None,
    clip: Optional[float] = None,
) -> np.ndarray:
    """Damped teleporting eigentrust with optional transmit-weighting / clip / seed.

    ``teleport`` defaults to the uniform floor (the shipped behaviour). Passing a
    seed distribution, a per-rater transmit weight, or a per-author clip yields the
    three candidate defenses; all reduce to the baseline when their arg is None.
    """
    n = T.shape[0]
    delta = cfg.eigentrust_delta
    if teleport is None:
        teleport = np.full(n, (1.0 - delta) / n)
    else:
        teleport = (1.0 - delta) * teleport
    Tt = T.T
    lam = np.full(n, 1.0 / n)
    for _ in range(cfg.eigentrust_iters):
        x = lam if transmit_w is None else transmit_w * lam
        flow = Tt @ x
        if clip is not None:
            flow = np.minimum(flow, clip)
        new = teleport + delta * flow
        s = new.sum()
        if s > 0:
            new = new / s
        if np.max(np.abs(new - lam)) < cfg.eigentrust_tol:
            lam = new
            break
        lam = new
    return lam


def seed_distribution(T_real: np.ndarray, n_total: int, frac: float = 0.05) -> np.ndarray:
    """Uniform distribution over structurally-trusted real accounts.

    Seeds = the real users with the highest (out-degree × normalized out-entropy):
    established, cross-divide approvers. Chosen from graph structure only (no
    external identity), then bound to the identity port in a real deployment (§12).
    """
    w = row_entropy_weights(T_real) * (T_real > 0).sum(axis=1)
    n_real = T_real.shape[0]
    k = max(1, int(frac * n_real))
    seeds = np.argsort(-w)[:k]
    p = np.zeros(n_total)
    p[seeds] = 1.0 / len(seeds)
    return p


def build_ring(reactions, posts, users, target: str, K: int):
    """Augment the graph with a K-puppet ring all approving one fresh target."""
    tgt_post = f"{target}@ring"
    aug_users = list(users) + [target] + [f"sybil{i}" for i in range(K)]
    aug_posts = dict(posts)
    aug_posts[tgt_post] = Post(tgt_post, author_id=target)
    aug_reactions = list(reactions) + [Reaction(f"sybil{i}", tgt_post, 1.0) for i in range(K)]
    return aug_reactions, aug_posts, aug_users


def percentile_of(value, population) -> float:
    population = np.asarray(population)
    return 100.0 * float(np.mean(population < value))


def sweep(reactions, posts, users, result, cfg, Ks=(5, 20, 50, 100)) -> Dict[str, Dict[int, float]]:
    """Target-percentile-vs-K for baseline and each candidate defense."""
    target = "SYBIL_TARGET"
    n_real = len(users)
    variants = ["baseline", "out-diversity", "per-author-clip", "seeded", "out-div+clip"]
    out: Dict[str, Dict[int, float]] = {v: {} for v in variants}

    for K in Ks:
        aug_reactions, aug_posts, aug_users = build_ring(reactions, posts, users, target, K)
        T = build_trust_matrix(aug_reactions, aug_posts, result, aug_users)
        tgt_idx = n_real  # target sits right after the real users
        real_slice = slice(0, n_real)

        w = row_entropy_weights(T)
        # clip threshold: the 99.9th percentile of real authors' baseline inflow,
        # so no legitimate author is capped but unbounded ring growth is.
        base_lam = eigentrust_variant(T, cfg)
        base_flow = (T.T @ base_lam)[real_slice]
        clip_c = float(np.quantile(base_flow, 0.999))
        p_seed = seed_distribution(T[real_slice, :][:, real_slice], T.shape[0])

        runs = {
            "baseline": eigentrust_variant(T, cfg),
            "out-diversity": eigentrust_variant(T, cfg, transmit_w=w),
            "per-author-clip": eigentrust_variant(T, cfg, clip=clip_c),
            "seeded": eigentrust_variant(T, cfg, teleport=p_seed),
            "out-div+clip": eigentrust_variant(T, cfg, transmit_w=w, clip=clip_c),
        }
        for name, lam in runs.items():
            out[name][K] = percentile_of(lam[tgt_idx], lam[real_slice])
    return out
