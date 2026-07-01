"""Deterministic opinion clustering (§4.2) — reproducible by construction.

The old path clustered k-means on the ALS user embedding ``x_u``. Because the biased
MF is bilinear/non-convex and was initialized randomly *by index*, the same data in a
different order landed in a different local optimum, so the clusters (and hence B_LCB)
were only ~0.78 ARI / ~0.73 Spearman reproducible (see
``tests/test_properties.py::test_permutation_and_order_invariance`` and
``VALIDATION_FINDINGS.md`` F2/F3).

This computes the opinion clusters directly from the reaction data via a *canonical*
truncated SVD of the mean-centred user×post matrix (the deterministic spectral
embedding — the same object Polis clusters, and the top singular directions of the
convex nuclear-norm completion; see the research note), then a deterministic k-means.
The embedding depends only on the data, not the input order/labelling or any RNG, so
the partition is reproducible. Sign/label ambiguity is harmless downstream because
B_LCB aggregates symmetrically over clusters (``min``/``nash``/``ede``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from ..types import Id, Reaction


@dataclass
class SpectralPartition:
    """Deterministic opinion clustering plus the continuous top-axis coordinate.

    ``coord[u]`` is user u's position on the primary opinion axis (the top canonical
    singular vector). The collusion gate uses this *continuous* spread rather than the
    discrete ``assignments`` because the 2-way split can be degenerate on weakly-divided
    data (e.g. the dense Community-Notes core), while the coordinate keeps full
    resolution — a camouflaged ring spans the axis, a coherent fanbase is concentrated.
    """

    assignments: Dict[Id, int]
    coord: Dict[Id, float] = field(default_factory=dict)


def spectral_opinion_clusters(
    reactions: Sequence[Reaction], users: Sequence[Id], n_clusters: int
) -> Dict[Id, int]:
    """Assign each user to an opinion cluster in ``[0, n_clusters)``, deterministically."""
    return spectral_partition(reactions, users, n_clusters).assignments


def spectral_partition(
    reactions: Sequence[Reaction], users: Sequence[Id], n_clusters: int
) -> SpectralPartition:
    """Deterministic opinion partition + continuous opinion-axis coordinate."""
    users = list(users)
    nu = len(users)
    if n_clusters <= 1 or nu <= n_clusters:
        # trivial/degenerate: a stable, order-invariant fallback keyed on the id itself
        return SpectralPartition(
            {u: (i % max(1, n_clusters)) for i, u in enumerate(_stable(users))},
            {u: 0.0 for u in users})

    posts = _stable({r.post_id for r in reactions})
    ui = {u: i for i, u in enumerate(users)}
    pj = {p: j for j, p in enumerate(posts)}
    npost = len(posts)
    if npost < 2:
        return SpectralPartition({u: 0 for u in users}, {u: 0.0 for u in users})

    # reaction matrix, centred **per column** (subtract each post's observed mean, ~ b_p)
    # so the top singular direction is the *opinion* axis, not post popularity; missing
    # entries are 0 (= at the post mean). Global centering instead let comment-popularity
    # dominate and matched real Polis groups poorly (ARI ~0.06).
    ssum = np.zeros((nu, npost))
    cnt = np.zeros((nu, npost))
    for r in reactions:
        i, j = ui.get(r.user_id), pj.get(r.post_id)
        if i is None or j is None:
            continue
        ssum[i, j] += r.value
        cnt[i, j] += 1.0
    obs = cnt > 0
    cell = np.where(obs, ssum / np.maximum(cnt, 1.0), 0.0)
    col_obs = obs.sum(axis=0)
    col_mean = np.where(col_obs > 0, cell.sum(axis=0) / np.maximum(col_obs, 1), 0.0)
    M = np.where(obs, cell - col_mean[None, :], 0.0)

    t = int(min(max(1, n_clusters - 1), min(nu, npost) - 1))     # embedding dimension
    X = _canonical_left_singular_vectors(M, t)
    labels = _deterministic_kmeans(X, n_clusters)
    return SpectralPartition(
        {users[i]: int(labels[i]) for i in range(nu)},
        {users[i]: float(X[i, 0]) for i in range(nu)})


def _stable(ids) -> List:
    """A deterministic ordering of opaque ids (sort by string form; ids are hashable)."""
    return sorted(ids, key=lambda x: (str(type(x)), str(x)))


def _canonical_left_singular_vectors(M: np.ndarray, t: int) -> np.ndarray:
    """Top-``t`` left singular vectors of ``M``, sign-canonicalized (order-invariant)."""
    try:
        from scipy.sparse.linalg import svds
        m = min(M.shape)
        v0 = np.ones(m) / np.sqrt(m)                 # fixed start ⇒ deterministic ARPACK
        U, S, _ = svds(M.astype(float), k=t, v0=v0)
        order = np.argsort(-S)                        # svds returns ascending
        U = U[:, order]
    except Exception:
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        U = U[:, :t]
    # canonical sign: flip each axis so its largest-magnitude entry is positive. That
    # entry belongs to a specific user, so the choice is invariant to row order.
    for j in range(U.shape[1]):
        i = int(np.argmax(np.abs(U[:, j])))
        if U[i, j] < 0:
            U[:, j] = -U[:, j]
    return U


def _deterministic_kmeans(X: np.ndarray, k: int, iters: int = 100) -> np.ndarray:
    """Lloyd's k-means with a deterministic, order-invariant init (quantiles of axis 0)."""
    n = len(X)
    order = np.argsort(X[:, 0], kind="stable")
    # seed at *interior* quantiles (i+0.5)/k, not the 0/1 extremes: the endpoints are
    # outliers, which on skewed data (e.g. Community Notes) produced a degenerate
    # singleton cluster. (i+0.5)/k → 0.25, 0.75 for k=2: balanced, robust.
    seeds = [order[int(round(((i + 0.5) / k) * (n - 1)))] for i in range(k)]
    centers = X[seeds].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return labels
