"""Thin helpers wiring the CHORD core onto dataset reactions.

Keeps the test files focused on *claims* rather than plumbing. Everything here is
a straight call into :mod:`chord` — no dataset-specific logic.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from chord.config import ChordConfig
from chord.model import (
    BridgingScorer,
    ClusterModel,
    FactorizationResult,
    MatrixFactorization,
    cluster_reception,
    fit_divisiveness,
)
from chord.model.spectral import spectral_partition
from chord.types import Id, Post, Reaction


def fit(
    reactions: Sequence[Reaction],
    posts: Dict[Id, Post],
    config: ChordConfig,
    weights: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> FactorizationResult:
    return MatrixFactorization(config, seed=seed).fit(reactions, posts, weights)


def predict(result: FactorizationResult, uid: Id, pid: Id, author_id: Id) -> float:
    """Reconstruct r_hat(u, p) = mu + b_u + b_a + b_p + <x_u, y_p> (§4.1).

    Missing entities fall back to their neutral value (0 / mu), so an unseen user
    or item yields the population mean.
    """
    x = result.x_user.get(uid)
    y = result.y_post.get(pid)
    dot = float(np.dot(x, y)) if (x is not None and y is not None) else 0.0
    return (
        result.mu
        + result.b_user.get(uid, 0.0)
        + result.b_author.get(author_id, 0.0)
        + result.b_post.get(pid, 0.0)
        + dot
    )


def cluster(
    reactions: Sequence[Reaction], result: FactorizationResult, config: ChordConfig,
    n_clusters: Optional[int] = None, seed: int = 0,
) -> ClusterModel:
    """Deterministic spectral opinion clusters from the reaction data (§4.2)."""
    k = n_clusters or config.n_clusters
    part = spectral_partition(reactions, list(result.x_user.keys()), k)
    clusters = ClusterModel.from_factorization(result, part.assignments)
    clusters.opinion_coord = part.coord
    return clusters


def bridging(
    reactions: Sequence[Reaction],
    result: FactorizationResult,
    clusters: ClusterModel,
    post_authors: Mapping[Id, Id],
    config: ChordConfig,
    weights: Optional[Sequence[float]] = None,
    reception_caps: Optional[Mapping[Id, float]] = None,
):
    """B_LCB (§4.2) for every post — empirical IPW-shrunk per-cluster reception."""
    if weights is None:
        weights = np.ones(len(reactions))
    reception = cluster_reception(reactions, weights, clusters)
    return BridgingScorer(config).score(
        result, clusters, post_authors, reception, reception_caps)


def divisiveness_of(
    result: FactorizationResult, config: ChordConfig,
) -> Dict[Id, float]:
    """D(p) = y_p^T A y_p per post (§4.1), A = I when no affective signal."""
    model = fit_divisiveness(result, config)
    return {pid: model.divisiveness(y) for pid, y in result.y_post.items()}
