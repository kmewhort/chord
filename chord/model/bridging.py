"""Bridged support — the per-cluster reconstruction and B_LCB (§4.2).

The scalar intercept ``b_p`` is a linear proxy for diverse approval but diverges
from the real target when clusters sit asymmetrically about the origin (it
rewards a post one cluster loves and another merely tolerates). So we do not
trust it: using the Partition port's clusters ``c`` we reconstruct each cluster's
predicted reception

    r_hat_cp = mu + b_bar_c + b_a(p) + b_p + <x_bar_c, y_p>

and define bridged support as a **tested-breadth lower confidence bound**:

    B_LCB(p) = min_c [ r_hat_cp - beta * sigma / sqrt(n_cp + 1) ]

where n_cp is the (propensity-corrected) number of cluster-c users actually
*exposed* to p. If a cluster that would disagree has not yet been exposed
(n_cp ~ 0), its penalty term is large and B_LCB stays low: **a post is not
credited as bridging until it has survived contact with the people who would
dislike it.** The min-over-clusters form is Ethelo's Rawlsian strength and
Polis's group-aware consensus.

Note the deliberate asymmetry (§4.2): the exploration pool samples *high*
uncertainty optimistically (what to audition); B_LCB uses uncertainty
*pessimistically* (what to crown). Optimism explores; pessimism rewards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

from ..config import ChordConfig
from ..types import Id
from .factorization import FactorizationResult


@dataclass
class ClusterModel:
    """Opinion clusters and their per-cluster statistics (§4.2).

    Produced by the Partition port. ``assignments`` maps user id -> cluster
    index in ``[0, n_clusters)``; the centroid embedding and mean leniency of
    each cluster are derived from the factorization.
    """

    assignments: Dict[Id, int]
    centroids: np.ndarray        # (n_clusters, d) x_bar_c
    mean_bias: np.ndarray        # (n_clusters,) b_bar_c
    n_clusters: int

    @staticmethod
    def from_factorization(
        result: FactorizationResult, assignments: Mapping[Id, int]
    ) -> "ClusterModel":
        # Determine cluster count from the assignments actually present.
        present = [c for c in assignments.values()]
        n_clusters = (max(present) + 1) if present else 1
        d = result.d
        sums = np.zeros((n_clusters, d))
        bias_sums = np.zeros(n_clusters)
        counts = np.zeros(n_clusters)
        for uid, c in assignments.items():
            x = result.x_user.get(uid)
            if x is None:
                continue
            sums[c] += x
            bias_sums[c] += result.b_user.get(uid, 0.0)
            counts[c] += 1
        counts_safe = np.where(counts > 0, counts, 1.0)
        centroids = sums / counts_safe[:, None]
        mean_bias = bias_sums / counts_safe
        return ClusterModel(
            assignments=dict(assignments),
            centroids=centroids,
            mean_bias=mean_bias,
            n_clusters=n_clusters,
        )


@dataclass
class BridgingScores:
    """Output of the bridging scorer for a set of posts."""

    b_lcb: Dict[Id, float] = field(default_factory=dict)        # tested support
    b_scalar: Dict[Id, float] = field(default_factory=dict)     # cheap pre-filter b_p
    per_cluster: Dict[Id, np.ndarray] = field(default_factory=dict)  # r_hat_cp


class BridgingScorer:
    """Computes B_LCB (§4.2) from a factorization and a cluster model."""

    def __init__(self, config: ChordConfig):
        self.config = config

    def score_post(
        self,
        post_id: Id,
        author_id: Id,
        result: FactorizationResult,
        clusters: ClusterModel,
        exposure_counts: Optional[Mapping[int, float]] = None,
    ) -> float:
        """B_LCB for a single post.

        ``exposure_counts`` maps cluster index -> propensity-corrected n_cp. A
        cluster missing from the mapping is treated as n_cp = 0 (never exposed),
        which maximizes its pessimism penalty.
        """
        y = result.y_post.get(post_id)
        if y is None:
            # Unseen post: no tested support yet.
            return float("-inf")
        b_p = result.b_post.get(post_id, 0.0)
        b_a = result.b_author.get(author_id, 0.0)
        mu = result.mu
        cfg = self.config

        best = np.inf
        for c in range(clusters.n_clusters):
            r_hat = (
                mu
                + clusters.mean_bias[c]
                + b_a
                + b_p
                + float(np.dot(clusters.centroids[c], y))
            )
            n_cp = 0.0 if exposure_counts is None else float(exposure_counts.get(c, 0.0))
            penalty = cfg.lcb_beta * cfg.lcb_sigma / np.sqrt(n_cp + 1.0)
            lcb_c = r_hat - penalty
            if lcb_c < best:
                best = lcb_c
        return float(best)

    def score(
        self,
        result: FactorizationResult,
        clusters: ClusterModel,
        post_authors: Mapping[Id, Id],
        exposure_counts: Optional[Mapping[Id, Mapping[int, float]]] = None,
    ) -> BridgingScores:
        """Score every post in the factorization."""
        out = BridgingScores()
        for pid in result.y_post:
            author = post_authors.get(pid)
            ec = None if exposure_counts is None else exposure_counts.get(pid)
            out.b_lcb[pid] = self.score_post(pid, author, result, clusters, ec)
            out.b_scalar[pid] = result.b_post.get(pid, 0.0)
            # cache the per-cluster receptions for diagnostics / monitoring
            y = result.y_post[pid]
            b_p = result.b_post.get(pid, 0.0)
            b_a = result.b_author.get(author, 0.0)
            rc = np.array([
                result.mu + clusters.mean_bias[c] + b_a + b_p
                + float(np.dot(clusters.centroids[c], y))
                for c in range(clusters.n_clusters)
            ])
            out.per_cluster[pid] = rc
        return out
