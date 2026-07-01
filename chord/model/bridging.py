"""Bridged support — the per-cluster reconstruction and B_LCB (§4.2).

The scalar intercept ``b_p`` is a linear proxy for diverse approval but diverges
from the real target when clusters sit asymmetrically about the origin (it
rewards a post one cluster loves and another merely tolerates). So we do not
trust it: using the Partition port's clusters ``c`` we reconstruct each cluster's
predicted reception

    r_hat_cp = mu + b_bar_c + b_a(p) + b_p + <x_bar_c, y_p>

and define bridged support by **shrinking each cluster's reception toward the
population mean by how much that cluster was exposed, then aggregating**:

    r_shrunk_cp = grand_p + [ n_cp / (n_cp + n0) ] * (r_hat_cp - grand_p)
    B_LCB(p)    = aggregate_c  r_shrunk_cp          (min | nash | ede)

where grand_p is the mean reception across clusters and n_cp is the (propensity-
corrected) number of cluster-c users actually *exposed* to p (§6). A cluster that
has barely been exposed (n_cp ~ 0) regresses to the mean — its apparent dissent is
sampling noise, not a tested divide — so a post sits near the population mean until
exposed, and only a *well-exposed* disagreeing cluster keeps its low reception and
pulls the score down: **a post is not credited as bridging above the mean until it
has survived contact with the people who would dislike it.** The default ``min``
aggregator is Ethelo's Rawlsian strength; ``nash`` (geometric mean of agree-
probabilities) is Polis's group-informed consensus.

This empirical-Bayes shrinkage (James-Stein / DerSimonian-Laird) replaces an
earlier subtractive penalty ``min_c[r_hat - beta*sigma/sqrt(n+1)]`` that demoted
under-*sampled* clusters and, on real data (Community Notes / Polis), was beaten by
both the scalar b_p and a naive rating mean — it subtracted noise, not risk
(Appendix C.5). Note the deliberate asymmetry (§4.2): the exploration pool samples
*high* uncertainty optimistically (what to audition); B_LCB shrinks *low*-exposure
clusters to the mean and rewards only tested breadth (what to crown).
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
        """B_LCB for a single post (§4.2, shrinkage form — Appendix C.5).

        ``exposure_counts`` maps cluster index -> propensity-corrected *exposure*
        ``n_cp`` (§6). Each cluster's reconstructed reception is shrunk toward the
        population mean with weight ``n_cp/(n_cp+n0)``, then aggregated
        (``config.bridging_aggregator``). A thinly-exposed cluster regresses to the
        mean — its apparent dissent is treated as sampling noise, not tested
        divisiveness — while a *well-exposed* disagreeing cluster keeps its low
        reception and pulls the score down. So a post is credited near the
        population mean until exposed, and only *observed* cross-cluster dissent
        (not a small rating count) lowers it: this replaces the old subtractive
        ``β·σ/√(n+1)`` penalty, which demoted under-sampled clusters and was beaten
        on real data (Community Notes / Polis) by both ``b_p`` and a naive mean.
        """
        y = result.y_post.get(post_id)
        if y is None:
            # Unseen post: no reception estimate at all.
            return float("-inf")
        b_p = result.b_post.get(post_id, 0.0)
        b_a = result.b_author.get(author_id, 0.0)
        mu = result.mu
        cfg = self.config
        K = clusters.n_clusters

        r_hat = np.array([
            mu + clusters.mean_bias[c] + b_a + b_p
            + float(np.dot(clusters.centroids[c], y))
            for c in range(K)
        ])
        grand = float(np.mean(r_hat))          # population (cluster-agnostic) reception
        n_cp = np.array([
            0.0 if exposure_counts is None else float(exposure_counts.get(c, 0.0))
            for c in range(K)
        ])
        w = n_cp / (n_cp + cfg.bridging_shrinkage_n0)   # empirical-Bayes trust in each cluster
        r_shrunk = grand + w * (r_hat - grand)
        return float(self._aggregate(r_shrunk))

    def _aggregate(self, r: np.ndarray) -> float:
        """Aggregate per-cluster (shrunk) receptions into one bridged score (§4.2).

        ``min`` is the Rawlsian worst-cluster default; ``nash`` is the geometric
        mean of per-cluster agree-probabilities (Polis's group-informed consensus);
        ``ede`` is the Atkinson equally-distributed-equivalent with inequality
        aversion ``bridging_ede_eps`` (→ ``min`` as eps→∞). Nash/EDE tracked genuine
        cross-group support better than hard ``min`` on Polis (Appendix C.5).
        """
        mode = self.config.bridging_aggregator
        if mode == "min":
            return float(np.min(r))
        p = np.clip((r + 1.0) / 2.0, 1e-6, 1.0)     # signed reception → agree-prob
        if mode == "nash":
            agg01 = float(np.exp(np.mean(np.log(p))))
        elif mode == "ede":
            eps = self.config.bridging_ede_eps
            if abs(eps - 1.0) < 1e-9:
                agg01 = float(np.exp(np.mean(np.log(p))))
            else:
                agg01 = float(np.mean(p ** (1.0 - eps)) ** (1.0 / (1.0 - eps)))
        else:
            raise ValueError(f"unknown bridging_aggregator {mode!r}")
        return agg01 * 2.0 - 1.0

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
