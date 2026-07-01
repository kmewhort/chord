"""Bridged support — the per-cluster reception and B_LCB (§4.2).

The scalar intercept ``b_p`` is a linear proxy for diverse approval but diverges
from the real target when clusters sit asymmetrically about the origin (it rewards a
post one cluster loves and another merely tolerates). So we do not trust it: using
the opinion clusters ``c`` we take each cluster's **empirical reception** — the
IPW-weighted mean of the signed reactions cluster-c members actually gave the post —

    r_emp_cp = ( Σ_{u∈c, u reacted to p} ω_up · r_up ) / ( Σ_{u∈c} ω_up ),   n_cp = Σ_{u∈c} ω_up

and define bridged support by **shrinking each cluster's reception toward a stable
prior by how much evidence that cluster gave, then aggregating**:

    r_shrunk_cp = grand_p + [ n_cp / (n_cp + n0) ] * (r_emp_cp - grand_p)
    B_LCB(p)    = aggregate_c  r_shrunk_cp          (min | nash | ede)

where grand_p = μ + b_a(p) + b_p is the post's author/intercept-adjusted prior (the
*convex, reproducible* part of the factorization) and n_cp is the (propensity-
corrected) evidence weight from cluster c. A cluster with little evidence (n_cp ~ 0)
regresses to the prior — its apparent dissent is sampling noise, not a tested divide
— so a post sits near its prior until rated, and only a *well-observed* disagreeing
cluster keeps its low reception and pulls the score down: **a post is not credited as
bridging above the prior until it has survived contact with the people who would
dislike it.** The default ``min`` aggregator is Ethelo's Rawlsian strength; ``nash``
(geometric mean of agree-probabilities) is Polis's group-informed consensus.

Reception is **empirical**, not the bilinear reconstruction ``<x_bar_c, y_p>`` used
previously: routing it through the non-convex MF embedding imported that embedding's
order-dependent local-optimum non-identifiability into B_LCB (rankings only ~0.73
Spearman-reproducible; VALIDATION_FINDINGS F2/F3). The empirical cluster mean depends
on the embedding only through the *discrete, deterministic* cluster label, so B_LCB is
reproducible — and on real Community Notes data it is also *more* faithful to the
helpful/not-helpful ground truth (AUC 0.998 vs 0.858). This empirical-Bayes shrinkage
(James-Stein / DerSimonian-Laird) also replaces an earlier subtractive penalty
``min_c[r_hat - beta*sigma/sqrt(n+1)]`` that demoted under-*sampled* clusters and, on
real data, was beaten by both ``b_p`` and a naive mean (Appendix C.5). Note the
deliberate asymmetry (§4.2): the exploration pool samples *high* uncertainty
optimistically (what to audition); B_LCB shrinks *low*-evidence clusters to the prior
and rewards only tested breadth (what to crown).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from ..config import ChordConfig
from ..types import Id, Reaction
from .factorization import FactorizationResult


def cluster_reception(
    reactions: Sequence[Reaction], weights: Sequence[float], clusters: "ClusterModel"
) -> Dict[Id, Dict[int, tuple]]:
    """Per (post, cluster): IPW-weighted evidence ``n_cp`` and empirical mean reception
    ``r_emp_cp`` (§4.2), from the observed signed reactions.

    ``weights[i]`` is the IPW weight ω_up of ``reactions[i]`` (§6.2). A cluster with no
    reactions to a post is simply absent (n_cp = 0), so the scorer shrinks it to the
    prior — untested dissent is not counted against the post.
    """
    wsum: Dict[Id, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    rsum: Dict[Id, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for r, w in zip(reactions, weights):
        c = clusters.assignments.get(r.user_id)
        if c is None:
            continue
        wsum[r.post_id][c] += float(w)
        rsum[r.post_id][c] += float(w) * r.value
    out: Dict[Id, Dict[int, tuple]] = {}
    for pid, cmap in wsum.items():
        out[pid] = {c: (n, rsum[pid][c] / n) for c, n in cmap.items() if n > 0}
    return out


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
    opinion_coord: Dict[Id, float] = field(default_factory=dict)  # continuous spectral axis

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

    def _receptions(
        self,
        post_id: Id,
        author_id: Id,
        result: FactorizationResult,
        clusters: ClusterModel,
        reception: Optional[Mapping[int, tuple]],
        reception_cap: Optional[float],
    ):
        """Return (grand_prior, shrunk per-cluster receptions) or (None, None) if unseen.

        ``reception`` maps cluster index -> (n_cp, r_emp_cp): the propensity-corrected
        evidence weight and the IPW-weighted empirical mean of cluster-c's signed
        reactions to the post. Each cluster shrinks toward ``grand = μ + b_a + b_p``
        (the reproducible convex biases) with weight ``n_cp/(n_cp+n0)``.
        """
        if post_id not in result.y_post:
            return None, None
        # Prior is the reproducible global mean μ, not μ+b_a+b_p: the MF biases carry the
        # embedding's order-dependence (they drop reproducibility to ~0.85), and on real
        # data the empirical cluster means already carry the post/author signal (grand=μ
        # scored AUC 0.9996). Thin clusters regress to the population mean — "not credited
        # until tested." (Trade-off: this is more lenient on untested one-sided content;
        # the author budget §8, not B_LCB, is what bounds a firehose's total reach.)
        grand = result.mu
        cfg = self.config
        K = clusters.n_clusters
        rec = reception or {}
        n_cp = np.array([float(rec.get(c, (0.0, grand))[0]) for c in range(K)])
        r_emp = np.array([float(rec.get(c, (0.0, grand))[1]) for c in range(K)])
        w = n_cp / (n_cp + cfg.bridging_shrinkage_n0)   # empirical-Bayes trust per cluster
        r_shrunk = grand + w * (r_emp - grand)
        if reception_cap is not None and np.isfinite(reception_cap):
            # Cap each cluster's reception at the unconfounded exploration-anchored
            # upper bound: a distributed ring's common-mode lift of organic reception
            # above what random exposure reveals is discarded (§13.10, §6.2).
            r_shrunk = np.minimum(r_shrunk, reception_cap)
        return grand, r_shrunk

    def score_post(
        self,
        post_id: Id,
        author_id: Id,
        result: FactorizationResult,
        clusters: ClusterModel,
        reception: Optional[Mapping[int, tuple]] = None,
        reception_cap: Optional[float] = None,
    ) -> float:
        """B_LCB for a single post (§4.2, empirical shrinkage form — Appendix C.5)."""
        grand, r_shrunk = self._receptions(
            post_id, author_id, result, clusters, reception, reception_cap)
        if r_shrunk is None:
            return float("-inf")          # post not in the factorization
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
        reception: Optional[Mapping[Id, Mapping[int, tuple]]] = None,
        reception_caps: Optional[Mapping[Id, float]] = None,
    ) -> BridgingScores:
        """Score every post in the factorization.

        ``reception`` maps post id -> {cluster index -> (n_cp, r_emp_cp)} (see
        ``_receptions``), built from the reactions + IPW weights + cluster labels.
        """
        out = BridgingScores()
        for pid in result.y_post:
            author = post_authors.get(pid)
            rec = None if reception is None else reception.get(pid)
            cap = None if reception_caps is None else reception_caps.get(pid)
            grand, r_shrunk = self._receptions(pid, author, result, clusters, rec, cap)
            out.b_lcb[pid] = float("-inf") if r_shrunk is None else float(self._aggregate(r_shrunk))
            out.b_scalar[pid] = result.b_post.get(pid, 0.0)
            out.per_cluster[pid] = r_shrunk if r_shrunk is not None else np.array([])
        return out
