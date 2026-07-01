"""Candidate aggregators/penalties for the §4.2 keystone, on real per-cluster data.

The finding (test_community_notes_keystone): B_LCB = min_c[r̂_cp − β·σ/√(n_cp+1)]
is beaten by the scalar b_p and a naive mean, because n_cp is a raw *rating* count,
so the pessimism subtracts noise (thin-cluster) rather than risk (divisive).

The research decomposed the fix into two axes and mapped `min_c` onto a family of
welfare/risk aggregators (Atkinson/CES, Nash geometric mean = Polis "group-informed
consensus", CVaR) with `min` as the extreme, and the penalty onto principled
n-dependent bounds (Wilson, empirical-Bayes shrinkage) instead of σ/√n_rating.

This module scores each note from its *empirical per-cluster reception* — the ratings
grouped by the rater's opinion cluster — under an aggregator × penalty grid, so we
can ask directly: does any cross-cluster variant beat b_p at reproducing real
decisions? Model-light on purpose (needs only the clustering); the raw signed
ratings carry the per-cluster mean/count/variance the penalties consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class ClusterStats:
    """Per-cluster empirical reception of one note."""
    mean: np.ndarray    # (k,) mean signed rating by cluster-c raters (nan if none)
    count: np.ndarray   # (k,) number of cluster-c raters
    var: np.ndarray     # (k,) within-cluster variance
    grand: float        # overall mean signed rating on the note


def cluster_stats_for_notes(
    reactions, assignments: Dict, n_clusters: int,
) -> Dict:
    """Group each post's signed ratings by the rater's cluster → ClusterStats."""
    # accumulate sums per (post, cluster)
    sums: Dict = {}
    sqs: Dict = {}
    cnts: Dict = {}
    tot: Dict = {}
    totc: Dict = {}
    for r in reactions:
        c = assignments.get(r.user_id)
        if c is None:
            continue
        pid = r.post_id
        if pid not in sums:
            sums[pid] = np.zeros(n_clusters)
            sqs[pid] = np.zeros(n_clusters)
            cnts[pid] = np.zeros(n_clusters)
            tot[pid] = 0.0
            totc[pid] = 0.0
        sums[pid][c] += r.value
        sqs[pid][c] += r.value * r.value
        cnts[pid][c] += 1.0
        tot[pid] += r.value
        totc[pid] += 1.0
    out: Dict = {}
    for pid in sums:
        n = cnts[pid]
        safe = np.where(n > 0, n, 1.0)
        mean = np.where(n > 0, sums[pid] / safe, np.nan)
        var = np.where(n > 0, sqs[pid] / safe - (sums[pid] / safe) ** 2, 0.0)
        var = np.clip(var, 0.0, None)
        grand = tot[pid] / totc[pid] if totc[pid] > 0 else 0.0
        out[pid] = ClusterStats(mean=mean, count=n, var=var, grand=float(grand))
    return out


# ------------------------------------------------------------ penalties
def penalty_none(m, n, v, g, beta=1.0):
    return m


def penalty_wald_count(m, n, v, g, beta=1.0):
    """The current B_LCB penalty: subtract β·σ/√(n+1) with σ=1 (rating count)."""
    return m - beta * 1.0 / np.sqrt(n + 1.0)


def penalty_wald_var(m, n, v, g, beta=1.0):
    """Variance-aware: subtract β·√(v/(n)) — empirical dispersion, not a constant."""
    return m - beta * np.sqrt(v / np.maximum(n, 1.0))


def penalty_wilson(m, n, v, g, beta=1.64):
    """Wilson score lower bound on the cluster mean rescaled to [0,1] (Agresti-Coull
    pseudocounts stop it blowing up at small n)."""
    p = np.clip((m + 1.0) / 2.0, 0.0, 1.0)     # signed [-1,1] → [0,1]
    z = beta
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low01 = centre - half
    return low01 * 2.0 - 1.0                    # back to [-1,1]


def penalty_james_stein(m, n, v, g, beta=1.0, n0=8.0):
    """Empirical-Bayes shrink each cluster mean toward the grand mean, more for
    thinly-rated clusters. Borrows strength instead of penalizing sparsity."""
    w = n / (n + n0)                            # trust ∝ how well-rated the cluster is
    return g + w * (m - g)


PENALTIES: Dict[str, Callable] = {
    "none": penalty_none,
    "wald_count": penalty_wald_count,      # ~ current B_LCB
    "wald_var": penalty_wald_var,
    "wilson": penalty_wilson,
    "james_stein": penalty_james_stein,
}


# ------------------------------------------------------------ aggregators
def agg_min(a):
    return np.nanmin(a)


def agg_mean(a):
    return np.nanmean(a)


def agg_nash(a):
    """Geometric mean of agree-probabilities = Polis 'group-informed consensus'."""
    p = np.clip((a + 1.0) / 2.0, 1e-6, 1.0)
    p = p[~np.isnan(p)]
    return float(np.exp(np.mean(np.log(p))) * 2.0 - 1.0)


def agg_cvar(a, alpha=0.5):
    """Mean of the worst ⌈α·k⌉ clusters (coherent risk; min is the α→0 limit)."""
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * a.size)))
    return float(np.mean(np.sort(a)[:k]))


def agg_ede(a, eps=1.0):
    """Atkinson equally-distributed-equivalent (ε=0 mean, 1 geo-mean, ∞ min)."""
    p = np.clip((a + 1.0) / 2.0, 1e-6, 1.0)
    p = p[~np.isnan(p)]
    if p.size == 0:
        return float("nan")
    if abs(eps - 1.0) < 1e-9:
        ede = np.exp(np.mean(np.log(p)))
    else:
        ede = np.mean(p ** (1 - eps)) ** (1.0 / (1 - eps))
    return float(ede * 2.0 - 1.0)


AGGREGATORS: Dict[str, Callable] = {
    "min": agg_min,
    "mean": agg_mean,
    "nash": agg_nash,
    "cvar.5": lambda a: agg_cvar(a, 0.5),
    "ede1": lambda a: agg_ede(a, 1.0),
    "ede4": lambda a: agg_ede(a, 4.0),
}


def score_notes(stats: Dict, ids: List, aggregator: str, penalty: str,
                beta: float = 1.0) -> np.ndarray:
    """Score every note under one (aggregator, penalty) combination."""
    pen = PENALTIES[penalty]
    agg = AGGREGATORS[aggregator]
    out = np.empty(len(ids))
    for i, pid in enumerate(ids):
        s = stats[pid]
        mask = s.count > 0
        adj = pen(s.mean[mask], s.count[mask], s.var[mask], s.grand, beta)
        out[i] = agg(adj)
    return out
