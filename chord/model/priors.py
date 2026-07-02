"""Hierarchical author×cluster prior for B_LCB shrinkage (§4.2, E9).

The global-mean prior μ makes B_LCB lenient on *untested one-sided* content: a firehose
post with few ratings regresses to neutral, so §8's budget has to do §4's job. Replace μ
with a **hierarchical** prior — shrink each cluster's reception toward the author's own
historical reception *in that cluster*, itself shrunk toward the cluster mean, itself
shrunk toward the global mean:

    prior_cp = shrink(author_ac_history → shrink(cluster_c_mean → μ))

so an untested post from an author cluster c has consistently disliked regresses to *that
low prior*, and B_LCB predicts-low before the budget bites — while a well-observed post
overwhelms the prior at the usual n_cp/(n_cp+n0) rate (self-correcting for reformed
authors). It stays fully reproducible: author history is a deterministic function of the
data. Author history accumulates across windows with exponential decay.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Tuple

from ..types import Id


class AuthorClusterReception:
    """Rolling, decayed per-(author, cluster) reception — the author-history level."""

    def __init__(self, decay: float = 0.7):
        self.decay = decay
        self._sum: Dict[Id, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._wt: Dict[Id, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, reception: Mapping[Id, Mapping[int, tuple]],
               post_authors: Mapping[Id, Id]) -> None:
        """Decay, then fold in this window's per-(post, cluster) reception by author."""
        for a in self._sum:
            for c in self._sum[a]:
                self._sum[a][c] *= self.decay
                self._wt[a][c] *= self.decay
        for pid, rec in reception.items():
            a = post_authors.get(pid)
            if a is None:
                continue
            for c, (n, m) in rec.items():
                self._sum[a][c] += n * m
                self._wt[a][c] += n

    def author_mean(self, author: Id, cluster: int) -> Tuple[Optional[float], float]:
        w = self._wt.get(author, {}).get(cluster, 0.0)
        return (self._sum[author][cluster] / w, w) if w > 0 else (None, 0.0)


def hierarchical_priors(
    reception: Mapping[Id, Mapping[int, tuple]],
    post_authors: Mapping[Id, Id],
    tracker: AuthorClusterReception,
    mu: float,
    n0: float,
    n0_author: float,
    n_clusters: int,
    vouch_tracker: Optional[AuthorClusterReception] = None,
) -> Dict[Id, List[float]]:
    """Per-post, per-cluster shrinkage prior ``prior_cp`` (uses ``tracker`` history only —
    call before ``tracker.update`` for this window so priors are leave-current-out).

    If ``vouch_tracker`` is given (E9-quality), the author lift is **asymmetric**: the
    approval history can only *lower* the prior below the cluster baseline (firehose pre-
    emption), while *raising* it above the baseline is licensed only by the author's earned
    cross-cluster **vouches** — a merit credit ``max(0, v̄_ac)·(w/(w+n0))`` per cluster. So a
    broadly-approved-but-unvouched bait (and a partisan-consistent author) cannot buy a prior
    lift with approval alone; only demonstrated quality does. Without ``vouch_tracker`` the
    original symmetric approval-history rule is used.
    """
    # cluster level: this window's overall reception per cluster, shrunk toward μ
    csum: Dict[int, float] = defaultdict(float)
    cwt: Dict[int, float] = defaultdict(float)
    for rec in reception.values():
        for c, (n, m) in rec.items():
            csum[c] += n * m
            cwt[c] += n
    cluster_prior = {
        c: mu + (cwt[c] / (cwt[c] + n0)) * (((csum[c] / cwt[c]) if cwt[c] > 0 else mu) - mu)
        for c in range(n_clusters)
    }
    # author level: shrink each author's cluster history toward the cluster prior
    ap: Dict[Tuple[Id, int], float] = {}
    for a in set(post_authors.values()):
        for c in range(n_clusters):
            am, aw = tracker.author_mean(a, c)
            cp = cluster_prior[c]
            if am is None:
                ap[(a, c)] = cp
                continue
            shrunk = cp + (aw / (aw + n0_author)) * (am - cp)
            if vouch_tracker is None:
                ap[(a, c)] = shrunk                       # symmetric: approval sets the prior
            else:
                vm, vw = vouch_tracker.author_mean(a, c)  # merit credit licenses raising only
                credit = 0.0 if vm is None else max(0.0, vm) * (vw / (vw + n0_author))
                ap[(a, c)] = min(shrunk, cp + credit)     # approval lowers freely; vouches cap the lift
    return {pid: [ap[(a, c)] for c in range(n_clusters)]
            for pid, a in post_authors.items()}
