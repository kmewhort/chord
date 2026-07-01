"""Coordination discounting for collusion resistance (§5/§10).

A *distributed* sybil ring camouflages its puppets across opinion clusters and has
them all boost one target — manufacturing fake cross-cluster support that defeats
B_LCB's min-over-clusters (the out-diversity λ weight, §5, does not help: the
puppets also rate genuine content, so they are not single-target *raters*). The
signature the attack cannot hide is **coordination**: a genuine bridging post is
approved by *independent* raters who otherwise behave differently, whereas the ring's
approvers co-approve the same narrow set in lockstep.

Following the pairwise-bounded / connection-oriented-cluster-matching idea from
collusion-resistant quadratic funding (Buterin–Hitzig–Weyl 2019; Miller–Weyl–Erichsen
COCM 2022), we score each post by how correlated its approvers are and discount the
bridged support of posts whose approval is coordinated. Independent approval — real
bridging — is untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from ..types import Id, Reaction


class CollusionTracker:
    """Rolling detector of *loyal* boosting blocs (the camouflaged-ring signature).

    A distributed ring hides its puppets' opinion positions and full approval sets,
    but it cannot hide the one thing it must do: have the *same* accounts approve
    *every* one of a target's posts, window after window. Genuine approval is casual
    (a user likes a post or two); a ring booster is maximally loyal. This tracker
    accumulates, with exponential decay, how large a fraction of each author's
    positive support comes from raters who approve almost all of that author's posts
    — the ``manufactured_fraction`` — which the loop subtracts from B_LCB. Because it
    keys on same-author loyalty over time, camouflage on *other* items does not dilute
    it (contrast :func:`coordination_scores`, which camouflage defeats).
    """

    def __init__(self, decay: float = 0.6, loyalty_threshold: float = 0.75,
                 min_evidence: float = 2.0):
        self.decay = decay
        self.loyalty_threshold = loyalty_threshold
        self.min_evidence = min_evidence
        self._support: Dict[Id, Dict[Id, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, reactions: Sequence[Reaction], post_authors: Mapping[Id, Id]) -> None:
        for a in self._support:
            for r in self._support[a]:
                self._support[a][r] *= self.decay
        for rx in reactions:
            if rx.value <= 0:
                continue
            a = post_authors.get(rx.post_id)
            if a is not None:
                self._support[a][rx.user_id] += 1.0

    def manufactured_fraction(
        self, author: Id,
        opinion_coord: Optional[Mapping[Id, float]] = None,
        cluster_of: Optional[Mapping[Id, int]] = None,
        n_clusters: int = 2,
    ) -> float:
        """Opinion-dispersed super-loyal support fraction for ``author`` ∈ [0,1].

        Loyalty is judged relative to the *most* loyal supporter: a distributed ring
        is a large bloc that all approve every one of the target's posts, so they pile
        up near the maximum support level, whereas genuine approval is graded. We take
        the mass of support at ≥ ``loyalty_threshold`` of the max, then — crucially —
        **gate it by how opinion-dispersed the loyal bloc is**. A camouflaged ring
        spreads its puppets across the whole opinion axis (that is how it fakes
        cross-cluster support); a genuine loyal fanbase is opinion-*coherent* and is not
        penalized. This is the anomaly gate — behavioral loyalty × opinion dispersion.

        Dispersion is measured on the **continuous** opinion axis ``opinion_coord`` (the
        spectral coordinate): the std of the loyal bloc's positions relative to the
        population std. This is robust to a *degenerate* discrete split (on weakly-
        divided data the 2-way clustering can be ~all-one-cluster, which made the old
        cluster-entropy gate read every bloc as coherent). ``cluster_of`` is a legacy
        discrete fallback used only when no coordinate is supplied.
        """
        support = self._support.get(author, {})
        if not support:
            return 0.0
        total = sum(support.values())
        max_s = max(support.values())
        if total <= 0.0 or max_s < self.min_evidence:
            return 0.0
        # Continuous loyalty weight, NOT a hard cutoff: an adaptive ring can sit just
        # under a threshold (approve 70% of a target's posts to dodge a 75% cutoff) and
        # still inflate. loyalty(r) = support/max ramps smoothly, so partial-approval
        # still counts — there is no cliff to exploit. (A ring that instead spreads so
        # thin each puppet approves ≲ one of the target's posts falls below min_evidence,
        # but then every puppet is indistinguishable from a genuine casual supporter — a
        # per-window impossibility, not a tuning gap; see test_adaptive_ring.)
        raters = [r for r, s in support.items() if s >= self.min_evidence]
        if not raters:
            return 0.0
        loyalty = {r: (support[r] / max_s) for r in raters}
        mass = sum(support[r] * loyalty[r] for r in raters)   # loyalty-weighted support
        frac = mass / total
        if opinion_coord is not None and len(raters) >= 2:
            coords = np.array([opinion_coord[r] for r in raters if r in opinion_coord])
            wts = np.array([support[r] * loyalty[r] for r in raters if r in opinion_coord])
            pop_c = np.array([v for v in opinion_coord.values()])
            if coords.size >= 2 and pop_c.size >= 2 and wts.sum() > 0 and pop_c.std() > 1e-9:
                m = np.average(coords, weights=wts)
                wstd = float(np.sqrt(np.average((coords - m) ** 2, weights=wts)))
                spread = min(1.0, wstd / float(pop_c.std()))   # concentrated fanbase → ~0
                frac *= spread
        elif cluster_of is not None and n_clusters > 1:
            labels = [cluster_of.get(r) for r in raters if support[r] >= self.loyalty_threshold * max_s]
            labels = [c for c in labels if c is not None]
            if labels:
                counts = np.array(list(Counter(labels).values()), dtype=float)
                p = counts / counts.sum()
                frac *= float(-(p * np.log(p)).sum()) / np.log(n_clusters)
        return float(frac)


def coordination_scores(
    reactions: Sequence[Reaction],
    min_approvers: int = 3,
    max_approvers: int = 200,
) -> Dict[Id, float]:
    """Per-post approver coordination ∈ [0,1] — mean pairwise approval-set Jaccard.

    For each post, look at the raters who approved it and how much their *overall*
    approval sets overlap. Coordinated boosters (who co-approve the same items) score
    near 1; a post approved by otherwise-dissimilar raters scores near 0. Posts with
    fewer than ``min_approvers`` approvers score 0 (too little evidence). Approver
    sets are capped at ``max_approvers`` (deterministic head) to bound the O(k²) pass.
    """
    approved: Dict[Id, set] = defaultdict(set)   # rater -> set of approved post ids
    approvers: Dict[Id, list] = defaultdict(list)  # post -> list of approving raters
    for r in reactions:
        if r.value > 0:
            approved[r.user_id].add(r.post_id)
            approvers[r.post_id].append(r.user_id)

    out: Dict[Id, float] = {}
    for pid, raters in approvers.items():
        uniq = list(dict.fromkeys(raters))[:max_approvers]
        if len(uniq) < min_approvers:
            out[pid] = 0.0
            continue
        sets = [approved[u] for u in uniq]
        sims = []
        for i in range(len(sets)):
            si = sets[i]
            for j in range(i + 1, len(sets)):
                sj = sets[j]
                union = len(si | sj)
                sims.append((len(si & sj) / union) if union else 0.0)
        out[pid] = float(np.mean(sims)) if sims else 0.0
    return out
