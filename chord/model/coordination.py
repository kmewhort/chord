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

from collections import defaultdict
from typing import Dict, Sequence

import numpy as np

from ..types import Id, Reaction


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
