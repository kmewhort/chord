"""Estimated content depth q_p — an *earned* latent, not an author-set feature (§10/§13).

The anti-bait mechanism (§7.3/§10) gates a post's positive bridged support by its
*depth*. If depth is a field the author writes (`post.features['depth']`), a baiter
forges it and shallow content is crowned — a §12 wall violation (depth is an authority
quantity on the author-settable side). So depth is instead **estimated the way
``B_LCB`` estimates bridged support**, but on a separate *quality/vouch* channel: raters
(or hard-to-forge reader signals) vouch that a post is substantive, and

    q_p = agg_c  [ empirical-Bayes-shrunk per-cluster mean of those vouches ]

is the tested, cross-cluster support for the content's *merit*, net of ideology. An
author cannot set it — it emerges from *other* accounts' λ-weighted, opinion-dispersed
vouching, and inherits CHORD's collusion defenses (a bait's fake vouchers face the same
loyalty/out-diversity/exploration machinery as a boost ring). A post with no vouches
sits at neutral depth (0.5): untested, neither promoted nor gated.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from ..config import ChordConfig
from ..types import Id, Reaction
from .bridging import BridgingScorer, ClusterModel, cluster_reception


def estimate_depth(
    vouch_reactions: Sequence[Reaction],
    weights: Sequence[float],
    clusters: ClusterModel,
    config: ChordConfig,
) -> Dict[Id, float]:
    """Per-post depth ∈ [0,1] from λ/IPW-weighted, per-cluster, EB-shrunk vouches.

    ``weights[i]`` is the propensity/λ weight of ``vouch_reactions[i]`` (so a fresh
    sybil's vouch counts ~0). Returns depth only for posts that received vouches; the
    caller treats a missing post as neutral (0.5).
    """
    if not vouch_reactions:
        return {}
    # Prior is NEUTRAL merit (0 → depth 0.5), not the global vouch mean: an unvouched or
    # thinly-vouched post sits at neutral, an anti-vouched bait falls *below* it, and a
    # cross-cluster-vouched post rises above — sharper than regressing toward a (positive)
    # population mean, which would leniently lift a bait back toward the crowd.
    grand = 0.0
    reception = cluster_reception(vouch_reactions, weights, clusters)
    n0 = config.bridging_shrinkage_n0
    aggregate = BridgingScorer(config)._aggregate
    K = clusters.n_clusters
    out: Dict[Id, float] = {}
    for pid, rec in reception.items():
        r_shrunk = np.array([
            grand + (rec.get(c, (0.0, grand))[0] / (rec.get(c, (0.0, grand))[0] + n0))
            * (rec.get(c, (0.0, grand))[1] - grand)
            for c in range(K)
        ])
        q = aggregate(r_shrunk)                       # signed merit reception ∈ [-1,1]
        out[pid] = float(np.clip((q + 1.0) / 2.0, 0.0, 1.0))   # → depth ∈ [0,1]
    return out
