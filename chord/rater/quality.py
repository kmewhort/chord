"""Quality-tracking rater weight (§5) — never inverse variance.

The naive choice (inverse-variance weighting) is *actively harmful*: it rewards
predictability, so extremists (predictable) get high weight while thoughtful
evaluators get inflated residuals and are downweighted — empirically amplifying
ideologically extreme raters and making the system more vulnerable to partisan
attack [QSMF 2026].

The correction is a quality-tracking / peer-prediction weight: give more
influence to raters whose *ideology-adjusted* ratings are consistent with the
note-quality estimate learned from all ratings. Concretely, project out the
alignment term ``<x_u, y_p>`` and measure how well a rater's residual tracks the
bridged-quality signal ``b_p`` across the posts they rated. **Weight by agreement
with the bridged-quality signal after ideology is projected out — never by
residual variance.**
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..config import ChordConfig
from ..types import Id, Post, Reaction
from ..model.factorization import FactorizationResult


def quality_tracking_weight(
    reactions: Sequence[Reaction],
    posts: Mapping[Id, Post],
    result: FactorizationResult,
    config: ChordConfig,
) -> Dict[Id, float]:
    """Per-rater agreement with the bridged-quality signal (§5).

    For each rater u and post p they rated, the ideology-adjusted rating is

        adj_up = r_up - mu - b_u - b_a(p) - <x_u, y_p>

    which, under a good quality tracker, should track the note-quality estimate
    ``b_p``. We take the (positive part of the) correlation between ``adj_up`` and
    ``b_p`` over each rater's rated posts as their quality weight, floored so no
    one is zeroed (mirroring eigentrust's teleport floor).
    """
    per_user_adj: Dict[Id, List[float]] = defaultdict(list)
    per_user_bp: Dict[Id, List[float]] = defaultdict(list)

    for rx in reactions:
        post = posts.get(rx.post_id)
        if post is None:
            continue
        u = rx.user_id
        x = result.x_user.get(u)
        y = result.y_post.get(rx.post_id)
        align = float(np.dot(x, y)) if x is not None and y is not None else 0.0
        adj = (
            rx.value
            - result.mu
            - result.b_user.get(u, 0.0)
            - result.b_author.get(post.author_id, 0.0)
            - align
        )
        per_user_adj[u].append(adj)
        per_user_bp[u].append(result.b_post.get(rx.post_id, 0.0))

    floor = (1.0 - config.eigentrust_delta)  # same floor philosophy as §5
    weights: Dict[Id, float] = {}
    for u, adj_list in per_user_adj.items():
        adj = np.asarray(adj_list)
        bp = np.asarray(per_user_bp[u])
        if adj.size < 2 or np.std(adj) < 1e-9 or np.std(bp) < 1e-9:
            # not enough signal to judge quality tracking -> floor
            weights[u] = floor
            continue
        corr = float(np.corrcoef(adj, bp)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        weights[u] = floor + (1.0 - floor) * max(0.0, corr)

    # normalize to a probability-like vector for blending with eigentrust
    total = sum(weights.values())
    if total > 0:
        weights = {u: w / total for u, w in weights.items()}
    return weights


def blend_lambda(
    eigentrust_lambda: Mapping[Id, float],
    quality_lambda: Mapping[Id, float],
    config: ChordConfig,
) -> Dict[Id, float]:
    """Blend eigentrust credibility with quality-tracking agreement (§5).

    Both are floored, non-negative weightings on the same rater set. The mix is
    a convex combination controlled by ``quality_track_mix``. The result is
    renormalized to sum to 1.
    """
    mix = config.quality_track_mix
    users = set(eigentrust_lambda) | set(quality_lambda)
    out: Dict[Id, float] = {}
    for u in users:
        e = eigentrust_lambda.get(u, 0.0)
        q = quality_lambda.get(u, 0.0)
        out[u] = (1.0 - mix) * e + mix * q
    total = sum(out.values())
    if total > 0:
        out = {u: w / total for u, w in out.items()}
    return out
