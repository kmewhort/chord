"""Personalized value V(u,p) (§7.1) and the factor vector (§7.3).

    V(u,p) = B_LCB(p)                     tested bridged support
           + (1-M) <x_u, y_p>            personalization
           - M * rho * y_p^T A y_p       divisiveness penalty

M in [0,1] is the master dial. M=0: give me what my side likes, divisiveness
included (engagement-like). M=1: pure bridging — broad tested support only,
partisan lean ignored, divisiveness penalized. This is a *consumption* choice and
therefore ungameable: it only changes the chooser's own feed.

The factor vector (§7.3): ``V_f`` are per-factor value functions (trend, scout,
depth, locality, recency…), each with its own consumption weight ``theta_f`` on
the simplex. ``bridge`` is the load-bearing factor built from V(u,p) above; other
factors are pluggable and blended by the user's theta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import numpy as np

from ..config import UserKnobs
from ..types import Id, Post
from ..model.factorization import FactorizationResult
from ..model.divisiveness import DivisivenessModel


def value(
    user_id: Id,
    post_id: Id,
    b_lcb: float,
    result: FactorizationResult,
    divisiveness: DivisivenessModel,
    knobs: UserKnobs,
) -> float:
    """Canonical V(u,p) (§7.1) with rho applied exactly once.

    D(p) = y^T A y is computed with the *unscaled* A (rho=1); the rho coefficient
    multiplies the penalty term exactly as written in §7.1.
    """
    x = result.x_user.get(user_id)
    y = result.y_post.get(post_id)
    if y is None:
        return b_lcb
    align = float(np.dot(x, y)) if x is not None else 0.0
    D = divisiveness.divisiveness(y, rho=1.0)
    M = knobs.M
    return b_lcb + (1.0 - M) * align - M * knobs.rho * D


@dataclass
class FactorContext:
    """Everything a factor value function needs to score one (user, post)."""

    user_id: Id
    post: Post
    b_lcb: float
    result: FactorizationResult
    divisiveness: DivisivenessModel
    knobs: UserKnobs
    extras: Dict[str, float]  # per-post precomputed signals (trend, recency…)


# A factor value function: context -> scalar V_f(u,p).
FactorFn = Callable[[FactorContext], float]


def bridge_factor(ctx: FactorContext) -> float:
    """The load-bearing bridging factor = V(u,p) (§7.1), with an optional depth gate.

    Bridging-bait — shallow, broadly-mildly-liked content — earns a high B_LCB it
    doesn't deserve (§10). When a system-level ``depth_gate`` weight is supplied
    (via ``extras['depth_gate']``, not a user knob), a post's *positive* bridged
    support is attenuated toward a floor by its depth/quality signal
    (``extras['depth']`` ∈ [0,1]): a shallow post cannot be crowned as bridging no
    matter how broad its approval, while negative scores are untouched. This is a
    multiplicative Goodhart gate — harder to game than adding a depth bonus, since a
    baiter cannot buy back a crown with more breadth. Gate off (=0) ⇒ exact §7.1.
    """
    b = ctx.b_lcb
    depth = ctx.extras.get("depth", 1.0)
    gate = ctx.extras.get("depth_gate", 0.0)
    if gate > 0.0 and b > 0.0:
        floor = 0.25  # even zero-depth content keeps this fraction of its bridge score
        b = b * (1.0 - gate * (1.0 - floor) * (1.0 - depth))
    v = value(ctx.user_id, ctx.post.id, b, ctx.result, ctx.divisiveness, ctx.knobs)
    # Structural depth reward (system integrity, not a user knob): promote genuine
    # depth/quality and demote shallow content, centred so median depth is neutral.
    reward = ctx.extras.get("depth_reward", 0.0)
    if reward > 0.0:
        v += reward * (depth - 0.5)
    return v


def recency_factor(ctx: FactorContext) -> float:
    """Simple recency factor: fresher posts score higher (extras['recency'])."""
    return ctx.extras.get("recency", 0.0)


def trend_factor(ctx: FactorContext) -> float:
    """Trend factor: current velocity of positive reception (extras['trend'])."""
    return ctx.extras.get("trend", 0.0)


def scout_factor(ctx: FactorContext) -> float:
    """Scout factor: leading indicator of trend (extras['scout'])."""
    return ctx.extras.get("scout", 0.0)


def depth_factor(ctx: FactorContext) -> float:
    """Depth factor: resists bridging-bait / shallow universal content (§10)."""
    return ctx.extras.get("depth", 0.0)


DEFAULT_FACTORS: Dict[str, FactorFn] = {
    "bridge": bridge_factor,
    "trend": trend_factor,
    "scout": scout_factor,
    "depth": depth_factor,
    "recency": recency_factor,
}


def blended_value(
    ctx: FactorContext,
    factors: Optional[Mapping[str, FactorFn]] = None,
) -> float:
    """sum_f theta_f V_f(u,p) — the user's factor mix on the simplex (§7.3)."""
    factors = factors or DEFAULT_FACTORS
    theta = ctx.knobs.normalized_theta()
    total = 0.0
    for f_name, weight in theta.items():
        fn = factors.get(f_name)
        if fn is None:
            continue
        total += weight * fn(ctx)
    return total
