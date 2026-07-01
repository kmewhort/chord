"""Semi-synthetic propensity harness (Appendix C.3).

Because the propensity model is deliberately an open menu (§6.3), the cleanest
test gives us control of ground truth:

1. Take a near-complete (MAR) ratings matrix R over users x posts.
2. Define a synthetic logging policy that exposes items in proportion to
   predicted **in-group alignment** — deliberately inducing MNAR of the exact
   kind §6.1 warns about (in-group over-exposure).
3. Sample an observed set E ~ pi_log; hide the rest.
4. Fit the model on E under each propensity option.
5. Score against the *hidden* full matrix as ground truth. Report:
   (a) does uncorrected fitting inflate b_p / shrink ||y|| (the predicted
       pathology)?
   (b) does IPW/DR recover the unbiased ranking?
   (d) the value of the random epsilon-anchor — sweep its size toward zero and
       watch identifiability fail.

This module builds the world and runs the experiment; :mod:`tests.test_mnar`
asserts the predicted qualitative outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ChordConfig
from ..model import MatrixFactorization, fit_divisiveness
from ..propensity import (
    UniformExplorationModel,
    compute_ipw_weights,
)
from ..propensity.base import PropensityModel
from ..types import Exposure, Id, Post, Reaction


@dataclass
class SyntheticWorld:
    """A ground-truth two-pole world with a full (MAR) ratings matrix."""

    n_users: int
    n_posts: int
    opinions: np.ndarray            # (n_users, d) latent x_u
    loadings: np.ndarray            # (n_posts, d) latent y_p
    intercepts: np.ndarray          # (n_posts,) true bridged support b_p
    full_ratings: np.ndarray        # (n_users, n_posts) dense true ratings
    posts: Dict[Id, Post]
    clusters: np.ndarray            # (n_users,) ground-truth cluster labels
    d: int


def make_world(
    n_users: int = 60,
    n_posts: int = 40,
    d: int = 2,
    frac_universal: float = 0.3,
    noise: float = 0.15,
    seed: int = 0,
) -> SyntheticWorld:
    """Build a bipolar world with a mix of universal and partisan posts.

    A fraction of posts are *universal* (near-zero loading, high intercept — they
    genuinely bridge); the rest are partisan (heavy loading on one pole). The
    dense rating is ``mu + b_p + <x_u, y_p> + noise``.
    """
    rng = np.random.default_rng(seed)
    # users split into two poles on axis 0
    clusters = (np.arange(n_users) >= n_users // 2).astype(int)
    opinions = rng.normal(0, 0.3, size=(n_users, d))
    opinions[:, 0] += np.where(clusters == 0, 1.5, -1.5)

    loadings = np.zeros((n_posts, d))
    intercepts = np.zeros(n_posts)
    n_univ = int(frac_universal * n_posts)
    for p in range(n_posts):
        if p < n_univ:
            loadings[p] = rng.normal(0, 0.1, size=d)   # universal: near origin
            intercepts[p] = rng.uniform(0.3, 0.6)      # genuinely well-received
        else:
            pole = 1 if (p % 2 == 0) else -1
            loadings[p] = rng.normal(0, 0.15, size=d)
            loadings[p, 0] = pole * rng.uniform(1.0, 1.6)  # partisan
            intercepts[p] = rng.uniform(-0.1, 0.1)

    mu = 0.0
    full = mu + intercepts[None, :] + opinions @ loadings.T
    full += rng.normal(0, noise, size=full.shape)
    full = np.clip(full, -1.0, 1.0)

    posts = {f"p{p}": Post(f"p{p}", author_id=f"auth{p}") for p in range(n_posts)}
    return SyntheticWorld(
        n_users=n_users, n_posts=n_posts, opinions=opinions, loadings=loadings,
        intercepts=intercepts, full_ratings=full, posts=posts, clusters=clusters,
        d=d,
    )


def logging_policy_exposures(
    world: SyntheticWorld,
    exposure_rate: float = 0.3,
    epsilon_anchor: float = 0.05,
    ingroup_bias: float = 4.0,
    seed: int = 0,
) -> Tuple[List[Reaction], List[Exposure], Dict[Tuple[Id, Id], float]]:
    """Sample MNAR observations from an in-group-over-exposing logging policy.

    The probability of exposing (u, p) rises with predicted in-group alignment
    ``<x_u, y_p>`` — so the outgroup that would dislike a partisan post rarely
    sees it (the §6.1 confound). A fraction ``epsilon_anchor`` of exposures are
    instead drawn uniformly at random (the unconfounded exploration anchor, §6.2)
    with known propensity ~ epsilon_anchor.

    Returns (reactions, exposures, true_propensities) where reactions carry the
    observed rating and exposures carry the source/propensity.
    """
    rng = np.random.default_rng(seed)
    reactions: List[Reaction] = []
    exposures: List[Exposure] = []
    true_pi: Dict[Tuple[Id, Id], float] = {}

    align = world.opinions @ world.loadings.T  # (n_users, n_posts)
    # organic exposure logit rises with in-group alignment
    organic_logit = ingroup_bias * align
    organic_p = 1.0 / (1.0 + np.exp(-organic_logit))
    # scale so the average organic exposure probability ~ exposure_rate
    organic_p *= exposure_rate / organic_p.mean()
    organic_p = np.clip(organic_p, 0.0, 0.98)

    for u in range(world.n_users):
        for p in range(world.n_posts):
            pid = f"p{p}"
            if rng.random() < epsilon_anchor:
                # exploration: uniform, known propensity, alignment-independent
                exposures.append(
                    Exposure(u, pid, source=_EXPLORE, propensity=epsilon_anchor)
                )
                true_pi[(u, pid)] = epsilon_anchor
                reactions.append(Reaction(u, pid, float(world.full_ratings[u, p])))
            elif rng.random() < organic_p[u, p]:
                pi = float(organic_p[u, p])
                exposures.append(Exposure(u, pid, source=_ORGANIC, propensity=pi))
                true_pi[(u, pid)] = pi
                reactions.append(Reaction(u, pid, float(world.full_ratings[u, p])))
    return reactions, exposures, true_pi


# import here to avoid a cycle at module import time
from ..types import ExposureSource as _ES  # noqa: E402
_ORGANIC = _ES.ORGANIC
_EXPLORE = _ES.EXPLORATION


@dataclass
class FitDiagnostics:
    """Pathology diagnostics on a fit (Appendix C.3 metrics)."""

    mean_intercept_partisan: float   # b_p averaged over partisan posts
    mean_loading_norm_partisan: float  # ||y_p|| averaged over partisan posts
    ranking_corr: float              # Spearman-ish corr of recovered vs true b_p


def _fit_and_diagnose(
    world: SyntheticWorld,
    reactions: List[Reaction],
    weights: Optional[np.ndarray],
    config: ChordConfig,
    seed: int = 0,
) -> FitDiagnostics:
    mf = MatrixFactorization(config, seed=seed)
    res = mf.fit(reactions, world.posts, weights)
    n_univ = int(0.3 * world.n_posts)
    partisan_ids = [f"p{p}" for p in range(n_univ, world.n_posts)]
    b_partisan = np.mean([res.b_post.get(pid, 0.0) for pid in partisan_ids])
    norm_partisan = np.mean([
        float(np.linalg.norm(res.y_post.get(pid, np.zeros(world.d))))
        for pid in partisan_ids
    ])
    # ranking correlation of recovered intercept vs true intercept
    recovered = np.array([res.b_post.get(f"p{p}", 0.0) for p in range(world.n_posts)])
    corr = float(np.corrcoef(_rank(recovered), _rank(world.intercepts))[0, 1])
    return FitDiagnostics(
        mean_intercept_partisan=float(b_partisan),
        mean_loading_norm_partisan=float(norm_partisan),
        ranking_corr=corr,
    )


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a))
    return ranks


def run_experiment(
    world: SyntheticWorld,
    config: Optional[ChordConfig] = None,
    epsilon_anchor: float = 0.05,
    seed: int = 0,
) -> Dict[str, FitDiagnostics]:
    """Fit uncorrected vs IPW-corrected and return the diagnostics for each.

    The predicted outcome (§6.1): uncorrected fitting *inflates* the partisan
    intercept b_p (in-group popularity masquerades as bridging) and *shrinks*
    ||y_p||; IPW correction recovers a ranking closer to the true intercepts.
    """
    config = config or ChordConfig(d=world.d, mf_iters=40, reg_bias_post=0.05)
    reactions, exposures, true_pi = logging_policy_exposures(
        world, epsilon_anchor=epsilon_anchor, seed=seed
    )

    # --- uncorrected fit (all weights = 1) ---
    uncorrected = _fit_and_diagnose(world, reactions, None, config, seed)

    # --- IPW-corrected fit using the TRUE logging propensities ---
    class _TruePi(PropensityModel):
        def propensity(self, u, p, exposure=None):
            return max(true_pi.get((u, p), epsilon_anchor), 1e-6)

    weights = compute_ipw_weights(reactions, _TruePi(), config, exposures={
        (e.user_id, e.post_id): e for e in exposures
    })
    corrected = _fit_and_diagnose(world, reactions, weights, config, seed)

    return {"uncorrected": uncorrected, "corrected": corrected}
