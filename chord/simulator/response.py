"""The response model P(react | x_u, y_p) (Appendix C.4).

A simulated user's reaction to a post depends on the alignment between their
opinion ``x_u`` and the post's true loading ``y_p``: aligned content is boosted,
anti-aligned content is muted, and near-orthogonal content draws a favorite or a
silent pass. This is the ground-truth data-generating process the estimator must
invert; crucially it is MNAR-inducing only through *exposure* (§6.1), so the
simulator can log exactly what was shown.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import ReactionKind
from .population import Agent


def react(
    agent: Agent,
    loading: np.ndarray,
    rng: np.random.Generator,
) -> Optional[ReactionKind]:
    """Sample a reaction (or None) given alignment (Appendix C.4).

    The affinity is the inner product between opinion and loading; a logistic
    map turns it into approve/disapprove probabilities scaled by the agent's
    reactivity and selectivity. Indiscriminate scrollers (low selectivity) react
    near-randomly — near-zero-information reactions the §5 weighting must discount.
    """
    affinity = float(np.dot(agent.opinion, loading))
    # selectivity sharpens the logistic; indiscriminate users are near-flat.
    z = agent.selectivity * affinity
    p_approve = 1.0 / (1.0 + np.exp(-z))

    # reactivity gates whether they react at all; the rest is a silent pass.
    if rng.random() > agent.reactivity * 0.8:
        return ReactionKind.EXPOSED_NO_REACTION

    if rng.random() < p_approve:
        # strong approve -> boost; mild -> favorite
        return ReactionKind.BOOST if p_approve > 0.7 else ReactionKind.FAVORITE
    else:
        # strong disapprove -> mute; mild -> silent pass (weak negative)
        return ReactionKind.MUTE if p_approve < 0.3 else ReactionKind.EXPOSED_NO_REACTION
