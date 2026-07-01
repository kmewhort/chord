"""The response model P(react | x_u, y_p) (Appendix C.4) — the non-circular DGP.

A simulated user's reaction depends on the alignment between their (true, possibly
higher-dimensional) opinion ``x_u`` and the post's true loading ``y_p`` — but two
things keep it from being the exact model the estimator fits:

* **Engagement ≠ value.** What makes a user *react at all* is driven by the post's
  ``toxicity`` (an affective-polarization pull) far more than its ``quality``; and
  toxicity *sharpens* the approve/disapprove divide. So a reaction-maximizing
  (engagement) ranker chases toxic, polarizing content, while genuine ``quality``
  — the thing a healthy feed should surface — barely moves the reaction signal.
* **Nonlinearity + hidden axes.** The map is a saturating logistic over a
  ``d_true``-dimensional opinion space the estimator only partially represents.

``react`` samples a reaction; ``expected_approval`` and ``reaction_pull`` give the
noise-free expectations the welfare/oracle metrics need.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import ReactionKind
from .content import PostTruth
from .population import Agent

# DGP constants (tunable). Toxicity dominates the engagement pull and amplifies the
# opinion divide; quality gives only a mild approval nudge.
TOX_ENGAGE_PULL = 1.2      # how much toxicity raises P(react at all)
TOX_DIVIDE_AMP = 1.5       # how much toxicity sharpens the approve/disapprove split
QUALITY_APPROVE_BONUS = 0.4
ALIGN_SCALE = 0.6          # scales the raw inner product into the logistic
BASE_REACT = 0.35


def _alignment(agent: Agent, truth: PostTruth) -> float:
    return float(np.dot(agent.opinion, truth.loading))


def approve_logit(agent: Agent, truth: PostTruth) -> float:
    a = ALIGN_SCALE * _alignment(agent, truth)
    return (agent.selectivity * a * (1.0 + TOX_DIVIDE_AMP * truth.toxicity)
            + QUALITY_APPROVE_BONUS * (truth.quality - 0.5))


def expected_approval(agent: Agent, truth: PostTruth) -> float:
    """Noise-free P(approve | react) — used by welfare/oracle, never by the ranker."""
    return float(1.0 / (1.0 + np.exp(-np.clip(approve_logit(agent, truth), -30, 30))))


def reaction_pull(agent: Agent, truth: PostTruth) -> float:
    """P(the user reacts at all) — the engagement signal, driven by toxicity."""
    align = abs(ALIGN_SCALE * _alignment(agent, truth))
    p = agent.reactivity * (BASE_REACT + TOX_ENGAGE_PULL * truth.toxicity + 0.1 * align)
    return float(np.clip(p, 0.0, 1.0))


def react(agent: Agent, truth: PostTruth, rng: np.random.Generator) -> Optional[ReactionKind]:
    """Sample a reaction (or None) given the hidden truth (Appendix C.4)."""
    if rng.random() > reaction_pull(agent, truth):
        return ReactionKind.EXPOSED_NO_REACTION      # scrolled past (weak negative)
    p_approve = expected_approval(agent, truth)
    if rng.random() < p_approve:
        return ReactionKind.BOOST if p_approve > 0.7 else ReactionKind.FAVORITE
    return ReactionKind.MUTE if p_approve < 0.3 else ReactionKind.EXPOSED_NO_REACTION
