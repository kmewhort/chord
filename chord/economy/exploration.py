"""Exploration pool — cold-start, base-rate-calibrated (§8).

New posts have no ``b_p``. Audition them via Thompson sampling — but **not** with
a flat optimistic prior, which systematically over-explores weak items because
the real base rate of "winners" is far below 50% [Dynamic Prior TS 2025].
Initialize the prior to the empirical newcomer strength rate.

A commons-funded fraction ``epsilon`` of every feed's slots is reserved for
high-uncertainty posts, routed preferentially to high-q_scout raters (they
resolve uncertainty in the fewest impressions). The audition closes on
**evaluation saturation** (posterior variance below threshold), not wall-clock,
so slow-burn long-form is not buried. Grants are bounded per verified identity.

The asymmetry of §4.2: this pool samples *high* uncertainty **optimistically** (a
Thompson draw) to decide what to audition, whereas B_LCB uses uncertainty
pessimistically to decide what to crown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..config import ChordConfig
from ..types import Id


@dataclass
class BetaPosterior:
    """A Beta(alpha, beta) belief about a post's probability of being a winner."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    def update(self, reward: float) -> None:
        """Bayesian update with a reward in [0,1] (fractional successes allowed)."""
        reward = float(min(1.0, max(0.0, reward)))
        self.alpha += reward
        self.beta += 1.0 - reward


class ExplorationPool:
    """Thompson-sampling audition of unproven posts (§8)."""

    def __init__(self, config: ChordConfig, seed: int = 0):
        self.config = config
        self._rng = np.random.default_rng(seed)
        self.posteriors: Dict[Id, BetaPosterior] = {}
        self.closed: Dict[Id, bool] = {}
        # Prior sample size (pseudo-count strength) for the base-rate prior.
        self._prior_strength = 4.0

    def _base_rate_prior(self) -> BetaPosterior:
        """Beta prior with mean = empirical newcomer strength rate (§8).

        Not a flat Beta(1,1): the mean is pinned to the real base rate so weak
        items are not systematically over-explored.
        """
        r = float(min(0.999, max(1e-3, self.config.newcomer_base_rate)))
        s = self._prior_strength
        return BetaPosterior(alpha=r * s, beta=(1.0 - r) * s)

    def register(self, post_id: Id) -> None:
        """Enroll a new post into the audition with the base-rate prior."""
        if post_id not in self.posteriors:
            self.posteriors[post_id] = self._base_rate_prior()
            self.closed[post_id] = False

    def sample_score(self, post_id: Id) -> float:
        """Optimistic Thompson draw Phi_tilde(p) used to decide what to audition."""
        post = self.posteriors.get(post_id)
        if post is None:
            self.register(post_id)
            post = self.posteriors[post_id]
        return float(self._rng.beta(post.alpha, post.beta))

    def observe(self, post_id: Id, reward: float) -> None:
        """Fold an audition outcome into the posterior; close on saturation."""
        if post_id not in self.posteriors:
            self.register(post_id)
        if self.closed.get(post_id):
            return
        self.posteriors[post_id].update(reward)
        if self.posteriors[post_id].variance <= self.config.exploration_saturation_var:
            self.closed[post_id] = True

    def is_open(self, post_id: Id) -> bool:
        """Whether a post is still auditioning (not yet saturated)."""
        return post_id in self.posteriors and not self.closed.get(post_id, False)

    def posterior_mean(self, post_id: Id) -> Optional[float]:
        post = self.posteriors.get(post_id)
        return None if post is None else post.mean
