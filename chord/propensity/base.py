"""Propensity model interface (§6.3) — an open menu, not a commitment.

Because misspecified propensities silently reintroduce the MNAR bias the whole
edifice corrects (§6.1), the propensity model is deliberately pluggable. Every
estimator implements :class:`PropensityModel`: given a (user, post) pair (and
optionally the logging context), return an estimate of the exposure probability
``pi_up``, floored at ``epsilon`` because the exploration pool guarantees
``pi >= epsilon > 0`` for audited items (§6.2) — the positivity requirement that
makes IPW unbiased.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import Exposure, Id


class PropensityModel(ABC):
    """Estimate the probability that a (user, post) pair was exposed."""

    @abstractmethod
    def propensity(self, user_id: Id, post_id: Id, exposure: Optional[Exposure] = None) -> float:
        """Return pi_up in (0, 1]. Implementations must floor at epsilon > 0."""
        raise NotImplementedError
