"""Concrete propensity estimators from the §6.3 menu.

* :class:`UniformExplorationModel` — option (d)/(intervention harvesting): the
  exploration pool's known, alignment-independent pi ~ epsilon. This is the
  **unconfounded anchor** (§6.2) — ground truth for calibrating the others.
* :class:`PositionBasedModel` — option (b): classic CLTR examination propensity,
  pi decays with the display rank/slot the item was shown at.
* :class:`PolicyDerivedModel` — option (c): if the logging ranker's scores are
  known, derive pi from its softmax selection probabilities directly.

Each floors its estimate at ``epsilon`` (the exploration guarantee, §6.2), which
also ties the inverse-propensity clip to ``W_max = 1/epsilon``.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np

from ..types import Exposure, ExposureSource, Id
from .base import PropensityModel


class UniformExplorationModel(PropensityModel):
    """Known exposure probability for the randomized exploration slice (§6.2).

    Exploration exposures happen at the (approximately) uniform floor rate
    ``epsilon`` and are independent of alignment — the unconfounded anchor. For
    organic exposures this model returns the same floor, so it is really only
    correct for the exploration cohort; use it to calibrate/validate the others.
    """

    def __init__(self, epsilon: float):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self.epsilon = float(epsilon)

    def propensity(self, user_id: Id, post_id: Id, exposure: Optional[Exposure] = None) -> float:
        return self.epsilon


class PositionBasedModel(PropensityModel):
    """Examination model (§6.3b): pi decays with display position.

    pi_up = max(epsilon, examination(rank)), with a geometric examination curve
    ``base ** rank`` by default. The rank is taken from the exposure's ``slot``.
    """

    def __init__(self, epsilon: float, base: float = 0.9, exam: Optional[Callable[[int], float]] = None):
        if not 0.0 < base <= 1.0:
            raise ValueError("base must be in (0,1]")
        self.epsilon = float(epsilon)
        self.base = float(base)
        self._exam = exam

    def examination(self, rank: int) -> float:
        if self._exam is not None:
            return float(self._exam(rank))
        return float(self.base ** max(0, rank))

    def propensity(self, user_id: Id, post_id: Id, exposure: Optional[Exposure] = None) -> float:
        rank = exposure.slot if exposure is not None else 0
        return max(self.epsilon, self.examination(rank))


class PolicyDerivedModel(PropensityModel):
    """Policy-derived propensity (§6.3c) from the logging ranker's softmax.

    Given per-(user, candidate-set) logging scores, pi is the item's softmax
    selection probability, floored at epsilon. Scores are supplied as a callable
    ``scores_for(user_id) -> Mapping[post_id, score]`` covering that user's
    candidate set at logging time.
    """

    def __init__(self, epsilon: float, scores_for: Callable[[Id], Mapping[Id, float]], temperature: float = 1.0):
        self.epsilon = float(epsilon)
        self._scores_for = scores_for
        self.temperature = float(temperature)

    def propensity(self, user_id: Id, post_id: Id, exposure: Optional[Exposure] = None) -> float:
        scores = self._scores_for(user_id)
        if not scores or post_id not in scores:
            return self.epsilon
        vals = np.array(list(scores.values()), dtype=float) / self.temperature
        vals -= vals.max()  # numerical stability
        exp = np.exp(vals)
        probs = exp / exp.sum()
        keys = list(scores.keys())
        pi = float(probs[keys.index(post_id)])
        return max(self.epsilon, pi)


class LoggedPropensityModel(PropensityModel):
    """Use the propensity recorded on the exposure event itself.

    When the serving plane logs ``exposure.propensity`` at serve time (the
    honest thing to do — Appendix D.2), the learning plane can just read it back.
    Falls back to epsilon when unavailable.
    """

    def __init__(self, epsilon: float):
        self.epsilon = float(epsilon)

    def propensity(self, user_id: Id, post_id: Id, exposure: Optional[Exposure] = None) -> float:
        if exposure is not None and exposure.propensity is not None:
            return max(self.epsilon, float(exposure.propensity))
        return self.epsilon
