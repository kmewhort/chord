"""Stability as a monitored runtime property (§9.3).

Because global convergence is *not* guaranteed in the nonconvex regime, the right
target is a **bounded stationary regime**. We run a controller on the estimator's
own concentration: track the effective rater count ``(sum lambda)^2 / sum
lambda^2`` (or ``Gini(lambda)``); if it collapses (concentration climbs),
automatically raise the teleport floor ``delta`` and the damping, and lift
``epsilon_min``. The exploration pool is load-bearing four times over — provider
fairness, cold-start, causal identification, and estimator stability.

Also implements the endogenous/exogenous shift separation (§9.3): drift measured
in the non-personalized exploration slice estimates the *exogenous* background
shift (breaking news), while drift in the personalized stream beyond that baseline
is attributable to the ranker's own feedback loop (endogenous, to damp).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .config import ChordConfig
from .types import Id


def effective_rater_count(rater_lambda: Mapping[Id, float]) -> float:
    """N_eff = (sum lambda)^2 / sum lambda^2 (§9.3).

    Equals the number of raters when weight is uniform, and collapses toward 1
    as a few raters dominate — the concentration signal to watch.
    """
    lam = np.array(list(rater_lambda.values()), dtype=float)
    if lam.size == 0:
        return 0.0
    denom = float(np.sum(lam ** 2))
    if denom <= 0:
        return 0.0
    return float((lam.sum() ** 2) / denom)


def gini(rater_lambda: Mapping[Id, float]) -> float:
    """Gini coefficient of the rater weights (§9.3). 0 = equal, ->1 = concentrated."""
    lam = np.sort(np.array(list(rater_lambda.values()), dtype=float))
    n = lam.size
    if n == 0:
        return 0.0
    s = lam.sum()
    if s <= 0:
        return 0.0
    # Gini via the ordered formulation.
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * lam) / (n * s)) - (n + 1.0) / n)


@dataclass
class ControllerState:
    """Mutable stability parameters the controller tunes (§9.3)."""

    eigentrust_delta: float
    epsilon_min: float
    budget_eta: float
    history: List[dict] = field(default_factory=list)


class ConcentrationController:
    """The §9.3 runtime guard keeping the coupled estimator bounded."""

    def __init__(self, config: ChordConfig):
        self.config = config
        self.state = ControllerState(
            eigentrust_delta=config.eigentrust_delta,
            epsilon_min=config.epsilon_min,
            budget_eta=config.budget_eta,
        )

    def step(self, rater_lambda: Mapping[Id, float]) -> ControllerState:
        """Observe rater concentration; tighten if it exceeds the ceiling.

        If ``Gini(lambda)`` climbs above ``gini_ceiling``, raise the teleport
        floor (lower ``delta`` toward more teleport → flatter fixed point) and
        lift ``epsilon_min`` so the system keeps sampling regions it stopped
        showing (persistent excitation). When concentration is healthy, relax
        gently back toward the configured defaults.
        """
        cfg = self.config
        g = gini(rater_lambda)
        n_eff = effective_rater_count(rater_lambda)

        if g > cfg.gini_ceiling:
            # Raise the teleport floor => reduce delta (more uniform teleport).
            self.state.eigentrust_delta = max(
                0.5, self.state.eigentrust_delta - cfg.controller_delta_step
            )
            self.state.epsilon_min = min(
                cfg.epsilon_max, self.state.epsilon_min + cfg.controller_epsilon_step
            )
        else:
            # relax back toward defaults, but never below the configured floor
            self.state.eigentrust_delta = min(
                cfg.eigentrust_delta,
                self.state.eigentrust_delta + 0.5 * cfg.controller_delta_step,
            )
            self.state.epsilon_min = max(
                cfg.epsilon_min,
                self.state.epsilon_min - 0.5 * cfg.controller_epsilon_step,
            )

        self.state.history.append({"gini": g, "n_eff": n_eff,
                                    "delta": self.state.eigentrust_delta,
                                    "epsilon_min": self.state.epsilon_min})
        return self.state


def endogenous_shift(
    exploration_drift: float,
    personalized_drift: float,
) -> float:
    """Separate endogenous (loop) shift from exogenous (news) shift (§9.3).

    The exploration slice is not driven by the personalized ranker, so its
    distributional drift estimates the exogenous background. Drift in the
    personalized stream *beyond* that baseline is attributable to the loop:

        endogenous = max(0, personalized_drift - exploration_drift)

    Feed this residual (not the raw personalized drift) to the damping controller,
    so a genuine news event is tracked rather than suppressed.
    """
    return max(0.0, float(personalized_drift) - float(exploration_drift))
