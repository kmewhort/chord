"""The randomization portfolio — allocating the scarce ε-slice by information value (§13).

Randomized (unconfounded) exposure is CHORD's one source of ground truth, and it is
load-bearing many times over: causal **identification** (§6.2), **cold-start**/provider
fairness (§8), estimator **stability** excitation (§9.3), plus every honest answer to the
open problems — confounding **calibration** (E2), collusion/ring **audit** (E5/E12),
performativity **probes** (E1), and amplification-collar re-certification (E3). These all
draw on the *same* floored budget ε, so it is not just a floor but a scarce resource that
needs an explicit **allocation policy**.

This is a bandit over information value. Each demand ("arm") reports the value realized
from the ε it received; the portfolio shifts the budget toward high-value arms — but keeps
a **floor** on every arm, because ε being a floored system invariant is precisely what
makes identification, fairness and stability hold. So: floor first (never starve an arm to
zero), then water-fill the remainder by estimated value. The whole policy is one number
per arm plus a softmax — cheaper than any single use it schedules.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


class RandomizationPortfolio:
    """Allocate a fixed ε budget across named demands by (learned) information value."""

    def __init__(self, arms: Sequence[str], floor_frac: float = 0.4,
                 lr: float = 0.35, temperature: float = 1.0):
        if not arms:
            raise ValueError("need at least one arm")
        if not 0.0 <= floor_frac <= 1.0:
            raise ValueError("floor_frac must be in [0,1]")   # 1.0 = pure even split (uniform)
        self.arms = list(arms)
        self.floor_frac = floor_frac          # total mass guaranteed to the floor, split evenly
        self.lr = lr
        self.temperature = max(temperature, 1e-6)
        self.value: Dict[str, float] = {a: 1.0 for a in self.arms}  # EWMA marginal value

    def allocate(self) -> Dict[str, float]:
        """Budget shares (sum to 1): an even floor to every arm + a value-weighted remainder.

        The floor keeps ε a floored invariant *per demand* (identification never starves);
        the remainder is a softmax over estimated value (water-filling toward the arm whose
        marginal information value is currently highest — e.g. audit during an attack)."""
        n = len(self.arms)
        base = self.floor_frac / n
        v = np.array([self.value[a] for a in self.arms], dtype=float)
        z = (v - v.max()) / self.temperature
        soft = np.exp(z)
        soft = soft / soft.sum()
        return {a: base + (1.0 - self.floor_frac) * float(soft[i])
                for i, a in enumerate(self.arms)}

    def observe(self, arm: str, realized_value: float) -> None:
        """Update the arm's estimated marginal information value (EWMA)."""
        if arm in self.value:
            self.value[arm] = (1 - self.lr) * self.value[arm] + self.lr * float(realized_value)

    def update(self, realized: Mapping[str, float]) -> None:
        for arm, val in realized.items():
            self.observe(arm, val)
