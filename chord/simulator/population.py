"""Synthetic population placed in opinion space (Appendix C.4).

A population of agents with latent opinion positions ``x_u`` (the ground truth
the estimator must recover) plus a per-user reactivity (how readily they react)
and an optional selectivity (how discriminating — feeds the §5 quality story).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Agent:
    """A simulated user."""

    id: int
    opinion: np.ndarray          # latent x_u in opinion space
    reactivity: float = 1.0      # scales reaction probability
    selectivity: float = 1.0     # >1 = discriminating; <1 = indiscriminate scroller
    cluster: int = 0             # ground-truth cluster label


@dataclass
class Population:
    agents: List[Agent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.agents)

    def opinions(self):
        return {a.id: a.opinion for a in self.agents}


def make_bipolar_population(
    n: int,
    d: int = 2,
    separation: float = 2.0,
    frac_indiscriminate: float = 0.2,
    seed: int = 0,
) -> Population:
    """A two-cluster (bipolar) population — the canonical divide (Appendix C).

    Half the agents sit near ``+separation`` on axis 0, half near ``-separation``;
    a ``frac_indiscriminate`` fraction are low-selectivity scrollers whose
    reactions carry near-zero information (§2 principle 2, §5).
    """
    rng = np.random.default_rng(seed)
    agents: List[Agent] = []
    for i in range(n):
        cluster = 0 if i < n // 2 else 1
        center = np.zeros(d)
        center[0] = separation if cluster == 0 else -separation
        opinion = center + rng.normal(0, 0.5, size=d)
        indiscriminate = rng.random() < frac_indiscriminate
        agents.append(
            Agent(
                id=i,
                opinion=opinion,
                reactivity=float(rng.uniform(0.7, 1.3)),
                selectivity=0.3 if indiscriminate else float(rng.uniform(1.0, 2.0)),
                cluster=cluster,
            )
        )
    return Population(agents)
