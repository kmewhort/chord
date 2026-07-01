"""Exploration-anchored reception cap — collusion audit via the §6.2 anchor.

The exploration pool (§8) floors a small fraction of *uniform-random* exposures: an
unconfounded, known-propensity logging policy that no manipulator can target
(Bottou et al. 2013). So the reception a post earns among **exploration-exposed**
users is an estimate of its *true* reception, independent of how many puppets boost
it organically. A distributed sybil ring inflates a post's organic reception (a
common-mode lift of the shared intercepts, §13.10); capping each cluster's
reconstructed reception at the upper confidence bound of the exploration-anchored
reception discards that manufactured surplus — and the bound contains no ``K``, so
the ring's reach saturates in ring size.

Because exploration traffic is thin at small ε, the anchor is pooled per **author**
(across its posts and windows, with decay) and shrunk toward a neutral prior with a
generous width, so it *misses* (never binds) when evidence is scarce — the safe
direction — and only bites once the exploration sample is large enough to prove the
organic reception is inflated. James–Stein / Agresti–Coull shrinkage; a Hoeffding
upper bound for the width.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..types import Exposure, ExposureSource, Id, Reaction


class ExplorationAnchor:
    """Rolling per-author reception from uniform-random (exploration) exposures."""

    def __init__(self, decay: float = 0.6, prior_n: float = 3.0, z: float = 1.0,
                 sigma: float = 0.5):
        self.decay = decay
        self.prior_n = prior_n
        self.z = z
        self.sigma = sigma
        self._sum: Dict[Id, float] = defaultdict(float)
        self._n: Dict[Id, float] = defaultdict(float)

    def update(self, reactions: Sequence[Reaction], exposures: Sequence[Exposure],
               post_authors: Mapping[Id, Id]) -> None:
        explore: Set[Tuple[Id, Id]] = {
            (e.user_id, e.post_id) for e in exposures
            if e.source is ExposureSource.EXPLORATION
        }
        if not explore and not self._n:
            return
        for a in self._sum:
            self._sum[a] *= self.decay
            self._n[a] *= self.decay
        for rx in reactions:
            if (rx.user_id, rx.post_id) in explore:
                a = post_authors.get(rx.post_id)
                if a is not None:
                    self._sum[a] += rx.value
                    self._n[a] += 1.0

    def ucb(self, author: Id) -> float:
        """Upper confidence bound on ``author``'s true reception; +inf if no evidence."""
        n = self._n.get(author, 0.0)
        if n < 1.0:
            return float("inf")   # no exploration evidence yet ⇒ never cap (safe)
        mean = self._sum[author] / (n + self.prior_n)          # shrunk toward 0
        width = self.z * self.sigma / np.sqrt(n + self.prior_n)
        return float(mean + width)

    def caps(self, post_authors: Mapping[Id, Id]) -> Dict[Id, float]:
        """Per-post reception cap = its author's exploration UCB."""
        return {pid: self.ucb(a) for pid, a in post_authors.items()}
