"""Ground-truth welfare, recovery, and regret metrics for the simulator.

Because the world is synthetic we *know* every hidden quantity, so we can score a
ranker on things no live system can observe: the true bridged value it delivers,
the toxicity/polarization it exposes people to, how well the estimator recovered
the latent opinions, and how far its allocation sits from an oracle that ranks by
true value. These are what turn "the loop runs and stays bounded" into "here is
what the loop actually *does*, versus the alternatives."
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .content import PostTruth
from .population import Population
from .response import expected_approval


class Welfare:
    """Precomputes cluster membership once, then scores posts against the truth."""

    def __init__(self, population: Population):
        self.pop = population
        self.clusters = sorted({a.cluster for a in population.agents})
        self._by_cluster = {
            c: [a for a in population.agents if a.cluster == c] for c in self.clusters
        }

    def cluster_mean_approval(self, truth: PostTruth) -> np.ndarray:
        """Expected approval of ``truth`` within each opinion cluster."""
        return np.array([
            float(np.mean([expected_approval(a, truth) for a in self._by_cluster[c]]))
            for c in self.clusters
        ])

    def bridged_reception(self, truth: PostTruth) -> float:
        """Worst-cluster expected approval — the true B_LCB the oracle knows."""
        return float(np.min(self.cluster_mean_approval(truth)))

    def divisiveness(self, truth: PostTruth) -> float:
        """Spread of reception across clusters — realized affective polarization."""
        m = self.cluster_mean_approval(truth)
        return float(np.max(m) - np.min(m))

    def true_value(self, truth: PostTruth) -> float:
        """Genuine worth: quality × how broadly the post is actually received.

        High only when a post is *both* good and bridging — so bridging-bait (broad
        but shallow) and toxic partisan bait (engaging but divisive) both score low.
        """
        return float(truth.quality * self.bridged_reception(truth))


def embedding_recovery(estimated: Dict, true_opinions: Dict) -> float:
    """Rotation/scale-free recovery: correlation of pairwise user distances.

    Works even when the estimator's dimension differs from the true one (§13.4):
    if the recovered geometry preserves who-is-near-whom, the pairwise-distance
    matrices correlate. 1.0 = geometry perfectly recovered; ~0 = no structure.
    """
    ids = [u for u in true_opinions if u in estimated]
    if len(ids) < 4:
        return float("nan")
    T = np.stack([np.asarray(true_opinions[u], float) for u in ids])
    E = np.stack([np.asarray(estimated[u], float) for u in ids])
    dt = _pdist(T)
    de = _pdist(E)
    if dt.std() < 1e-12 or de.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(dt, de)[0, 1])


def _pdist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(X), k=1)
    return d[iu]


def mean_or_nan(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")
