"""Pluggable rankers so the simulator can run counterfactuals (Appendix C.4).

Each ranker runs its *own* closed loop on the same seeded world, so the comparison
is a genuine "what if we had ranked this way" — feeds diverge, and so do the
reaction logs each ranker then learns from. The interface is deliberately small:

    rank(user, candidates, n_slots, window) -> [post_id, ...]     # serving plane
    observe(reactions, posts, exposures, window) -> None          # learning plane

* **ChordRanker** — the full CHORD loop (fit_window → rank), bridging value + budget
  + exploration anchor.
* **EngagementRanker** — the honest baseline: the *same* biased MF, but rank by
  predicted *personalized* reception (maximize this user's approval), no bridging,
  no budget, no exploration. This is what "optimize engagement" means.
* **ChronologicalRanker / RandomRanker** — reverse-chron and random floors.
* **OracleRanker** — cheats: ranks by the hidden true value (quality × bridged
  reception). The achievable ceiling.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import ChordConfig, UserKnobs
from ..loop import Chord
from ..model import MatrixFactorization
from ..monitor import effective_rater_count, gini
from ..propensity import LoggedPropensityModel
from ..types import Id, Post
from .metrics import Welfare


class Ranker:
    name = "base"
    wants_exploration = False

    def rank(self, user_id: Id, candidates: List[Post], n_slots: int, window: int) -> List[Id]:
        raise NotImplementedError

    def observe(self, reactions, posts, exposures, window: int) -> None:
        pass

    def estimated_opinions(self) -> Optional[Dict[Id, np.ndarray]]:
        return None

    def diagnostics(self) -> Dict[str, float]:
        return {}


def _chronological(candidates: List[Post], n_slots: int) -> List[Id]:
    return [p.id for p in sorted(candidates, key=lambda p: -p.created_at)[:n_slots]]


class ChordRanker(Ranker):
    name = "chord"
    wants_exploration = True

    def __init__(self, config: ChordConfig, knobs: UserKnobs, seed: int = 0):
        self.config = config
        self.knobs = knobs
        self.chord = Chord(config, propensity_model=LoggedPropensityModel(config.epsilon_min),
                           seed=seed)

    def rank(self, user_id, candidates, n_slots, window):
        if self.chord.state is None:
            return _chronological(candidates, n_slots)
        return self.chord.rank(user_id, candidates, self.knobs, n_slots=n_slots)

    def observe(self, reactions, posts, exposures, window):
        if reactions:
            self.chord.fit_window(reactions, posts, exposures)

    def estimated_opinions(self):
        st = self.chord.state
        return dict(st.result.x_user) if st is not None else None

    def diagnostics(self):
        st = self.chord.state
        if st is None:
            return {}
        b = [v for v in st.bridging.b_lcb.values() if np.isfinite(v)]
        return {
            "gini_lambda": gini(st.rater_lambda_eff),
            "n_eff": effective_rater_count(st.rater_lambda_eff),
            "mean_b_lcb": float(np.mean(b)) if b else 0.0,
        }


class EngagementRanker(Ranker):
    """Maximize predicted personalized reception — the engagement baseline."""

    name = "engagement"
    wants_exploration = False

    def __init__(self, config: ChordConfig, seed: int = 0):
        self.config = config
        self.seed = seed
        self._result = None

    def rank(self, user_id, candidates, n_slots, window):
        res = self._result
        if res is None:
            return _chronological(candidates, n_slots)
        scored = [(self._predict(res, user_id, p), p.id) for p in candidates]
        scored.sort(key=lambda t: -t[0])
        return [pid for _, pid in scored[:n_slots]]

    def observe(self, reactions, posts, exposures, window):
        if reactions:
            self._result = MatrixFactorization(self.config, seed=self.seed).fit(reactions, posts)

    @staticmethod
    def _predict(res, uid, post: Post) -> float:
        x = res.x_user.get(uid)
        y = res.y_post.get(post.id)
        dot = float(np.dot(x, y)) if (x is not None and y is not None) else 0.0
        return (res.mu + res.b_user.get(uid, 0.0) + res.b_author.get(post.author_id, 0.0)
                + res.b_post.get(post.id, 0.0) + dot)

    def estimated_opinions(self):
        return dict(self._result.x_user) if self._result is not None else None


class ChronologicalRanker(Ranker):
    name = "chronological"

    def rank(self, user_id, candidates, n_slots, window):
        return _chronological(candidates, n_slots)


class RandomRanker(Ranker):
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def rank(self, user_id, candidates, n_slots, window):
        idx = self._rng.permutation(len(candidates))[:n_slots]
        return [candidates[i].id for i in idx]


class OracleRanker(Ranker):
    """Ranks by the hidden true value (quality × bridged reception). The ceiling."""

    name = "oracle"

    def __init__(self, welfare: Welfare, true_post_fn):
        self.welfare = welfare
        self._true_post = true_post_fn

    def rank(self, user_id, candidates, n_slots, window):
        scored = []
        for p in candidates:
            t = self._true_post(p.id)
            v = self.welfare.true_value(t) if t is not None else -1.0
            scored.append((v, p.id))
        scored.sort(key=lambda t: -t[0])
        return [pid for _, pid in scored[:n_slots]]
