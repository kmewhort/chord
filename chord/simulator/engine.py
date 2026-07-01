"""The closed-loop simulator (Appendix C.4) — the only way to exercise §9.

The feedback loop is untestable on fixed data (a static archive cannot respond to
the ranker's own allocations). This engine runs the full loop:

    rank -> simulated reactions -> retrain

over an adapting population and adapting author-agents, and measures the §9
targets: convergence to a performatively stable point vs oscillation; the
effective-rater-count / Gini controller holding concentration bounded; and whether
exploration at rate epsilon sustains the identifiability anchor over time.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import ChordConfig, UserKnobs
from ..loop import Chord
from ..monitor import effective_rater_count, gini
from ..propensity import LoggedPropensityModel
from ..types import Exposure, ExposureSource, Post, Reaction, ReactionKind
from .content import AuthorAgent, make_authors, reset_truth, true_loading
from .population import Population, make_bipolar_population
from .response import react


@dataclass
class WindowMetrics:
    window: int
    n_posts: int
    n_reactions: int
    gini_lambda: float
    n_eff: float
    exploration_rate: float
    mean_bridge_score: float
    firehose_reach_per_post: float
    universal_reach_per_post: float
    score_drift: float  # performative stability signal (change in feed scores)


@dataclass
class SimulationResult:
    metrics: List[WindowMetrics] = field(default_factory=list)

    def score_drifts(self) -> List[float]:
        return [m.score_drift for m in self.metrics]

    def is_stable(self, tail: int = 3, tol: float = 0.15) -> bool:
        """Heuristic: the feed-score drift has settled in the last ``tail`` windows."""
        drifts = [m.score_drift for m in self.metrics[-tail:]]
        return len(drifts) > 0 and max(drifts) < tol


class Simulator:
    """Agent-based closed-loop simulator for CHORD (Appendix C.4)."""

    def __init__(
        self,
        config: Optional[ChordConfig] = None,
        n_users: int = 40,
        d: int = 2,
        knobs: Optional[UserKnobs] = None,
        n_slots: int = 6,
        seed: int = 0,
    ):
        # Default sim config uses a small per-author budget so the conserved-
        # budget mechanism (§8) actually binds within a feed and firehose
        # dilution is observable.
        self.config = config or ChordConfig(
            d=d, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0
        )
        self.d = d
        self.n_slots = n_slots
        self.knobs = knobs or UserKnobs(M=1.0)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.population: Population = make_bipolar_population(n_users, d=d, seed=seed)
        self.authors: List[AuthorAgent] = make_authors(d=d, seed=seed)
        reset_truth()
        self.chord = Chord(
            self.config,
            propensity_model=LoggedPropensityModel(self.config.epsilon_min),
            seed=seed,
        )
        self._prev_scores: Dict[int, float] = {}

    def run(self, n_windows: int = 8) -> SimulationResult:
        result = SimulationResult()
        live_posts: List[Post] = []  # posts still in circulation (recent windows)
        for w in range(n_windows):
            new_posts = []
            for author in self.authors:
                new_posts.extend(author.generate(w, self._rng))
            # candidate pool = new posts + a memory of the last window's posts
            live_posts = new_posts + live_posts[: 3 * len(new_posts)]
            post_map = {p.id: p for p in live_posts}

            reactions: List[Reaction] = []
            exposures: List[Exposure] = []
            reach: Dict[int, int] = defaultdict(int)  # author -> exposures
            posts_by_author: Dict[int, int] = defaultdict(int)
            for p in new_posts:
                posts_by_author[p.author_id] += 1

            drift_samples: List[float] = []
            n_explore = 0
            n_exposure_total = 0

            for agent in self.population.agents:
                feed_sourced = self._serve_with_source(agent, live_posts, w)
                feed_ids = [pid for pid, _ in feed_sourced]
                # baseline exploration: inject uniform-random exposures at the
                # floor rate — the unconfounded anchor (§6.2).
                explore_extra = self._exploration_exposures(live_posts, feed_ids)
                shown = feed_sourced + explore_extra

                for pid, source in shown:
                    post = post_map[pid]
                    loading = true_loading(pid)
                    if loading is None:
                        continue
                    kind = react(agent, loading, self._rng)
                    pi = (self.config.epsilon_min
                          if source is ExposureSource.EXPLORATION else 0.5)
                    exposures.append(
                        Exposure(agent.id, pid, timestamp=float(w),
                                 source=source, propensity=pi)
                    )
                    reach[post.author_id] += 1
                    n_exposure_total += 1
                    if source is ExposureSource.EXPLORATION:
                        n_explore += 1
                    if kind is not None:
                        reactions.append(self._to_reaction(agent.id, pid, kind, w))

                # performative drift: change in this agent's top feed score
                cur = self._top_score(agent, feed_ids)
                if agent.id in self._prev_scores:
                    drift_samples.append(abs(cur - self._prev_scores[agent.id]))
                self._prev_scores[agent.id] = cur

            # retrain on the window
            if reactions:
                st = self.chord.fit_window(reactions, post_map, exposures)
                bridge_scores = [v for v in st.bridging.b_lcb.values() if np.isfinite(v)]
                g = gini(st.rater_lambda_eff)
                neff = effective_rater_count(st.rater_lambda_eff)
            else:
                bridge_scores, g, neff = [], 0.0, 0.0

            # author reach feedback (reach per post) -> drives adaptivity
            for author in self.authors:
                np_posts = max(1, posts_by_author.get(author.id, 1))
                author.realized_reach = reach.get(author.id, 0) / np_posts

            firehose = next(a for a in self.authors if a.id == 1003)
            universal = next(a for a in self.authors if a.id == 1000)

            result.metrics.append(
                WindowMetrics(
                    window=w,
                    n_posts=len(new_posts),
                    n_reactions=len(reactions),
                    gini_lambda=g,
                    n_eff=neff,
                    exploration_rate=(n_explore / n_exposure_total) if n_exposure_total else 0.0,
                    mean_bridge_score=float(np.mean(bridge_scores)) if bridge_scores else 0.0,
                    firehose_reach_per_post=firehose.realized_reach,
                    universal_reach_per_post=universal.realized_reach,
                    score_drift=float(np.mean(drift_samples)) if drift_samples else 0.0,
                )
            )
        return result

    # ------------------------------------------------------------- helpers
    def _serve(self, agent, live_posts, window) -> List[str]:
        if self.chord.state is None:
            # cold start: chronological (most recent first)
            ordered = sorted(live_posts, key=lambda p: -p.created_at)
            return [p.id for p in ordered[: self.n_slots]]
        return self.chord.rank(agent.id, live_posts, self.knobs, n_slots=self.n_slots)

    def _exploration_exposures(self, live_posts, feed):
        """Uniform-random exposures at the floor rate (the identifiability anchor).

        Uses stochastic rounding so the *expected* number of random exposures is
        ``epsilon_min * n_slots`` even when that is below 1 — otherwise a small
        feed would round the anchor away and silently lose positivity (§6.2).
        """
        eps = self.config.epsilon_min
        expected = eps * self.n_slots
        k = int(np.floor(expected))
        if self._rng.random() < (expected - k):
            k += 1
        feed_set = set(feed)
        pool = [p.id for p in live_posts if p.id not in feed_set]
        if not pool or k == 0:
            return []
        idx = self._rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [(pool[i], ExposureSource.EXPLORATION) for i in np.atleast_1d(idx)]

    def _serve_with_source(self, agent, live_posts, window):
        feed = self._serve(agent, live_posts, window)
        return [(pid, ExposureSource.ORGANIC) for pid in feed]

    def _to_reaction(self, uid, pid, kind: ReactionKind, window) -> Reaction:
        from ..types import DEFAULT_REACTION_VALUES
        val = DEFAULT_REACTION_VALUES[kind]
        if kind is ReactionKind.EXPOSED_NO_REACTION:
            val = -abs(self.config.exposed_no_reaction_c)
        return Reaction(uid, pid, float(val), kind=kind, timestamp=float(window) + self._rng.random())

    def _top_score(self, agent, feed) -> float:
        if self.chord.state is None or not feed:
            return 0.0
        st = self.chord.state
        return float(st.bridging.b_lcb.get(feed[0], 0.0)) if np.isfinite(
            st.bridging.b_lcb.get(feed[0], float("-inf"))) else 0.0
