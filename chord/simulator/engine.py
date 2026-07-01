"""The closed-loop simulator (Appendix C.4) — the only way to exercise §9.

The feedback loop is untestable on fixed data (a static archive cannot respond to
the ranker's own allocations). This engine runs the full loop

    rank -> simulated reactions -> retrain

over an adapting population and adapting author-agents, for a *pluggable ranker*
(:mod:`chord.simulator.rankers`), and measures both the §9 dynamics (concentration,
exploration anchor, performative stability) and — because the world is synthetic —
the **ground-truth welfare** each ranker actually delivers (:mod:`chord.simulator.metrics`).
Running several rankers on the same seeded world is the counterfactual that answers
"does bridging beat engagement?".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from ..config import ChordConfig, UserKnobs
from ..types import DEFAULT_REACTION_VALUES, Exposure, ExposureSource, Post, Reaction, ReactionKind
from .content import AuthorAgent, make_authors, reset_truth, true_post
from .metrics import Welfare, embedding_recovery, mean_or_nan
from .population import Agent, Population, make_bipolar_population
from .rankers import (
    ChordRanker, ChronologicalRanker, EngagementRanker, OracleRanker, RandomRanker, Ranker,
)
from .response import expected_approval, react


@dataclass
class WindowMetrics:
    window: int
    n_posts: int
    n_reactions: int
    exploration_rate: float
    # CHORD-only diagnostics (nan/0 for other rankers)
    gini_lambda: float
    n_eff: float
    mean_bridge_score: float
    # ground-truth welfare of what was shown, per impression
    true_value: float          # quality × bridged reception delivered
    toxicity: float            # affective-polarization exposure
    divisiveness: float        # realized cross-cluster spread of shown posts
    satisfaction: float        # viewer's expected approval of shown posts
    recovery: float            # estimator's recovery of the true opinion geometry
    feed_churn: float          # performative stability (1 - Jaccard vs last window)
    # ecosystem-level: what authors *produce* (captures performative adaptation)
    content_divisiveness: float = float("nan")
    content_true_value: float = float("nan")
    partisan_extremity: float = float("nan")   # mean |style·axis0| of partisan authors
    ring_target_blcb: float = float("nan")      # bridged support the sybil ring bought
    reach_by_label: Dict[str, float] = field(default_factory=dict)

    @property
    def ring_target_reach(self) -> float:
        return self.reach_by_label.get("ring_target", float("nan"))

    # back-compat alias
    @property
    def firehose_reach_per_post(self) -> float:
        return self.reach_by_label.get("firehose", 0.0)

    @property
    def universal_reach_per_post(self) -> float:
        return self.reach_by_label.get("universal", 0.0)

    @property
    def score_drift(self) -> float:
        return self.feed_churn


@dataclass
class SimulationResult:
    metrics: List[WindowMetrics] = field(default_factory=list)

    def tail(self, key: str, n: int = 4) -> float:
        return mean_or_nan(getattr(m, key) for m in self.metrics[-n:])

    def series(self, key: str) -> List[float]:
        return [getattr(m, key) for m in self.metrics]

    def is_stable(self, tail: int = 3, tol: float = 0.5) -> bool:
        churn = [m.feed_churn for m in self.metrics[-tail:]]
        return len(churn) > 0 and max(churn) < tol


class Simulator:
    """Agent-based closed-loop simulator for CHORD (Appendix C.4).

    One ``Simulator`` holds the *world spec*; ``run(ranker)`` executes one full
    closed loop for a ranker (rebuilding a fresh, identically-seeded world each
    call), and ``compare(...)`` runs several rankers on that same world.
    """

    def __init__(
        self,
        config: Optional[ChordConfig] = None,
        n_users: int = 40,
        d: int = 2,
        knobs: Optional[UserKnobs] = None,
        n_slots: int = 6,
        seed: int = 0,
        d_true: Optional[int] = None,
        adaptive_authors: bool = True,
        include_toxic_and_bait: bool = True,
        performativity: float = 0.0,
        sybil_ring_size: int = 0,
        ring_target_quality: float = 0.4,
        ring_mode: str = "naive",
        ring_camouflage: int = 4,
    ):
        # small per-author budget so the conserved-budget mechanism (§8) binds.
        self.config = config or ChordConfig(
            d=d, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0
        )
        self.d = d
        self.d_true = d_true if d_true is not None else d + 1  # one hidden axis by default
        self.n_users = n_users
        self.n_slots = n_slots
        self.knobs = knobs or UserKnobs(M=1.0)
        self.seed = seed
        self.adaptive_authors = adaptive_authors
        self.include_toxic_and_bait = include_toxic_and_bait
        self.performativity = performativity
        self.sybil_ring_size = sybil_ring_size
        self.ring_target_quality = ring_target_quality
        self.ring_mode = ring_mode              # "naive" | "distributed" (camouflaged)
        self.ring_camouflage = ring_camouflage  # camouflage reactions per sybil per window

    # ---------------------------------------------------------------- world
    def _fresh_world(self):
        pop = make_bipolar_population(self.n_users, d=self.d, seed=self.seed,
                                      d_true=self.d_true)
        authors = make_authors(d=self.d, seed=self.seed, d_true=self.d_true,
                               include_toxic_and_bait=self.include_toxic_and_bait,
                               performativity=self.performativity)
        if not self.adaptive_authors:
            for a in authors:
                a.adaptivity = 0.0
        ring = None
        if self.sybil_ring_size > 0:
            # A mediocre-quality target author near the origin (so it *could* look
            # bridging if its reception were inflated), boosted by a ring of
            # single-target sybil raters — the exact §5 attack, now in the loop.
            target = AuthorAgent(id=2000, style=np.zeros(self.d_true), spread=0.2,
                                 adaptivity=0.0, base_volume=1, max_volume=2,
                                 toxicity=0.1, quality=self.ring_target_quality,
                                 label="ring_target")
            authors = authors + [target]
            sybil_ids = [90000 + i for i in range(self.sybil_ring_size)]
            ring = {"target_id": 2000, "sybil_ids": sybil_ids, "sybil_agents": {}}
            if self.ring_mode == "distributed":
                # Camouflaged ring: each puppet borrows a genuine agent's opinion so
                # the MF places it inside a *real* cluster; the ring spreads its
                # puppets evenly across clusters, then all boost the target — faking
                # cross-cluster support to beat B_LCB's min-over-clusters.
                by_cluster = defaultdict(list)
                for a in pop.agents:
                    by_cluster[a.cluster].append(a)
                clusters = sorted(by_cluster)
                for i, sid in enumerate(sybil_ids):
                    c = clusters[i % len(clusters)]
                    host = by_cluster[c][i % len(by_cluster[c])]
                    ring["sybil_agents"][sid] = Agent(
                        id=sid, opinion=host.opinion.copy(),
                        reactivity=1.0, selectivity=1.5, cluster=c)
        reset_truth()
        return pop, authors, ring

    def make_ranker(self, spec: Union[str, Ranker], population: Population) -> Ranker:
        if isinstance(spec, Ranker):
            return spec
        name = spec
        if name == "chord":
            return ChordRanker(self.config, self.knobs, seed=self.seed)
        if name == "engagement":
            return EngagementRanker(self.config, seed=self.seed)
        if name == "chronological":
            return ChronologicalRanker()
        if name == "random":
            return RandomRanker(seed=self.seed)
        if name == "oracle":
            return OracleRanker(Welfare(population), true_post)
        raise ValueError(f"unknown ranker {name!r}")

    # ------------------------------------------------------------------ run
    def run(self, ranker: Union[str, Ranker] = "chord", n_windows: int = 8) -> SimulationResult:
        population, authors, ring = self._fresh_world()
        welfare = Welfare(population)
        r = self.make_ranker(ranker, population)
        true_opinions = {a.id: a.opinion for a in population.agents}

        content_rng = np.random.default_rng(self.seed)         # world/content stream
        react_rng = np.random.default_rng(self.seed + 1)       # reaction/exploration stream

        result = SimulationResult()
        live_posts: List[Post] = []
        prev_feeds: Dict[int, set] = {}

        for w in range(n_windows):
            new_posts: List[Post] = []
            for author in authors:
                new_posts.extend(author.generate(w, content_rng))
            live_posts = new_posts + live_posts[: 3 * len(new_posts)]
            post_map = {p.id: p for p in live_posts}

            # ecosystem: divisiveness/value of everything produced this window
            new_truths = [true_post(p.id) for p in new_posts]
            new_truths = [t for t in new_truths if t is not None]
            content_div = mean_or_nan(welfare.divisiveness(t) for t in new_truths)
            content_val = mean_or_nan(welfare.true_value(t) for t in new_truths)

            reactions: List[Reaction] = []
            exposures: List[Exposure] = []
            reach: Dict[int, int] = defaultdict(int)
            posts_by_author: Dict[int, int] = defaultdict(int)
            for p in new_posts:
                posts_by_author[p.author_id] += 1

            n_explore = n_impr = 0
            churn_samples: List[float] = []
            v_sum = tox_sum = div_sum = sat_sum = 0.0

            for agent in population.agents:
                feed = r.rank(agent.id, live_posts, self.n_slots, w)
                shown = [(pid, ExposureSource.ORGANIC) for pid in feed]
                if r.wants_exploration:
                    shown += self._exploration(live_posts, set(feed), react_rng)

                for pid, source in shown:
                    truth = true_post(pid)
                    if truth is None:
                        continue
                    post = post_map[pid]
                    kind = react(agent, truth, react_rng)
                    pi = (self.config.epsilon_min
                          if source is ExposureSource.EXPLORATION else 0.5)
                    exposures.append(Exposure(agent.id, pid, timestamp=float(w),
                                              source=source, propensity=pi))
                    reach[post.author_id] += 1
                    n_impr += 1
                    if source is ExposureSource.EXPLORATION:
                        n_explore += 1
                    # ground-truth welfare accounting (per impression)
                    v_sum += welfare.true_value(truth)
                    tox_sum += truth.toxicity
                    div_sum += welfare.divisiveness(truth)
                    sat_sum += expected_approval(agent, truth)
                    if kind is not None:
                        reactions.append(self._reaction(agent.id, pid, kind, w, react_rng))

                cur = set(feed)
                if agent.id in prev_feeds and (prev_feeds[agent.id] or cur):
                    inter = len(cur & prev_feeds[agent.id])
                    union = len(cur | prev_feeds[agent.id])
                    churn_samples.append(1.0 - inter / union if union else 0.0)
                prev_feeds[agent.id] = cur

            # Adversary: a single-target sybil ring boosts the target's posts. These
            # are the reactions the ranker then learns from — the §5 attack in-loop.
            target_new = []
            if ring is not None:
                target_new = [p for p in new_posts if p.author_id == ring["target_id"]]
                sybil_agents = ring.get("sybil_agents", {})
                for spid in ring["sybil_ids"]:
                    # Distributed ring: camouflage by rating genuine content like the
                    # cluster the puppet is hiding in, so the MF places it there.
                    sa = sybil_agents.get(spid)
                    if sa is not None and self.ring_camouflage > 0:
                        pool = [p for p in live_posts if p.author_id != ring["target_id"]]
                        if pool:
                            idx = react_rng.choice(len(pool),
                                                   size=min(self.ring_camouflage, len(pool)),
                                                   replace=False)
                            for j in np.atleast_1d(idx):
                                cp = pool[int(j)]
                                ct = true_post(cp.id)
                                if ct is None:
                                    continue
                                kind = react(sa, ct, react_rng)
                                exposures.append(Exposure(spid, cp.id, timestamp=float(w),
                                                          source=ExposureSource.ORGANIC, propensity=0.5))
                                if kind is not None:
                                    reactions.append(self._reaction(spid, cp.id, kind, w, react_rng))
                    # the attack itself: boost the target's posts
                    for tp in target_new:
                        exposures.append(Exposure(spid, tp.id, timestamp=float(w),
                                                  source=ExposureSource.ORGANIC, propensity=0.5))
                        reactions.append(Reaction(spid, tp.id,
                                                  DEFAULT_REACTION_VALUES[ReactionKind.BOOST],
                                                  kind=ReactionKind.BOOST,
                                                  timestamp=float(w) + react_rng.random()))

            r.observe(reactions, post_map, exposures, w)

            ring_target_blcb = mean_or_nan(
                r.post_score(p.id) for p in target_new
            ) if ring is not None else float("nan")

            # per-author reach feedback (drives volume) + (1+1)-ES style adaptation (§9.2)
            for author in authors:
                npp = max(1, posts_by_author.get(author.id, 1))
                author.realized_reach = reach.get(author.id, 0) / npp
                author.adapt_style(author.realized_reach)
            reach_by_label = {
                a.label: (reach.get(a.id, 0) / max(1, posts_by_author.get(a.id, 1)))
                for a in authors
            }
            partisan = [a for a in authors if "partisan" in a.label]
            partisan_extremity = mean_or_nan(
                abs(float((a._committed if a._committed is not None else a.style)[0]))
                for a in partisan
            )

            diag = r.diagnostics()
            recovery = embedding_recovery(r.estimated_opinions() or {}, true_opinions)

            result.metrics.append(WindowMetrics(
                window=w, n_posts=len(new_posts), n_reactions=len(reactions),
                exploration_rate=(n_explore / n_impr) if n_impr else 0.0,
                gini_lambda=diag.get("gini_lambda", 0.0),
                n_eff=diag.get("n_eff", 0.0),
                mean_bridge_score=diag.get("mean_b_lcb", float("nan")),
                true_value=v_sum / n_impr if n_impr else float("nan"),
                toxicity=tox_sum / n_impr if n_impr else float("nan"),
                divisiveness=div_sum / n_impr if n_impr else float("nan"),
                satisfaction=sat_sum / n_impr if n_impr else float("nan"),
                recovery=recovery,
                feed_churn=float(np.mean(churn_samples)) if churn_samples else 0.0,
                content_divisiveness=content_div,
                content_true_value=content_val,
                partisan_extremity=partisan_extremity,
                ring_target_blcb=ring_target_blcb,
                reach_by_label=reach_by_label,
            ))
        return result

    def compare(self, rankers: List[str], n_windows: int = 8) -> Dict[str, SimulationResult]:
        """Run each ranker on the same seeded world; returns name -> result."""
        return {name: self.run(name, n_windows=n_windows) for name in rankers}

    # -------------------------------------------------------------- helpers
    def _exploration(self, live_posts, feed_set, rng):
        """Uniform-random exposures at the floor rate (the identifiability anchor, §6.2).

        Stochastic rounding keeps the *expected* rate at ``epsilon_min * n_slots``
        even below 1, so a small feed never silently rounds the anchor away.
        """
        eps = self.config.epsilon_min
        expected = eps * self.n_slots
        k = int(np.floor(expected))
        if rng.random() < (expected - k):
            k += 1
        pool = [p.id for p in live_posts if p.id not in feed_set]
        if not pool or k == 0:
            return []
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [(pool[i], ExposureSource.EXPLORATION) for i in np.atleast_1d(idx)]

    def _reaction(self, uid, pid, kind: ReactionKind, window, rng) -> Reaction:
        val = DEFAULT_REACTION_VALUES[kind]
        if kind is ReactionKind.EXPOSED_NO_REACTION:
            val = -abs(self.config.exposed_no_reaction_c)
        return Reaction(uid, pid, float(val), kind=kind, timestamp=float(window) + rng.random())
