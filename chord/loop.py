"""The CHORD estimation loop and serving path (§9.1).

Per window (learning plane):
  (1) fit the doubly-robust, IPW-corrected, lambda-weighted MF (block-convex ALS)
      -> embeddings, biases;
  (2) whiten, recompute A and D;
  (3) update quality-tracking lambda on the new geometry; iterate 1-3;
  (4) update q_scout;
  (5) update author budgets;
  (6) update Thompson posteriors;
  (7) apply recycling -> lambda_eff.

Per request (serving plane):
  retrieve candidates -> score V(u,p) with the user's knobs -> greedy constrained
  select -> serve -> log.

The two planes run at different cadences — this *is* the two-timescale separation
of §9: the serving plane and the fast MF refit are the fast timescale; the
lambda/credibility update is the slow timescale.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import ChordConfig, UserKnobs
from .types import Exposure, ExposureSource, Id, Post, Reaction
from .model import (
    BridgingScorer,
    BridgingScores,
    ClusterModel,
    DivisivenessModel,
    FactorizationResult,
    MatrixFactorization,
    CollusionTracker,
    ExplorationAnchor,
    coordination_scores,
    fit_divisiveness,
)
from .rater import (
    apply_recycling,
    blend_lambda,
    compute_lambda,
    compute_scout_precision,
    quality_tracking_weight,
)
from .propensity import (
    LoggedPropensityModel,
    PropensityModel,
    UniformExplorationModel,
    compute_ipw_weights,
)
from .economy import AuthorBudgetLedger, ExplorationPool
from .feed import Candidate, FactorContext, blended_value, greedy_assemble
from .monitor import ConcentrationController
from .ports.adapters import KMeansPartitionAdapter


@dataclass
class WindowState:
    """The fitted model state after one learning window (§9.1)."""

    result: FactorizationResult
    divisiveness: DivisivenessModel
    clusters: ClusterModel
    bridging: BridgingScores
    rater_lambda: Dict[Id, float]
    rater_lambda_eff: Dict[Id, float]
    scout: Dict[Id, float]
    post_authors: Dict[Id, Id]
    realized_strength: Dict[Id, float]
    n_iter_inner: int = 0


class Chord:
    """End-to-end CHORD estimator + ranker.

    Holds the pluggable propensity model (default: logged/uniform-exploration),
    the author-budget ledger, the exploration pool, and the stability controller.
    Call :meth:`fit_window` per learning window and :meth:`rank` per request.
    """

    def __init__(
        self,
        config: Optional[ChordConfig] = None,
        propensity_model: Optional[PropensityModel] = None,
        seed: int = 0,
        inner_iters: int = 2,
    ):
        self.config = config or ChordConfig()
        self.config.validate()
        self.seed = seed
        self.inner_iters = max(1, inner_iters)  # steps 1-3 iteration count
        self.propensity_model = propensity_model or LoggedPropensityModel(
            self.config.epsilon_min
        )
        self.budget = AuthorBudgetLedger(self.config)
        self.exploration = ExplorationPool(self.config, seed=seed)
        self.controller = ConcentrationController(self.config)
        self.collusion = CollusionTracker()   # rolling loyal-bloc detector (§13.10)
        self.reception_anchor = ExplorationAnchor()   # exploration-anchored cap (§6.2/§13.10)
        self.state: Optional[WindowState] = None

    # ------------------------------------------------------------- learning
    def fit_window(
        self,
        reactions: Sequence[Reaction],
        posts: Mapping[Id, Post],
        exposures: Optional[Sequence[Exposure]] = None,
        identity_of: Optional[Mapping[Id, Id]] = None,
        affective_signal: Optional[Mapping[Id, float]] = None,
    ) -> WindowState:
        """Run one learning window (steps 1-7 of §9.1)."""
        # Apply the §9.3 concentration controller's current response: if a prior window
        # saw Gini(λ) breach the ceiling, the controller lowered δ (more teleport) and
        # raised ε_min; use those here instead of the static config. In healthy
        # operation the controller relaxes back to the configured δ/ε_min, so this is a
        # no-op until concentration actually breaches — then it acts (§9.3).
        cs = self.controller.state
        cfg = replace(self.config, eigentrust_delta=cs.eigentrust_delta,
                      epsilon_min=cs.epsilon_min)
        posts = dict(posts)
        post_authors = {pid: p.author_id for pid, p in posts.items()}
        exposures = list(exposures or [])
        exposure_index: Dict[Tuple[Id, Id], Exposure] = {
            (e.user_id, e.post_id): e for e in exposures
        }
        users = list({rx.user_id for rx in reactions})

        # --- steps 1-3: inner actor-critic iteration (fast critic = MF; slow
        #     actor = lambda). Start lambda uniform, refine on the new geometry.
        rater_lambda: Dict[Id, float] = {u: 1.0 / max(1, len(users)) for u in users}
        result = None
        clusters = None
        divis = None
        n_inner = 0
        for it in range(self.inner_iters):
            n_inner = it + 1
            weights = compute_ipw_weights(
                reactions,
                self.propensity_model,
                cfg,
                rater_lambda=rater_lambda,
                exposures=exposure_index,
            )
            mf = MatrixFactorization(cfg, seed=self.seed)
            result = mf.fit(reactions, posts, weights)

            # step 2: whiten, recompute A and D
            divis = fit_divisiveness(result, cfg, dict(affective_signal) if affective_signal else None)

            # partition (default adapter) on the new geometry
            partitioner = KMeansPartitionAdapter(
                n_clusters=cfg.n_clusters, seed=self.seed
            )
            assignments = partitioner.assign(users, result.x_user)
            clusters = ClusterModel.from_factorization(result, assignments)

            # step 3: update quality-tracking lambda on the new geometry
            eig = compute_lambda(reactions, posts, result, users, cfg)
            qual = quality_tracking_weight(reactions, posts, result, cfg)
            rater_lambda = blend_lambda(eig, qual, cfg)

        assert result is not None and clusters is not None and divis is not None

        # --- bridging scores with propensity-corrected per-cluster exposure counts
        exposure_counts = _cluster_exposure_counts(exposures, clusters)
        reception_caps = None
        if cfg.exploration_anchor_cap:
            self.reception_anchor.update(reactions, exposures, post_authors)
            reception_caps = self.reception_anchor.caps(post_authors)
        scorer = BridgingScorer(cfg)
        bridging = scorer.score(result, clusters, post_authors, exposure_counts, reception_caps)
        # collusion defense (§5/§10): discount posts whose approvers are coordinated,
        # which a distributed cross-cluster ring cannot avoid (min-over-clusters can).
        if cfg.coordination_penalty > 0.0:
            coord = coordination_scores(reactions)
            for pid, score in coord.items():
                v = bridging.b_lcb.get(pid)
                if v is not None and np.isfinite(v):
                    bridging.b_lcb[pid] = v - cfg.coordination_penalty * score
        # stronger, camouflage-resistant defense: discount authors whose support is
        # manufactured by a loyal bloc across windows (§13.10).
        if cfg.collusion_loyalty_penalty > 0.0:
            self.collusion.update(reactions, post_authors)
            for pid in list(bridging.b_lcb.keys()):
                a = post_authors.get(pid)
                v = bridging.b_lcb.get(pid)
                if a is not None and v is not None and np.isfinite(v):
                    frac = self.collusion.manufactured_fraction(
                        a, clusters.assignments, clusters.n_clusters)
                    bridging.b_lcb[pid] = v - cfg.collusion_loyalty_penalty * frac
        realized_strength = dict(bridging.b_lcb)
        # replace -inf (unseen) with a low finite floor for downstream arithmetic
        for pid, v in realized_strength.items():
            if not np.isfinite(v):
                realized_strength[pid] = 0.0

        # --- step 4: scout precision
        scout = compute_scout_precision(reactions, posts, realized_strength, cfg)

        # --- step 5: author budgets (bind to identity, §8/§10)
        exposure_per_post: Dict[Id, float] = defaultdict(float)
        for e in exposures:
            exposure_per_post[e.post_id] += 1.0
        post_identity = {
            pid: (identity_of.get(a, a) if identity_of else a)
            for pid, a in post_authors.items()
        }
        self.budget.replenish(realized_strength, exposure_per_post, post_identity)

        # --- step 6: Thompson posteriors for auditioned posts
        self._update_exploration(bridging, exposures)

        # --- step 7: influence recycling -> lambda_eff
        satisfaction = self._realized_satisfaction(reactions, result, divis)
        rater_lambda_eff = apply_recycling(rater_lambda, satisfaction, cfg)

        # --- stability controller (§9.3): tighten if concentration climbs
        self.controller.step(rater_lambda_eff)

        self.state = WindowState(
            result=result,
            divisiveness=divis,
            clusters=clusters,
            bridging=bridging,
            rater_lambda=rater_lambda,
            rater_lambda_eff=rater_lambda_eff,
            scout=scout,
            post_authors=post_authors,
            realized_strength=realized_strength,
            n_iter_inner=n_inner,
        )
        return self.state

    # -------------------------------------------------------------- serving
    def rank(
        self,
        user_id: Id,
        candidates: Sequence[Post],
        knobs: Optional[UserKnobs] = None,
        n_slots: int = 10,
        extras: Optional[Mapping[Id, Dict[str, float]]] = None,
    ) -> List[Id]:
        """Score and select a feed for one request (§9.1 serving path).

        Unseen candidates (no fitted loading) are routed through the exploration
        pool with an optimistic Thompson score and flagged to satisfy the
        exploration floor.
        """
        if self.state is None:
            raise RuntimeError("call fit_window before rank")
        knobs = knobs or UserKnobs()
        knobs.validate()
        # honor the controller's raised ε_min (§9.3 persistent excitation) when it has
        # tightened in response to concentration; otherwise this equals the config floor.
        eps = float(min(self.config.epsilon_max,
                        max(self.controller.state.epsilon_min, knobs.epsilon)))
        st = self.state
        extras = extras or {}

        # Anti-bait depth handling (§10): fold the per-post depth signal and the
        # system depth weights into each post's extras when configured.
        cfg = self.config
        depth_on = cfg.depth_reward > 0.0 or cfg.depth_gate > 0.0

        cand_objs: List[Candidate] = []
        for post in candidates:
            b_lcb = st.bridging.b_lcb.get(post.id, float("-inf"))
            seen = post.id in st.result.y_post and np.isfinite(b_lcb)
            post_extras = dict(extras.get(post.id, {}))
            if depth_on:
                post_extras.setdefault("depth", float(post.features.get("depth", 1.0)))
                post_extras.setdefault("depth_reward", cfg.depth_reward)
                post_extras.setdefault("depth_gate", cfg.depth_gate)
            ctx = FactorContext(
                user_id=user_id,
                post=post,
                b_lcb=b_lcb if np.isfinite(b_lcb) else 0.0,
                result=st.result,
                divisiveness=st.divisiveness,
                knobs=knobs,
                extras=post_extras,
            )
            base_value = blended_value(ctx)
            if not seen:
                self.exploration.register(post.id)
            expl_score = self.exploration.sample_score(post.id) if not seen else 0.0
            coverage = self._approval_coverage(post.id) if seen else None
            budget_identity = post.author_id
            cand_objs.append(
                Candidate(
                    post_id=post.id,
                    author_id=budget_identity,
                    base_value=base_value,
                    exploration_value=expl_score,
                    approval_coverage=coverage,
                    exposure_cost=1.0,
                    posdisc=1.0,
                    is_exploration=not seen,
                )
            )

        author_budgets = {
            c.author_id: self.budget.budget(c.author_id) for c in cand_objs
        }
        res = greedy_assemble(
            cand_objs,
            n_slots=n_slots,
            epsilon=eps,
            author_budgets=author_budgets,
            n_clusters=st.clusters.n_clusters,
        )
        return res.selected

    # ------------------------------------------------------------- helpers
    def _approval_coverage(self, post_id: Id) -> Optional[np.ndarray]:
        """Per-cluster earned approval used by the diverse-approval submodular term."""
        st = self.state
        rc = st.bridging.per_cluster.get(post_id)
        if rc is None:
            return None
        return np.maximum(rc, 0.0)

    def _update_exploration(
        self, bridging: BridgingScores, exposures: Sequence[Exposure]
    ) -> None:
        """Fold audition outcomes (normalized B_LCB) into Thompson posteriors."""
        exposed_posts = {e.post_id for e in exposures}
        vals = [v for v in bridging.b_lcb.values() if np.isfinite(v)]
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        for pid, b in bridging.b_lcb.items():
            if pid not in exposed_posts or not np.isfinite(b):
                continue
            reward = (b - lo) / span  # normalize B_LCB into [0,1]
            self.exploration.observe(pid, reward)

    def _realized_satisfaction(
        self,
        reactions: Sequence[Reaction],
        result: FactorizationResult,
        divis: DivisivenessModel,
    ) -> Dict[Id, float]:
        """S_bar(u): model-estimated realized value over what a user reacted to (§8)."""
        per_user_sum: Dict[Id, float] = defaultdict(float)
        per_user_cnt: Dict[Id, int] = defaultdict(int)
        for rx in reactions:
            x = result.x_user.get(rx.user_id)
            y = result.y_post.get(rx.post_id)
            if x is None or y is None:
                continue
            per_user_sum[rx.user_id] += float(np.dot(x, y))
            per_user_cnt[rx.user_id] += 1
        return {
            u: per_user_sum[u] / per_user_cnt[u]
            for u in per_user_sum
            if per_user_cnt[u] > 0
        }


def _cluster_exposure_counts(
    exposures: Sequence[Exposure], clusters: ClusterModel
) -> Dict[Id, Dict[int, float]]:
    """Propensity-corrected count n_cp of cluster-c users exposed to post p (§4.2).

    Exploration exposures (known epsilon) and organic exposures both count toward
    tested breadth; a cluster with no exposures to a post gets n_cp = 0, which
    maximizes its B_LCB pessimism penalty.
    """
    counts: Dict[Id, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for e in exposures:
        c = clusters.assignments.get(e.user_id)
        if c is None:
            continue
        counts[e.post_id][c] += 1.0
    return {pid: dict(cmap) for pid, cmap in counts.items()}
