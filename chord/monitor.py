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
    # E12 CUSUM state: slow baseline (mean/var, frozen during an alarm) + drift accumulator
    cusum: float = 0.0
    base_mean: float = 0.0
    base_var: float = 0.0
    n_obs: int = 0
    cusum_alarm: bool = False


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

        breach = g > cfg.gini_ceiling
        if cfg.controller_cusum:
            breach = self._cusum_breach(g) or breach

        if breach:
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
                                    "epsilon_min": self.state.epsilon_min,
                                    "cusum": self.state.cusum,
                                    "cusum_alarm": self.state.cusum_alarm})
        return self.state

    def _cusum_breach(self, g: float) -> bool:
        """CUSUM change-point alarm on Gini drift vs a slow baseline (E12, §9.3).

        Accumulates upward deviations beyond ``k`` baseline-σ and alarms past ``h`` σ —
        a *data-derived* threshold, so it fires on a concentration attack that never
        approaches the fixed 0.6 ceiling. The baseline is frozen once alarmed (so it does
        not adapt to the attack it just caught)."""
        s, cfg = self.state, self.config
        s.n_obs += 1
        a = 1.0 / min(s.n_obs, 20)                 # exact running mean early → ~EWMA later
        if s.n_obs <= cfg.controller_cusum_warmup:
            d = g - s.base_mean
            s.base_mean += a * d
            s.base_var = (1 - a) * s.base_var + a * d * d
            return False
        sd = max(s.base_var ** 0.5, 1e-3)
        s.cusum = max(0.0, s.cusum + (g - s.base_mean - cfg.controller_cusum_k * sd))
        if s.cusum > cfg.controller_cusum_h * sd:
            s.cusum = 0.0
            s.cusum_alarm = True
            return True
        if not s.cusum_alarm:                      # adapt the baseline only while healthy
            d = g - s.base_mean
            s.base_mean += a * d
            s.base_var = (1 - a) * s.base_var + a * d * d
        return False


def coreaction_adjacency(reactions, users: Sequence[Id]) -> np.ndarray:
    """User×user co-reaction similarity (E4): positive cosine of mean-centred reaction
    vectors. Captures the *full* opinion structure — including axes beyond the fitted
    rank — which is what lets a hidden divide show up as spatial autocorrelation. Dense;
    intended for the crowning-candidate gate at modest scale."""
    users = list(users)
    ui = {u: i for i, u in enumerate(users)}
    posts = sorted({r.post_id for r in reactions}, key=lambda x: str(x))
    pj = {p: j for j, p in enumerate(posts)}
    R = np.zeros((len(users), len(posts)))
    C = np.zeros_like(R)
    for r in reactions:
        i, j = ui.get(r.user_id), pj.get(r.post_id)
        if i is not None and j is not None:
            R[i, j] += r.value
            C[i, j] += 1.0
    R = np.where(C > 0, R / np.maximum(C, 1.0), 0.0)
    R = R - R.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / np.maximum(norm, 1e-9)
    W = np.clip(Rn @ Rn.T, 0.0, None)          # positive cosine similarity
    np.fill_diagonal(W, 0.0)
    return W


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    """Moran's I spatial autocorrelation of ``values`` under adjacency ``W``."""
    r = values - values.mean()
    S0 = W.sum()
    denom = float(r @ r)
    if S0 <= 0 or denom <= 0:
        return 0.0
    return float((len(r) / S0) * (r @ W @ r) / denom)


def residual_whiteness(result, reactions, post_authors, users, W,
                       candidates, min_reactors: int = 6, n_perm: int = 99, seed: int = 0):
    """Per-post (Moran's I, permutation p-value) of the model residuals (E4, §4/§13#4).

    A post genuinely bridging at the fitted rank has *exchangeable* residuals (white);
    divisiveness along an unmodeled axis shows up as residuals autocorrelated with the
    co-reaction graph. Computed only for ``candidates`` (crowning gate). ``users`` indexes
    the rows/cols of ``W``.
    """
    ui = {u: i for i, u in enumerate(users)}
    by_post: Dict[Id, list] = {}
    for r in reactions:
        by_post.setdefault(r.post_id, []).append(r)
    rng = np.random.default_rng(seed)
    out: Dict[Id, tuple] = {}
    for pid in candidates:
        rs = by_post.get(pid, [])
        idx, resid = [], []
        y = result.y_post.get(pid)
        b_p = result.b_post.get(pid, 0.0)
        b_a = result.b_author.get(post_authors.get(pid), 0.0)
        for r in rs:
            i = ui.get(r.user_id)
            x = result.x_user.get(r.user_id)
            if i is None:
                continue
            pred = (result.mu + result.b_user.get(r.user_id, 0.0) + b_a + b_p
                    + (float(x @ y) if (x is not None and y is not None) else 0.0))
            idx.append(i)
            resid.append(r.value - pred)
        if len(idx) < min_reactors:
            out[pid] = (0.0, 1.0)
            continue
        sub = W[np.ix_(idx, idx)]
        vals = np.asarray(resid)
        obs = morans_i(vals, sub)
        null = [morans_i(vals[rng.permutation(len(vals))], sub) for _ in range(n_perm)]
        p = (1.0 + sum(1 for x in null if x >= obs)) / (n_perm + 1.0)
        out[pid] = (obs, p)
    return out


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
