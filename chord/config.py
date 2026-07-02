"""Configuration and the knob panel (§12).

Two kinds of parameter live here:

* **Consumption knobs** — ``M``, ``rho``, ``theta``, ``epsilon`` — freely set by
  a user; they only change the chooser's own feed and are therefore ungameable
  (§7.1). Represented by :class:`UserKnobs`.
* **System / estimator config** — MF dimension, regularization, LCB ``beta``,
  eigentrust teleport ``delta``, budget ``B0``/``eta``, recycling ``zeta``, the
  IPW clip, the exploration floor. Represented by :class:`ChordConfig`.

Earned authority (``lambda_u``, ``q_scout``, ``B(a)``) is *never* user-set — the
consumption-vs-authority wall (§12) is enforced structurally by simply not
exposing those on :class:`UserKnobs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class UserKnobs:
    """Per-user consumption knobs (the free rows of the §12 table)."""

    # M in [0,1]: bridging vs personalization master dial (§7.1).
    #   M=0 -> engagement-like (my side's content, divisiveness included)
    #   M=1 -> pure bridging (broad tested support only; divisiveness penalized)
    M: float = 0.7
    # rho in [0,1]: which divides to bridge; scales the divide-weighting A (§7.1).
    rho: float = 1.0
    # Per-factor consumption weights on the simplex (§7.3). Missing factors -> 0.
    theta: Dict[str, float] = field(default_factory=lambda: {"bridge": 1.0})
    # Exploration appetite, floored by the system (§8, §12).
    epsilon: float = 0.1

    def validate(self) -> None:
        if not 0.0 <= self.M <= 1.0:
            raise ValueError(f"M must be in [0,1], got {self.M}")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [0,1], got {self.rho}")
        if self.epsilon < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if self.theta:
            s = sum(self.theta.values())
            if s <= 0:
                raise ValueError("theta weights must sum to a positive value")
            if any(v < 0 for v in self.theta.values()):
                raise ValueError("theta weights must be non-negative")

    def normalized_theta(self) -> Dict[str, float]:
        """theta projected onto the simplex (sums to 1)."""
        if not self.theta:
            return {"bridge": 1.0}
        s = sum(self.theta.values())
        return {k: v / s for k, v in self.theta.items()}


@dataclass
class ChordConfig:
    """System-level configuration for the estimator and mechanisms."""

    # --- Signed reaction scaling (§4.1) ---
    # Magnitude c of the exposed-no-reaction weak negative, 0<c<1.
    exposed_no_reaction_c: float = 0.1

    # --- Matrix factorization (§4.1, §9.1) ---
    d: int = 8  # opinion-space dimension (bias-variance knob, §13.4)
    mf_iters: int = 20  # ALS sweeps
    reg_embedding: float = 0.05  # L2 on x_u, y_p
    reg_bias_user: float = 0.1  # ridge / partial-pool on b_u
    reg_bias_post: float = 0.1  # tau_p^-2 partial pooling on b_p
    reg_bias_author: float = 0.05  # tau_a^-2 partial pooling on b_a
    mf_tol: float = 1e-5  # early-stop tolerance on weighted RMSE

    # --- Bridging / B_LCB (§4.2) ---
    # The cross-cluster reception is aggregated after an empirical-Bayes shrinkage
    # of each cluster's reconstructed reception toward the population mean, weighted
    # by that cluster's (propensity-corrected) *exposure* n_cp. Thinly-exposed
    # clusters regress to the mean (their apparent dissent is sampling noise), and
    # only well-exposed dissent pulls the score down. Validated on Community Notes /
    # Polis to beat the old subtractive penalty and the scalar b_p (§4.2, App C.5).
    bridging_shrinkage_n0: float = 8.0  # exposure pseudo-count; n_cp/(n_cp+n0) trust
    # Cross-cluster aggregator: "nash" (geometric mean of agree-probabilities =
    # Polis's group-informed consensus — the default; a single opposed group still
    # blocks high consensus, without hard-min brittleness), "min" (Ethelo's Rawlsian
    # worst-cluster), or "ede" (Atkinson equally-distributed-equivalent, inequality
    # aversion bridging_ede_eps). Nash tracked genuine cross-group support best on
    # Polis and reached b_p parity on Community Notes (Appendix C.5).
    bridging_aggregator: str = "nash"
    bridging_ede_eps: float = 4.0      # Atkinson inequality aversion (eps→∞ ≈ min)
    # Anti-bait depth handling (§10): when a per-post depth/quality signal is
    # available (``post.features['depth']`` ∈ [0,1]), the bridge factor promotes
    # genuine depth (additive ``depth_reward``·(depth−½)) and attenuates a *shallow*
    # post's positive bridged support toward a floor (multiplicative ``depth_gate``),
    # so shallow "bridging-bait" cannot be crowned. Both 0 ⇒ off (needs a signal).
    depth_reward: float = 0.0
    depth_gate: float = 0.0
    # E9 (§4.2): shrink B_LCB toward a hierarchical author×cluster prior (author history →
    # cluster mean → global μ) instead of μ, so an untested one-sided firehose is
    # predicted-low by B_LCB itself, not left to §8's budget. Deterministic. Off preserves
    # the plain-μ prior.
    hierarchical_prior: bool = False
    hierarchical_n0_author: float = 8.0   # author-history prior strength
    hierarchical_decay: float = 0.7       # cross-window decay of author reception
    # E2 (§6/§13.2): calibrate organic reception against the ε-slice — fit a per-cluster
    # bias model r_exp≈a+b·r_org on paired (organic, exploration) reception (accumulated
    # across windows) and predict unconfounded reception everywhere. Off = no correction.
    bias_calibration: bool = False
    # E5a (§13#10): down-weight reactions NOT preceded by a ranker/ε delivery (source
    # OUT_OF_BAND) for the AUTHORITY signal B_LCB — a low-reach ring must route out-of-
    # band, so this throttles it to the commons rate ε. 1.0 = off; ~0.05 = near-zero.
    authority_out_of_band_weight: float = 1.0
    n_clusters: int = 2  # default Partition adapter cluster count
    # Retained for the legacy subtractive-LCB path and external references; the
    # default shrinkage bound (above) does not use them.
    lcb_beta: float = 1.0
    lcb_sigma: float = 1.0

    # --- Divisiveness A (§4.1) ---
    # Weight of the affective-polarization-correlated axes when building A. With
    # affective_weighting=False, A=I (the glib default of §4.1).
    affective_weighting: bool = True

    # --- Rater weighting / eigentrust (§5) ---
    eigentrust_delta: float = 0.85  # teleport (1-delta is the floor)
    eigentrust_iters: int = 50
    eigentrust_tol: float = 1e-8
    quality_track_mix: float = 0.5  # blend eigentrust with quality-agreement weight
    # Ring defense (§5, App C.5): weight each rater's *transmitted* trust by the
    # normalized entropy of its outgoing row, so a single-target rater (every
    # collusion-ring puppet, out-degree 1) forwards ≈0 of its teleport-floor mass.
    # Validated on Wikipedia RfA to flatten the ring-size→influence curve at no cost
    # to honest ranking.
    sybil_out_diversity: bool = True
    out_diversity_floor: float = 0.0   # floor on the transmit weight (0 = full defense)
    # Collusion defense (§5/§10): subtract `coordination_penalty · coordination(p)`
    # from B_LCB, where coordination(p) ∈ [0,1] is how correlated a post's approvers
    # are. A *distributed* sybil ring (puppets camouflaged across clusters, all
    # boosting one target) fakes cross-cluster support that min-over-clusters misses,
    # but cannot hide that its boosters co-approve in lockstep. 0 ⇒ off.
    coordination_penalty: float = 0.0
    # Stronger, camouflage-resistant collusion defense (§13.10): subtract
    # `collusion_loyalty_penalty · manufactured_fraction(author)` from B_LCB, where
    # manufactured_fraction is the rolling fraction of an author's positive support
    # from raters who approve nearly *all* its posts window-after-window — the one
    # signature a camouflaged distributed ring cannot erase. 0 ⇒ off.
    collusion_loyalty_penalty: float = 0.0
    # Exploration-anchor cap (§6.2/§13.10): cap each cluster's reconstructed reception
    # at the upper confidence bound of the author's reception among *exploration*
    # (uniform-random, unconfounded) exposures. Discards a ring's common-mode organic
    # lift, K-independently. Needs enough exploration traffic (a higher epsilon floor)
    # to bind; misses safely (never false-positives) when evidence is thin. False ⇒ off.
    exploration_anchor_cap: bool = False

    # --- Scout precision (§5) ---
    scout_alpha: float = 0.5  # rank-decay in q_scout

    # --- Influence recycling (§8) ---
    recycling_zeta: float = 0.3  # boost the under-served, damp the satisfied

    # --- Propensity / IPW (§6) ---
    W_max: float = 20.0  # inverse-propensity clip; tie to 1/epsilon (§6.2)
    self_normalized: bool = True  # use SNIPW (§6.2)

    # --- Author visibility budget (§8) ---
    budget_B0: float = 10.0  # base per-window budget
    budget_eta: float = 1.0  # strength-replenishment rate
    budget_max: float = 100.0  # ceiling to keep budgets bounded
    # Budget-recursion refinements (Fable review, §8).
    # #1 memory γ: carry (B_t − B_0) across windows so a quiet cadence isn't reset to the
    #    floor ("posted nothing" ≠ "earned nothing"); the author-side anti-ossification
    #    half-life mirroring rater recycling. 0 = memoryless (original behavior).
    budget_memory: float = 0.0
    # #4 share-based issuance: a fixed aggregate pool per window (budget_aggregate_factor ·
    #    n · B_0) distributed by *relative* realized strength, so total issuance is not
    #    procyclical with aggregate engagement — genuine system-wide conservation. Off =
    #    per-author η·ΣΦE (procyclical).
    budget_share_based: bool = False
    budget_aggregate_factor: float = 2.0

    # --- Exploration pool (§8) ---
    epsilon_min: float = 0.05  # floored system invariant (§9.3)
    epsilon_max: float = 0.5
    exploration_saturation_var: float = 0.05  # close audition below this variance
    newcomer_base_rate: float = 0.1  # empirical newcomer "winner" rate (§8 prior)

    # --- Concentration controller (§9.3) ---
    gini_ceiling: float = 0.6  # if Gini(lambda) exceeds -> tighten
    controller_delta_step: float = 0.02  # raise teleport floor
    controller_epsilon_step: float = 0.01  # raise epsilon_min
    # E12 (§9.3): a CUSUM change-point alarm on Gini(λ) drift vs a slow rolling baseline,
    # in addition to the fixed level ceiling. The ceiling is data-derived (h·σ of the
    # baseline), so the guard is *active* against a concentration attack instead of dormant
    # far below 0.6. Off = level ceiling only.
    controller_cusum: bool = False
    controller_cusum_k: float = 0.5      # slack (in baseline σ) before drift accumulates
    controller_cusum_h: float = 5.0      # alarm threshold (in baseline σ)
    controller_cusum_warmup: int = 5     # windows to establish the baseline
    # E4 (§4/§13#4): gate crowning on a residual-whiteness test — a post whose rank-d
    # residuals are spatially autocorrelated with the co-reaction graph (Moran's I,
    # permutation p<α) is dividing along an *unmodeled* axis, so demote it. Off = no gate.
    whiteness_gate: bool = False
    whiteness_alpha: float = 0.05
    whiteness_penalty: float = 0.5
    # E3 (§13#3): amplification collar — throttle a post's realized strength (hence its
    # budget/reach) when its reach outruns its *tested* audience, E(p) > κ·n_tested, so
    # amplification proceeds in rungs that re-certify B_LCB before each expansion. Off = no cap.
    amplification_collar: bool = False
    collar_kappa: float = 4.0
    # E6 (§8): only boost recycling λ for apparent under-service that is *corroborated
    # off-policy* — the user realizes value on ε-slice items the ranker wouldn't have shown.
    # A farmer acting dissatisfied but not preferring ε content gets no boost. Off = model
    # dissatisfaction only.
    recycling_offpolicy_verify: bool = False

    def validate(self) -> None:
        if self.d < 1:
            raise ValueError("d must be >= 1")
        if not 0.0 < self.eigentrust_delta < 1.0:
            raise ValueError("eigentrust_delta must be in (0,1) for a contraction")
        if not 0.0 < self.exposed_no_reaction_c < 1.0:
            raise ValueError("exposed_no_reaction_c must be in (0,1)")
        if self.epsilon_min <= 0.0:
            raise ValueError("epsilon_min must be > 0 (identifiability anchor, §6.2)")
        if self.epsilon_min > self.epsilon_max:
            raise ValueError("epsilon_min must be <= epsilon_max")
        if self.W_max <= 1.0:
            raise ValueError("W_max must be > 1")

    def clamp_epsilon(self, eps: float) -> float:
        """Clamp a user's exploration appetite to [epsilon_min, epsilon_max]."""
        return float(min(self.epsilon_max, max(self.epsilon_min, eps)))
