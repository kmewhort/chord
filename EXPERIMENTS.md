# Experiment plan — Fable's whitepaper feedback

One concrete, runnable experiment per open-problem proposal, mapped to the harness that
can actually test it (the committed `validate/` datasets — Coat/Polis/CN/RfA — or the
closed-loop simulator). Ordered by leverage × feasibility. Ethos is the same as Appendix
C: an experiment that *fails* to help is a finding, not a defeat.

Legend — **harness**: `CN`=Community Notes, `RfA`=Wikipedia RfA, `Coat`=MAR block,
`Polis`, `sim`=simulator. **det**=deterministic (reproducibility-safe by construction).

## Tier 1 — deterministic, prototypable now on committed real data

### E9. Hierarchical author×cluster prior (fixes §9 leniency) — *det, CN + sim*
Replace the global-mean prior μ in `B_LCB` shrinkage with a hierarchical prior: shrink
`r_cp` toward the author's own historical cluster-`c` reception → the cluster mean → the
global mean.
- **CN experiment** (single-window, leave-one-out for "history"): for each note, form the
  author prior from the author's *other* notes' per-cluster reception. Compare B_LCB(μ)
  vs B_LCB(hierarchical) AUC against helpful/not, *restricted to thin notes* (few
  ratings) where μ is lenient. Claim: hierarchical predicts-low on a consistently-bad
  author's thin notes without hurting overall AUC, and stays order-reproducible.
- **sim experiment**: firehose reach with grand=μ (budget carries it, ~0.75×) vs
  hierarchical (B_LCB itself suppresses before the budget bites). Claim: `firehose_reach`
  drops and true_value holds. Reproducibility check: property test still ≥0.95.

### E5a. Exposure-attributed rating weight (collusion) — *det, sim (+ CN caveat)*
Every reaction is exposure-attributed (delivered by the ranker/ε-pool) or out-of-band. A
low-reach ring must route out-of-band. Weight non-exposure-attributed reactions ≈0 for
*authority* (keep them for personalization).
- **sim experiment**: tag ring reactions as out-of-band (they are — puppets boost a
  low-reach target directly); weight them ~0 in `cluster_reception`/λ. Claim: ring
  inflation collapses without the loyalty penalty even engaged, and the throttle is the
  commons rate ε. Compare to the loyalty defense (complementary).
- CN caveat: the slice has no exposure log, so CN only checks the weight doesn't hurt the
  honest keystone AUC; the attack test is sim-only.

### E2. Bias model + E-values on the ε-slice (confounding) — *det, Coat*
Coat's MAR block *is* the ε-slice. Fit a bias model `g` on paired (organic MNAR, random
MAR) item observations; subtract `g` from organic reception everywhere.
- **Coat experiment**: split MAR into fit/eval. Fit `g(item features)= MNAR_mean −
  MAR_mean` on fit-half; de-bias MNAR with `g`; measure |estimate − true| vs (a) raw
  MNAR, (b) IPW, (c) the anchor-cap (E-anchor). Claim: bias-model subtraction ≥ IPW at
  removing the level bias. Plus: compute a per-item **E-value** (min confounding to flip a
  crowning verdict) and show it's one line on existing numbers.

## Tier 2 — simulator experiments (need a DGP knob)

### E3. Amplification collar (audience non-monotonicity) — *sim*
Cap exposure at `E(p) ≤ κ·n_tested` (staged rungs). Add a DGP where a post bridges at
small scale but divides at large scale (a hidden-axis post whose opposed cluster is rare).
Claim: the collar re-certifies each rung and halts amplification when the large-audience
divide appears, vs uncapped over-amplification. Reuses the budget ledger.

### E1. Measured performativity / empirical Lipschitz (convergence) — *sim*
Inject small randomized ranking perturbations (micro-A/B on the ε machinery); measure the
induced reaction-distribution shift = empirical Lipschitz of the distribution map; read
loss curvature off the ALS block. Claim: the ratio predicts the stable→oscillate boundary
better than the performativity knob alone (sharpen §13#F5), and feeding it to the §9.3
controller holds the ratio below threshold.

### E4. Residual whiteness test for hidden axes (§4/#4) — *sim + Polis*
Moran's I (permutation test) of a post's residual vector against the user co-reaction
graph. DGP: set the true opinion rank > fitted `d`. Claim: crowned posts with a hidden
divide have significantly non-white residuals (Moran's I rejects), giving a per-post gate
and a `d`-too-low diagnostic (rejection rate rises as d falls). Cross-check on Polis.

### E6. Off-policy recycling verification (§6) — *sim*
Boost λ only when model-estimated dissatisfaction is corroborated on the ε-slice (the user
realizes high value on exploration items the ranker wouldn't have shown). DGP: a recycling
farmer who *acts* under-served but doesn't prefer ε content. Claim: farmer gets no λ boost;
a genuinely under-served user still does.

### E11. Behavioral / saturation-trajectory depth (complements F4) — *sim*
Add a third depth channel to the vouch channel: the audition saturation *trajectory*
(bait saturates fast; slow-burn depth has a distinctive σ²_p decay). Claim: trajectory
shape separates bait from depth, and 3 channels (vouch + corrected-dwell + trajectory)
make the forge require compromising independent modalities.

### E12. CUSUM change-point controller (§9.3) — *sim*
Reframe the concentration controller as a CUSUM drift alarm on Gini(λ) vs a rolling
baseline, ceiling = rolling extreme-quantile × safety factor. Claim: it fires on a genuine
concentration attack (active) but not on the healthy ~0.06 baseline (resolves §13#12's
dormancy without hand-tuning a level).

## Tier 3 — design + meta

### E8. Endogenous seed criterion (§5/§8) — *RfA + sim*
Derive the eigentrust seed set from a published *criterion* (long-horizon exposure-
attributed, cross-cluster, high-scout-precision, forge-cost-paid), not a hand-list. On RfA:
show a criterion-seeded asymmetric teleport preserves the ring-collapse while surviving a
seed the attacker can't cheaply join.

### E-meta. Randomization portfolio (the ε-pool as a bandit) — ✅ **LANDED** (§13#13)
Make ε allocation explicit: a floored bandit over information value across {audition,
calibration (E2), audit (E5/E12), probe (E1)}. Landed as `chord.economy.RandomizationPortfolio`
(`test_randomization_portfolio.py`): keeps a floor on every arm (so ε stays a floored invariant
per demand), then water-fills the remainder by learned value. Under a fixed budget with shifting
needs (cold-start early, an audit spike during an attack) it captures **+9%** total information
value vs a uniform split (approaching the oracle water-fill) and **+29%** during the attack
window. Sharpens the paper's core argument (added as §13#13): ε is a floored invariant not for
identifiability alone but because *every* open-problem fix spends it, so the commons must be
budgeted, not merely guaranteed.

## Results — Tier 1 (run)

### E9 — hierarchical author×cluster prior: ✅ works on the intended case → **LANDED** (gated)
*Landed as `config.hierarchical_prior` (`chord/model/priors.py`); end-to-end in the sim it
raises delivered true value +28% and suppresses firehose reach. Default off (flipping it
would re-tune the μ-calibrated sim suite). Test: `test_hierarchical_prior.py`.*
- **CN (finding, not the court):** hierarchical prior *hurts* AUC (0.998→0.95) — CN's dense
  k-core notes are all well-tested, so a note's own ratings predict the label near-perfectly
  and the author prior only adds noise. CN has no *untested* content, so it can't show the
  benefit.
- **Controlled firehose-vs-quality world (the intended case):** with a global-μ prior a
  firehose author's thin one-sided posts sit near neutral (B_LCB −0.11); the hierarchical
  (author-history) prior suppresses them to **−0.36** — B_LCB itself now predicts-low, before
  the budget bites. Well-tested quality is unchanged (+0.55), and a *reformed* author's
  genuinely-good post is barely dragged (+0.55→+0.48) — evidence overwhelms the prior at the
  n_cp/(n_cp+n0) rate, i.e. self-correcting. The firehose-vs-quality gap widens −0.66→−0.92.
  **Verdict:** the right fix for the §9 leniency gap; deterministic (reproducibility-safe).
  Next: wire author×cluster history into the loop and re-run the sim firehose end-to-end.

### E2 — bias model + E-values on Coat's ε-slice: ✅ strong positive → **LANDED** (gated)
*Landed as `config.bias_calibration` (`chord/model/calibration.py`, cross-window per-cluster
bias model). Clear win on Coat's dense MAR block; in the closed loop the effect is within
seed noise at the ε floor — it *spends* ε and needs allocating (the meta-point). Deterministic
mechanism unit-tested (`test_bias_calibration.py`); closed-loop benefit not asserted.*
The MNAR→MAR bias is *structured* (corr with MNAR level +0.50), hence modelable. Fitting
`g` on ε-covered items and applying it to **held-out** items (transportability) de-biases
better than IPW: mean |error| **0.189** (bias-model) vs 0.236 (IPW) vs 0.254 (raw MNAR);
residual level bias +0.016 vs +0.080 (IPW). An E-value crowning gate flagged the truly-
negative items. **Verdict:** converts §13.2 from "bounded" to "calibrated"; deterministic.

### E5a — exposure-attributed rating weight: ✅ at the authority level (nuanced)
Implemented as a gated feature (`config.authority_out_of_band_weight`, default 1.0=off;
`ExposureSource.OUT_OF_BAND`; the sim marks a low-reach ring's self-asserted boosts
out-of-band). In isolation it does exactly its job: down-weighting out-of-band boosts drops
the ring target's B_LCB **+0.654 → +0.117** (reverts to genuine low reception). In the
closed-loop *reach* metric it looks flat (1.9×→2.2×) because a de-authoritied target still
draws exploration/audition reach — i.e. it is no longer *crowned* as bridging, but the
exploration pool still auditions it, which is correct. **Verdict:** neutralizes the ring's
manufactured *authority* (its stated purpose); complementary to the loyalty penalty, not a
standalone reach-killer. On CN there is no exposure log, so the throttle-to-ε claim is
sim-only. (Sharp caveat, per the meta-point: the throttle *spends* ε — the puppets must wait
for the commons pool to serve them the target.)

## Results — Tier 2 (run → **all six LANDED**, gated, `tests/test_tier2.py`)

All six ran positive and are now landed as gated core features (default off, suite green):
E4 `whiteness_gate`, E12 `controller_cusum`, E3 `amplification_collar`, E6
`recycling_offpolicy_verify` (loop config flags); E1 `monitor.empirical_lipschitz` and E11
`monitor.saturation_depth_prior` (measurement utilities).

- **E12 — CUSUM controller: ✅.** A concentration attack lifts Gini(λ) to a sustained ~0.24
  (healthy ~0.08); *neither* trips the level ceiling (0.6) — the §13#12 dormancy, confirmed.
  A CUSUM on a rolling baseline (ceiling = h·σ, data-derived) fires one window after onset,
  and stays silent on healthy noise. (Nuance: a system that *starts* concentrated shows no
  drift — change-point detection working as designed.)
- **E4 — residual whiteness (Moran's I): ✅ strong.** With the true opinion rank > fitted d,
  a post that divides along the *hidden* axis has Moran's I +0.63 on its rank-1 residuals
  (permutation p<0.001 → flagged); a genuinely bridging post is +0.00 (p=0.22 → white). A
  clean per-post crowning gate and d-too-low diagnostic.
- **E1 — measured performativity: ✅.** An empirical Lipschitz of the performative map
  (reaction-shift per feed-change) is measurable from existing metrics and scales with the
  performativity rate (0.067→0.093), tracking steady-state instability — so performativity
  can be *measured*, not assumed, and fed to the controller.
- **E3 — amplification collar: ✅.** For a post that bridges small but divides large (rare
  10% opposition), B_LCB is monotone-decreasing in tested audience (−0.08→−0.64), so capping
  reach at κ·n_tested forces re-certification per rung and stops a lucky small-sample post
  from over-amplifying before the divide surfaces.
- **E6 — off-policy recycling verification: ✅.** Gating the λ-boost on off-policy
  corroboration (does the user realize value on ε items?) zeros a recycling farmer's boost
  (0.70→0.04) while keeping a genuine under-served user's (0.48).
- **E11 — saturation-trajectory depth: ✅.** Bait's reception-variance decays ~5× faster
  than slow-burn depth (rate 0.68 vs 0.14), so trajectory shape is a usable depth prior
  orthogonal to the vouch channel — a third channel toward "forge requires compromising
  independent modalities."

## Re-validation finding — the refinements do NOT compose cleanly as defaults

Attempted to graduate the whole gated set (E9, E2, E4, E12, E3, E6 + budget memory/share/
streaming) to defaults and re-validate. The **headline claims survive** (welfare 8/8,
robustness, validate/ all green) — but turning them all on at once undermines several
validated *sub*-claims through mechanism interactions. Each was isolated:

- **E9 (hierarchical prior) is double-edged.** Its prior is on *approval* history, so while
  it suppresses a low-approval firehose (its intended win), it **props up high-approval bad
  actors** — a broadly-approved shallow bait and a distributed-ring target — blunting the
  depth defense (bait reach ×0.61 → ×1.12) and weakening ring containment. Worse, it rewards
  partisan consistency (in-cluster approval), *increasing* author extremity above engagement
  (Δ+0.156 vs +0.041) — hitting the core "CHORD doesn't breed extremists" claim.
- **E4 (residual whiteness) false-positives as a default gate.** Residuals correlate with the
  cluster structure, so it flags ~all crowned posts (p<0.01) in any clustered population — a
  fine *diagnostic*, an unsafe default *gate*.
- **E2 (bias calibration) subsumes the exploration-anchor cap** — both de-confound the ε-slice,
  and with both on they double-correct and *reduce* delivered value (E2 is the better one).
- **budget_memory (cadence carry) lets a firehose accumulate budget**, reversing the budget's
  firehose dilution (a persistent firehose earns a little each window and carries it).
- **budget_share_based dilutes** the depth/welfare gains and risks the ηΦ̄ bifurcation with memory.

**Conclusion:** the refinements are individually validated and valuable, but they belong as
**opt-in, not monolithic defaults** — the tuned guarantees (extremity incentive, ring-below-
parity, bait/firehose demotion) depend on them being off or *carefully* combined. Composition
is not free: several refinements interact with the very defenses they sit beside. So the shipped
defaults keep them off; enabling any is a deliberate, per-deployment choice with its own
re-validation. (Budget #3 streaming credit landed gated alongside #1/#2/#4; §8 documents all four.)

## Suggested order
1. **E9** (hierarchical prior) — highest leverage, closes the one gap where §8 does §4's
   job; deterministic; CN + sim today.
2. **E5a** (exposure-attribution) — cheap, complements the loyalty defense with a
   logistical (not statistical) constraint.
3. **E2** (bias model + E-values) — turns §13.2 from bounded to calibrated; Coat has the
   ε-slice to prove it.
4. Then Tier 2 as simulator DGP knobs land; E-meta last as the synthesizing frame.
