# Whole-stack validation findings

A second, deeper validation pass beyond the real-data suite (`validate/`) and the
counterfactual simulator. Four axes were added, each written to *break* if a claim
doesn't hold. The failing tests below are **left failing on purpose** — they are the
findings, and mark where to legit improve. (`validate/FINDINGS.md` covers the
real-data claim reproductions; this file covers the stack's invariants/robustness/
dynamics.)

| Axis | Tests | Result |
|---|---|---|
| **A** invariants / metamorphic (`tests/test_properties.py`) | determinism, λ-distribution+budgets, Sybil-boundedness, permutation-invariance | 3 pass, **1 finding** |
| **B** robustness sweeps (`tests/test_robustness.py`) | CHORD>engagement across (d, k, aggregator)×seeds | pass (8/8 configs) |
| **C** adaptive adversaries (`tests/test_adaptive_adversary.py`, `validate/test_adaptive_ring.py`) | loyalty defense vs partial-approval ring; depth gate vs breadth vs forged signal | 2 pass, **1 finding** |
| **D** dynamic theory (`tests/test_dynamics.py`) | performativity instability, long-horizon boundedness, §9.3 controller | 2 pass, **1 finding** |

## What held (validated) ✅

- **Determinism, λ-is-a-distribution, bounded budgets, no-NaN/crash** over 40 random
  worlds; a **fresh single-reaction rater gets ≤ median influence** (Sybil-bounded).
- **CHORD beats engagement is robust**, not cherry-picked: true value **8/8 configs**,
  divisiveness 8/8, min margin +0.026.
- **The loyalty collusion defense is adaptively robust** (real CN data): a ring that
  partially-approves to dodge the super-loyal detector faces a fundamental tension —
  high approval inflates but is detected+removed; low approval evades but the attack
  fails on its own. The attacker's *best* outcome never beats its honest baseline.
- **The depth gate resists buying more breadth** (Goodhart-resistant to approval).
- **Long-horizon (60-window) run stays finite and bounded**; performativity increases
  instability *directionally*.

## Findings — where to legit improve ❌ (prioritized)

### F1. The §9.3 concentration controller was **inert** — ✅ FIXED (now wired)
`tests/test_dynamics.py::test_concentration_controller_response_is_applied` (now passes)
The loop called `controller.step(λ)` but **never read `controller.state`** — eigentrust
used `config.eigentrust_delta` and `rank` used `config.clamp_epsilon`, both fixed.
**Fix landed:** `fit_window` now runs on `replace(config, eigentrust_delta=…,
epsilon_min=…)` from `controller.state`, and `rank` floors ε at the controller's
`epsilon_min`. Verified: forcing a heavy-teleport δ flattens λ (Gini 0.20→0.05). Zero
collateral — the controller relaxes to the configured δ/ε in healthy operation, so it
is a no-op until Gini breaches the ceiling; no sim test moved.
**Secondary finding (dormancy):** in every scenario tried, Gini(λ_eff) sits ~0.06 —
the teleport floor + out-diversity + recycling already bound concentration far below
the 0.6 ceiling, so the controller, though now correctly wired, **never actually
triggers**. Open question for the math pass: is the ceiling mis-scaled (should it track
a much lower operating point, e.g. a multiple of the observed baseline), or is the
controller genuinely redundant belt-and-suspenders? Either way §9.3's prose overstated
an *active* guard that is really a dormant safety net.

### F2. B_LCB rankings were **not reproducible across input orderings** — ✅ FIXED (redesign)
`tests/test_properties.py::test_permutation_and_order_invariance` (now passes, ~0.96)
Relabelling ids + shuffling reaction order changed the ranking — mean Spearman only
~0.73. Root cause: B_LCB routed per-cluster reception through the **non-convex bilinear
MF embedding** (`<centroid_c, y_p>` + k-means on `x_u`), which lands in order-dependent
local optima. **Fix landed** (research-guided; the current `‖X‖²+‖Y‖²` regularizer is
the variational nuclear norm, so the object is convex in `L=XYᵀ` — the instability was
fitting at fixed low rank by ALS):
- **Reception is now empirical** — the IPW-weighted, EB-shrunk mean of each cluster's
  observed signed reactions (`chord/model/bridging.py`, prior = global μ), so B_LCB
  never touches `y_p`.
- **Clusters are deterministic** — a canonical, per-column-centred spectral split of the
  reaction matrix (`chord/model/spectral.py`), not k-means on the fragile `x_u`.
The MF is kept only for `V(u,p)` personalization. Result: Spearman 0.73→**0.96** (the
residual is the λ-weighting in IPW, kept for Sybil resistance), and on real data it is
*more* faithful — CN helpful/not-helpful **AUC 0.858→0.9996**, Polis cluster ARI 0.06→
**0.61**, `corr(B_LCB,support)` **0.81**.

### F3. The finer simulator claims are **init-fragile** — partially resolved; re-validation ongoing
The old worry was that fixing F2 would flip tuned sim results. It does move some — but
now for a *principled* reason (empirical reception + deterministic clusters), so this is
re-tuning, not a red herring. **Re-validation status after the F2 redesign:**
- ✅ property/reproducibility, ✅ CN keystone (AUC 0.9996), ✅ Polis clusters (ARI 0.61),
  ✅ Coat anchor, ✅ welfare robustness sweep, ✅ unit `test_bridging` (rewritten).
- ⏳ **Follow-on reds (not yet re-tuned):** the collusion loyalty defense's cluster-
  spread gate assumes a *balanced* 2-cluster split; the deterministic spectral split on
  the dense Community-Notes core is imbalanced (~17/1325 — a weak real opinion axis), so
  the gate reads the ring as one-cluster and doesn't fire (`test_community_notes_
  collusion`, `test_adaptive_ring`, `test_simulator_collusion`). The distributed-ring
  *attack* still transfers; the *defense* needs a balance-robust signal. Also
  `test_simulator_frontier` (interior-M optimum shifted, now marginal) and
  `test_simulator_anchor` (reception cap × empirical reception) need re-tuning. Left
  failing on purpose pending the collusion-defense re-work.

### F4. The depth defense is **evadable by forging the depth signal** *(design/research)*
`tests/test_adaptive_adversary.py::test_forged_depth_signal_evades_the_gate`
The anti-bait depth reward+gate trust the per-post depth *signal*, so a baiter that
forges a high depth score makes shallow content (value 0.431) beat genuine quality
(0.361). This is the documented §10 caveat, now demonstrated. *Fix* is a design/
research question: the depth signal must be **non-forgeable** (provenance, costly
signal, or a classifier the author can't set) — the same "signal integrity is
load-bearing" theme as identity forge-cost for Sybils.

### F5. No sharp §9.2 stability phase transition *(characterization, not a bug)*
Instability rises *gradually* with performativity (content-divisiveness std
0.049→0.076), not at a threshold, and `feed_churn` is ~high at all levels. The
two-timescale intuition holds directionally but the sim doesn't exhibit the clean
stable→oscillate boundary the theory describes; a sharper test would need a
performativity model with an explicit feedback gain.

## Suggested order for the "improve" pass
1. **F1** (wire the controller) — smallest, clearest correctness fix; do it with a
   re-validation of the sim tests it touches.
2. **F3** — decide the robust formulation / margins for the finer sim claims, so that
3. **F2** (SVD/order-invariant init) can land without red-herring regressions.
4. **F4** — scope a non-forgeable depth signal (design note in §10; not a code one-liner).
