# CLAUDE.md

Orientation for agents working on this repo. Read this first, then skim
`README.md` (user-facing) and the whitepaper
`whitepaper/CHORD-whitepaper.md` (the spec — code is faithful to it
section-by-section). The paper is Markdown with LaTeX math; `whitepaper/compile.sh`
renders it to `whitepaper/CHORD-whitepaper.pdf` via pandoc + xelatex (see
`whitepaper/header.tex` and the `fit-codeblocks.lua` filter that scales the wide
Appendix D.1 diagram). Edit the `.md` and rebuild the PDF when you change the spec.

## What this is

A reference Python implementation of **CHORD**, a bridging, attention-economy
feed-ranking algorithm. It ranks content by *tested cross-cluster support net of
divisiveness* instead of engagement. The whitepaper is the source of truth; every
module maps to a paper section and cites it in its docstring (e.g. `§4.2`, `§6.2`).

The package is pure `numpy`/`scipy` — no ML frameworks, no network calls, no
datasets. Everything runs on modest hardware, by design (the fediverse target).

## Commands

```bash
pip install -e '.[test]'      # numpy, scipy, pytest
pytest -q                     # core suite (~110+ tests, ~8s) — testpaths=["tests"], offline
python -m examples.demo       # runnable demo of the three headline claims

pip install -e '.[validate]'  # + pandas, pyarrow, requests
python -m validate.fetch all  # download real datasets into validate/data (Git LFS)
pytest validate/ -rxX         # opt-in real-data validation (Appendix C); needs git-lfs
```

- Python ≥ 3.9. `pyproject.toml` sets `filterwarnings = ["error::RuntimeWarning"]`
  — **numpy RuntimeWarnings fail the suite**. If you introduce a divide-by-zero or
  overflow, guard it (see the `col_safe`/`row_safe`/`counts_safe` patterns).

## Repo layout (paper → code)

| Paper | Module | Notes |
|---|---|---|
| §4.1 | `chord/model/factorization.py` | Weighted biased MF via ALS (`MatrixFactorization` → `FactorizationResult`) |
| §4.1 | `chord/model/divisiveness.py` | Whitening, divide-weighting `A`, `D(p)=yᵀAy` |
| §4.2 | `chord/model/bridging.py` | `ClusterModel`, `BridgingScorer` → `B_LCB` |
| §5   | `chord/rater/eigentrust.py`, `quality.py`, `scout.py` | λ, quality-tracking, `q_scout` |
| §8   | `chord/rater/recycling.py` | `λ_eff` influence recycling |
| §6   | `chord/propensity/` | `base` interface, `models` (menu), `ipw`, `doubly_robust` |
| §7.1/7.3 | `chord/feed/value.py` | `value()` = V(u,p); factor vector + `blended_value` |
| §7.2 | `chord/feed/assembly.py` | `greedy_assemble` (submodular, budget, floor) |
| §8   | `chord/economy/budget.py`, `exploration.py` | `AuthorBudgetLedger`, `ExplorationPool` |
| §9.1 | `chord/loop.py` | `Chord` — `fit_window` (learning) + `rank` (serving) |
| §9.3 | `chord/monitor.py` | `N_eff`, `Gini`, `ConcentrationController`, endo/exo shift |
| §3/App D | `chord/ports/` | `base` protocols + `adapters` (crude defaults only) |
| App C.3 | `chord/eval/mnar_harness.py` | Semi-synthetic MNAR experiment |
| App C.4 | `chord/simulator/` | `population`, `content`, `response` (non-circular DGP), `rankers` (CHORD vs engagement/oracle/…), `metrics` (welfare), `engine` (ranker-driven loop + sybil-ring adversary). See `SIMULATOR.md`. |

`chord/types.py` holds the shared dataclasses (`Post`, `Reaction`, `Exposure`,
`ReactionKind`). `chord/config.py` holds `ChordConfig` (system/estimator params)
and `UserKnobs` (the free consumption knobs M/ρ/θ/ε — see the §12 wall below).

## The two entry-point flows (`chord/loop.py`)

- `Chord.fit_window(reactions, posts, exposures, ...)` — the per-window **learning
  plane**. Runs the §9.1 steps: inner actor–critic loop (IPW-weighted MF fit →
  whiten/`A`/`D` → clusters → quality-tracking λ, iterated `inner_iters` times),
  then `B_LCB`, scout, budget replenishment, Thompson updates, recycling → `λ_eff`,
  then the concentration controller. Produces a `WindowState`.
- `Chord.rank(user_id, candidates, knobs, n_slots)` — the per-request **serving
  plane**. Scores `V(u,p)` with the viewer's knobs, routes unseen posts to the
  exploration pool, and calls `greedy_assemble`. Requires a prior `fit_window`.

## Non-obvious invariants — don't silently break these

These are the subtle spots (several were bugs during the first build; all now have
tests and are documented in the whitepaper's own prose):

1. **EigenTrust `T` is row-stochastic + out-diversity transmit weight**
   (`rater/eigentrust.py`): each *rater's outgoing* trust sums to 1. Column-
   normalizing lets a one-puppet Sybil inherit its booster's full weight. Row
   normalization alone still lost to a *ring* (many one-vote puppets pooling their
   teleport-floor mass onto one target), so the iteration weights each rater's
   *transmitted* trust by the entropy of its row (`outgoing_diversity_weights`,
   `config.sybil_out_diversity`): a single-target rater forwards ≈0. Tests:
   `test_adversarial.py::{test_fresh_sybil_gets_minimal_trust,test_sybil_ring_cannot_harvest_influence}`,
   `validate/test_signed_nets_eigentrust.py`. Validated on RfA (Appendix C.5).
2. **IPW weights are rescaled to mean 1** (`propensity/ipw.py`, `normalize=True`).
   λ is a normalized distribution, so raw weights are O(1/|E|); the MF's fixed
   embedding regularization would then swamp the data and collapse embeddings to
   zero. SNIPW is scale-invariant so this changes nothing statistically.
3. **`ρ` is applied exactly once** (`feed/value.py`): `D(p)` uses `A` at `ρ=1`;
   `ρ` multiplies the penalty term in `V`. Don't also scale `A` by `ρ`.
4. **Exploration floor preserves positivity**: `greedy_assemble` uses
   `ceil(εN)` (hard floor never rounds to 0); the simulator's randomized anchor
   uses *stochastic* rounding (correct expected rate). Both protect `π ≥ ε > 0`,
   which §6.2 identifiability depends on. `ChordConfig.epsilon_min > 0` is
   validated for the same reason.
5. **Budget replenishment is rectified + clamped** (`economy/budget.py`):
   `[Φ]₊` and clip to `[0, budget_max]`. A divisive post must not drain `B₀` or
   drive `B(a)` negative.
6. **Consumption vs. authority wall (§12)**: `UserKnobs` exposes only M/ρ/θ/ε.
   Earned quantities (λ, `q_scout`, `B(a)`) are **never** user-settable — keep it
   that way. The author budget binds to the *identity* port, not the raw account.
7. **`B_LCB` is empirical per-cluster reception + shrinkage, not a subtractive penalty
   and not a bilinear reconstruction** (`model/bridging.py`): each cluster's reception
   is the **IPW-weighted empirical mean of that cluster's observed signed reactions**
   (`cluster_reception`), shrunk toward the prior `grand = μ` by `n_cp/(n_cp+n0)`
   (empirical Bayes, `n_cp` = IPW-weighted rating evidence), then aggregated
   (`config.bridging_aggregator`, default `"nash"`; `"min"`/`"ede"` available). It used
   to reconstruct reception as `μ+b̄_c+b_a+b_p+<x̄_c,y_p>` — routing it through the
   non-convex MF embedding made B_LCB order-irreproducible (~0.73 Spearman); the
   empirical form is reproducible (~0.96) and *more* faithful on real data (CN AUC
   0.9996). Opinion **clusters are deterministic** (`model/spectral.py`: per-column-
   centred canonical spectral split), not k-means on `x_u`. The MF stays only for
   `V(u,p)` personalization. The old `min_c[r̂ − βσ/√(n+1)]` penalized *under-sampled*
   clusters and lost to `b_p`/naive-mean. Validated on Community Notes / Polis (App C.5).
   Tests: `test_bridging.py`, `test_properties.py`. (Re-validation of the collusion
   defense under the new clustering is in progress — see `VALIDATION_FINDINGS.md` F3.)

## Conventions

- **Ids are opaque hashables** (`Id = Hashable`) — ints in tests, strings in the
  simulator. Don't assume a type.
- **Determinism**: anything stochastic takes a `seed`; `np.random.default_rng` only.
  Tests rely on this. `Date`/wall-clock is never used for logic.
- **Ports pattern**: each port is a `Protocol` in `ports/base.py` with a crude
  default adapter in `ports/adapters.py`. Rich adapters (verified/ZK identity,
  Solid/ATProto pods, external Polis clustering) are intentionally *not* built —
  they're integration slots. Add new adapters alongside the defaults; don't make
  the core depend on them.
- **Docstrings cite the paper.** When you add/change math, cite the section and, if
  it diverges from the paper, update the whitepaper prose too (the user prefers
  fixes baked into the relevant section, not a separate errata list).

## Testing approach

- `tests/conftest.py` provides the canonical bipolar toy world (10 users, 2 poles,
  3 posts: universal A, partisan B/C). Reuse it.
- Unit tests per module; **broader tests** assert the paper's *claims* reproduce:
  `test_loop_integration.py` (keystone + M dial + budget end-to-end),
  `test_simulator.py` (closed-loop stability, firehose dilution, exploration
  sustained), `test_mnar.py` (IPW recovers ranking under MNAR; identifiability
  fails as the anchor → 0), `test_adversarial.py` (Sybil/brigade). The simulator's
  **counterfactual** suite proves CHORD beats the engagement baseline end-to-end:
  `test_simulator_welfare.py` (more true value / less polarization),
  `test_simulator_performativity.py` (incentive doesn't breed extremists),
  `test_simulator_adversary.py` (ring contained), `test_simulator_problems.py`
  (budget suppresses firehose). See `chord/simulator/SIMULATOR.md` for the
  problem→test map.
- MNAR/simulator assertions are averaged over seeds to test the *systematic*
  effect, not a single noisy draw — keep that if you touch them.

## What's deliberately not here

- No dataset loaders *in the core* (`chord/`). Real-data validation lives in the
  separate opt-in `validate/` suite (Appendix C): adapters for Coat, Polis,
  MovieLens, Wikipedia-RfA, and a Community Notes slice, mapping each to
  `chord.types`, with datasets committed under `validate/data` via Git LFS. It
  found (and then fixed, see §5/§4.2 + Appendix C.5) two real weaknesses — the
  Sybil-ring harvest and the old subtractive `B_LCB`. `chord/eval/` remains the
  semi-synthetic MNAR harness only.
- No serving/HTTP layer, no Mastodon/ATProto integration — CHORD is the
  valuation-and-allocation core; retrieval and presentation stay upstream/downstream.
- Rich port adapters (see Conventions).

## Working agreement (from the task setup)

- Develop on branch **`claude/whitepaper-project-impl-245yuo`**; push there.
- Don't open a PR unless the user asks.
- Run `pytest -q` before committing; keep the suite green.
