# CHORD

CHORD (Cross-cluster Harmonized Optimization of Reception and Dissonance) is an
experimental, first-draft algorithm that aims to solve some of the major problems
with today's *anti*-social networks.

The big platforms use dark-pattern mathematics to maximize engagement — and
engagement is a trap: the content that provokes the most reactions is the content
that divides us most. Outrage and tribalism are high-variance bets — half the room
loves them, half hates them — and that very split is what the reaction-counter
rewards. However neutral the feed looks, its math quietly pays out for division.

CHORD takes the opposite approach. Inspired by the
[Ethelo algorithm](https://github.com/Ethelo/ethelo-os-engine), instead of
rewarding whatever provokes the loudest reaction, it optimizes for broad support
*across* a community's divides — content earns reach by bridging disagreement
rather than inflaming it. The goal is an algorithm tuned for positive social
interaction, not attention at any cost.

---

This repository is a reference Python implementation of the CHORD whitepaper
([`strength-ranked-attention-whitepaper-2.md`](strength-ranked-attention-whitepaper-2.md)) —
the full valuation-and-allocation layer, plus a closed-loop simulator and an
evaluation harness. On top of the bridging core it layers three attention-economy
mechanisms: quality-weighted raters (a discriminating reaction counts more than a
reflexive one), a conserved author visibility budget (posting more dilutes rather
than multiplies reach), and a commons-funded exploration pool (unproven newcomers
get a fair audition).

The one-line architecture (whitepaper Appendix B):

> Rank by tested cross-cluster support net of weighted divisiveness; weight raters
> by quality-tracking not variance; price authors by a strength-replenished
> conserved budget; audition the unproven from a floored commons pool that doubles
> as the identifiability anchor; correct exposure MNAR with doubly-robust
> propensity weighting; stabilize the coupled estimator as two-timescale stochastic
> approximation held in a monitored bounded regime; and expose M/ρ/θ/ε as knobs
> while keeping all authority earned.

## Install

```bash
pip install -e .            # numpy + scipy
pip install -e '.[test]'    # + pytest
```

## Quick start

```python
from chord import Chord, ChordConfig, UserKnobs, Post, Reaction, Exposure
from chord.propensity import UniformExplorationModel

# A tiny bipolar world: users 0-4 "left", 5-9 "right".
# Post A is universal, B is partisan-left, C is partisan-right.
posts = {"A": Post("A", "auth1"), "B": Post("B", "auth2"), "C": Post("C", "auth3")}
rx, exps = [], []
for u in range(10):
    left = u < 5
    for pid, val in [("A", 1.0), ("B", 1.0 if left else -1.0), ("C", -1.0 if left else 1.0)]:
        rx.append(Reaction(u, pid, val, timestamp=float(u)))
        exps.append(Exposure(u, pid, propensity=0.5))

chord = Chord(ChordConfig(d=4, n_clusters=2),
              propensity_model=UniformExplorationModel(0.5))
chord.fit_window(rx, posts, exps)

# Pure bridging (M=1): the universal post wins.
print(chord.rank(0, list(posts.values()), UserKnobs(M=1.0), n_slots=3))  # ['A', ...]

# Engagement-like (M=0) for a left user: their in-group post B beats C.
print(chord.rank(0, list(posts.values()), UserKnobs(M=0.0), n_slots=3))
```

Run the closed-loop simulator (whitepaper Appendix C.4):

```python
from chord.simulator import Simulator
result = Simulator(n_users=30).run(n_windows=8)
for m in result.metrics:
    print(m.window, round(m.gini_lambda, 3), round(m.exploration_rate, 3))
```

Or run the bundled demo:

```bash
python -m examples.demo
```

## How the code maps to the whitepaper

| Paper section | Module | What it implements |
|---|---|---|
| §4.1 relation model | `chord/model/factorization.py` | Weighted, biased matrix factorization via ALS (with author term + partial pooling) |
| §4.1 divisiveness | `chord/model/divisiveness.py` | Whitening, the divide-weighting matrix `A`, `D(p)=yᵀAy` |
| §4.2 bridged support | `chord/model/bridging.py` | Per-cluster reconstruction and the `B_LCB` tested-breadth lower confidence bound |
| §5 rater weighting | `chord/rater/eigentrust.py`, `quality.py` | Damped teleporting EigenTrust + quality-tracking (never inverse-variance) |
| §5 scout precision | `chord/rater/scout.py` | `q_scout`: reward being early on eventual winners |
| §8 influence recycling | `chord/rater/recycling.py` | `λ_eff` boosts the under-served, damps the satisfied |
| §6 identifiability | `chord/propensity/` | IPW/SNIPW, the propensity menu (§6.3), doubly-robust wrapping |
| §7.1 value / §7.3 factors | `chord/feed/value.py` | `V(u,p)`, the M dial, the factor vector |
| §7.2 feed assembly | `chord/feed/assembly.py` | Greedy submodular constrained selection (1−1/e), author budget, diverse-approval coverage, exploration floor |
| §8 author budget | `chord/economy/budget.py` | Conserved, strength-replenished, identity-bound |
| §8 exploration pool | `chord/economy/exploration.py` | Base-rate-calibrated Thompson auditions, saturation-closed |
| §9.1 the loop | `chord/loop.py` | Per-window learning (steps 1–7) + per-request serving |
| §9.3 stability monitor | `chord/monitor.py` | `N_eff`, `Gini(λ)`, the concentration controller, endogenous/exogenous shift split |
| §3 / App. D ports | `chord/ports/` | Identity, Preference, Signal, Candidate, Partition adapters (crude defaults) |
| App. C.3 MNAR harness | `chord/eval/mnar_harness.py` | Semi-synthetic propensity experiment |
| App. C.4 simulator | `chord/simulator/` | Agent-based closed loop: population, response model, adaptive authors |

## What the tests demonstrate

`pytest` runs 100+ unit tests plus broader/integration tests. The headline
paper claims each have a test that reproduces them:

- **The keystone (§4):** a universal post out-bridges partisan posts on `B_LCB`,
  and an untested cluster drives `B_LCB` down (a post is not crowned as bridging
  until it survives contact with people who would dislike it).
  → `tests/test_bridging.py`, `tests/test_loop_integration.py`
- **The §5 pathology:** quality-tracking does **not** reduce to inverse-variance
  weighting, so a reflexive extremist is not over-weighted.
  → `tests/test_rater.py::test_quality_tracking_not_inverse_variance`
- **Identifiability (§6):** under an in-group-over-exposing logging policy,
  IPW-corrected fitting recovers the true bridging ranking better than uncorrected
  fitting; and as the randomized exploration anchor → 0, identifiability *fails*.
  → `tests/test_mnar.py`
- **Attention economy (§8):** a firehose author's reach-per-post is diluted by the
  conserved budget; sockpuppet sharding gains nothing (budget binds to identity).
  → `tests/test_simulator.py`, `tests/test_adversarial.py`
- **Stability (§9.3):** across the closed loop the concentration controller keeps
  `Gini(λ)` bounded and `N_eff` from collapsing, and exploration stays floored.
  → `tests/test_simulator.py`
- **Adversarial robustness (§10):** brigading creates a split distribution that
  `B_LCB` penalizes; a Sybil boosted by one colluder gets less trust than an
  honestly-approved author. → `tests/test_adversarial.py`

```bash
pytest -q
```

## Design notes / honest residuals

This implementation is faithful to the whitepaper's *structure* and reproduces its
qualitative claims on synthetic data. Consistent with §13, several things are
"named, not solved": the propensity model is the softest load-bearing wall
(unobserved confounding is bias, not variance, and no estimator here removes it);
global convergence of the coupled estimator is monitored, not proven; and the
high-precision collusive clique is not defeated. The ports ship only the crude
default adapters — rich adapters (verified/ZK identity, portable pods, external
Polis clustering) are left as integration slots, exactly as §3/§11 describe.

## License

AGPL-3.0-or-later (matching the Ethelo engine referenced by the whitepaper).
