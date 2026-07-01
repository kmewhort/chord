# The CHORD simulator — end-to-end, counterfactual, non-circular

The feedback loop (§9) cannot be tested on any fixed dataset: a static archive
can't respond to the ranker's own allocations. This agent-based simulator is the
only way to exercise it — and it is built to *prove CHORD solves the problems it
claims to*, not merely to run without crashing.

## Three design commitments

1. **Counterfactual.** Each ranker (`rankers.py`) runs its *own* closed loop on the
   same seeded world, so results are "what if we had ranked this way." Baselines:
   `engagement` (the same biased MF, but ranked by predicted *personalized* approval
   — what "optimize engagement" means), `chronological`, `random`, and `oracle`
   (ranks by the hidden true value — the ceiling). CHORD is measured against them.

2. **Non-circular DGP** (`population.py`, `content.py`, `response.py`). The world is
   deliberately *not* the model the estimator fits, so results aren't self-fulfilling:
   - a hidden opinion axis the `d=2` model can't represent (`d_true = d+1`);
   - per-post **toxicity** (drives whether users react at all, and sharpens the
     divide) and **quality** (genuine value, which barely moves reactions) — so
     *engagement ≠ value* and **bridging-bait** (broad but shallow) can exist;
   - a saturating nonlinear response.
   If the DGP shared the estimator's shape, "CHORD recovers it" would be trivially
   true and meaningless (the same lesson as Appendix C.5's F3).

3. **Ground-truthed** (`metrics.py`). Because the world is synthetic we score what no
   live system can see: true bridged value delivered (`quality × min-cluster
   reception`), toxicity/divisiveness exposure, satisfaction, and estimator recovery
   of the true opinion geometry (rotation-free pairwise-distance correlation).

## What it shows (seed-averaged; see the tests for exact numbers)

| Problem CHORD targets | Counterfactual result | Test |
|---|---|---|
| Engagement drives polarization | CHORD delivers **more true value** (0.11 vs 0.06), **less divisiveness** (0.55 vs 0.73) and **less toxicity** (0.28 vs 0.38) than engagement, at a satisfaction cost (0.50 vs 0.58 — the honest tradeoff); it also **recovers the opinion geometry better** (0.64 vs 0.46), because its exploration anchor keeps the estimate identifiable (§6.2 in-loop). | `tests/test_simulator_welfare.py` |
| The incentive breeds extremists | With strategic authors (a (1+1)-ES chasing reach), **engagement entrenches partisan extremity** over time while **CHORD does not** — its reward gradient doesn't point at the poles (§9.2). | `tests/test_simulator_performativity.py` |
| A sybil ring promotes bad content | Under engagement a ring buys its target reach that **grows** with ring size; under CHORD it **backfires** (reach falls 46→27 as the ring grows) — the sybils read as an outlier group and B_LCB's min-over-clusters keeps the target's bridged support low. | `tests/test_simulator_adversary.py` |
| Firehosing floods the feed | The **conserved author budget** (§8) dilutes a high-volume firehose author's reach-per-post below a quality author's; disabling the budget narrows the gap (the problem returns). | `tests/test_simulator_problems.py` |

The two fixes shipped earlier are proven on *real* data rather than in-sim (the
faithful court of appeal): the §5 out-diversity rater-influence defense on Wikipedia
RfA, and the §4.2 shrinkage keystone on Community Notes / Polis — see
`validate/FINDINGS.md`. A notable in-loop finding: the out-diversity λ fix does *not*
by itself stop a ring that boosts a target's *posts* (that path runs through the MF
reception, gated by the sybils' own floor λ); what contains it in the loop is the
keystone's min-over-clusters, above.

## Weaknesses the simulator surfaced — and what we did

Running the loop against baselines and adaptive adversaries turned up three real
weaknesses; each was then chased down (tests in `tests/test_simulator_*`):

1. **Quality-blindness / bridging-bait — fixed.** B_LCB rewards broad approval and
   couldn't tell genuine quality from shallow broad-appeal, so CHORD delivered only
   ~1/3 of the oracle's true value and a bait author kept high reach. The anti-bait
   depth handling (`config.depth_reward` + `depth_gate`, §10) promotes genuine depth
   and multiplicatively gates a shallow post's positive bridged support. Result: true
   value 0.12→0.18, bait reach 27→5. (`test_simulator_bait.py`)
2. **Distributed (camouflaged) sybil ring — defended.** A naive ring is contained by
   min-over-clusters, but a ring that camouflages puppets into every cluster and
   boosts one target fakes cross-cluster support (inflation — target reach ÷
   legit-author reach — of 2.83× at K=30), and neither the §5 out-diversity λ nor a
   plain co-approval `coordination_penalty` stops it (camouflage dilutes co-approval).
   **Mechanism** (from the RMT/mechanism-design research): the attack is a *rank-1
   common-mode lift* of the shared `b_p`/`b_a` intercepts — the puppets scatter across
   clusters so their directional pull cancels and only the shared positive residual
   survives, lifting every cluster's reception equally. **Defense** (`CollusionTracker`
   + `config.collusion_loyalty_penalty`): discount an author by the fraction of its
   support coming from a super-loyal bloc (accounts that approve *every* one of its
   posts over time — the one act camouflage can't hide) **gated by that bloc's
   opinion-cluster spread**, so a dispersed ring is penalized while a coherent
   single-cluster fanbase is not. Contains the ring at inflation 0.82× (and 0.63× for
   a high-quality target the depth defense can't catch), negligible collateral.
   (`test_simulator_collusion.py`)

   *Principled complements (validated in the literature, not yet built):* the
   **exploration-anchor cap** — cap each cluster's organic reception at the upper
   confidence bound of its unconfounded ε-exploration reception (K-independent,
   provably non-collateral, but needs more exploration traffic than this sim floors);
   and **spectral spike-removal** — deflate the rank-1 boost block from the MF residual
   before re-fitting the intercepts (detection *strengthens* with K, σ_spike ∝ √K).
   RPCA is the wrong tool (the rank-1 boost lands in the low-rank part).
3. **Bridging↔satisfaction tradeoff — characterized** (see `test_simulator_frontier.py`).

## Layout

```
population.py   synthetic agents (bipolar + hidden axis, heterogeneous reactivity/selectivity)
content.py      author-agents: archetypes (universal/partisan/firehose/toxic/bait), hidden
                toxicity+quality, (1+1)-ES style adaptation (performativity)
response.py     the DGP P(react | x_u, y_p): toxicity-driven engagement, quality-decoupled value
rankers.py      Ranker protocol + CHORD / engagement / chronological / random / oracle
metrics.py      ground-truth welfare, recovery, Welfare helper
engine.py       ranker-driven closed loop; welfare WindowMetrics; compare(); sybil-ring adversary
```

Run a comparison yourself:

```python
from chord.simulator import Simulator
sim = Simulator(n_users=40, seed=1, adaptive_authors=False)
res = sim.compare(["oracle", "chord", "engagement", "random"], n_windows=8)
print({k: round(v.tail("true_value"), 3) for k, v in res.items()})
```
