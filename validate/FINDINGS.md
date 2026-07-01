# Validation findings

This suite (whitepaper Appendix C) checks CHORD's static components against **real
public datasets**. The goal is honest validation: a claim that does *not* survive
contact with real data is the most useful result, because it tells us where to
iterate. Each finding below is a live `xfail` in the suite — run `pytest validate/
-rx` to see it with current numbers, and it will flip to `XPASS` (a loud signal) if
a future change makes the claim hold.

Numbers are from the committed slices (2025-02-22 Community Notes snapshot, Coat,
MovieLens-100K, Wikipedia-RfA, Polis openData) on `d=2`, and are averaged over MF
seeds where noted. They are indicative, not the last word — the point is the
*direction* and *ordering*, which are stable across seeds.

## What held ✅

| Claim | Dataset | Result |
|---|---|---|
| §6.2 IPW recovers a better ranking on a true MAR holdout | **Coat** | NDCG@5 0.714→0.718, AUC 0.657→0.663 with IPW. Real but **modest**. |
| §4.2 `B_LCB` tracks genuine cross-group support | **Polis** | Spearman(B_LCB, min-across-group vote) ≈ **+0.71** mean (0.37 / 0.80 / 0.97). |
| §6.2 the random ε-anchor buys identifiability | **MovieLens** | Unbiased NDCG@5 falls monotonically 0.769→0.723 as the anchor → 0. |
| §5 λ is a concentrated, valid influence distribution | **Wiki-RfA** | Row-stochastic T; N_eff ≪ n (quality-tracking, not uniform). |
| §4 `B_LCB` beats chance vs the deployed CN model | **Community Notes** | AUC 0.93 vs CN `currentStatus`. |

## Findings — claims that did **not** pan out ❌

### F1. §5 Sybil starvation is incomplete against a *ring* (Wiki-RfA)
`test_signed_nets_eigentrust::test_sybil_ring_cannot_buy_top_influence`

A collusion ring of *K* fresh accounts that all boost one target lifts that
target's influence to the top of the real-editor distribution — and it grows with
ring size:

| ring size K | target's percentile among real editors |
|---|---|
| 5 | 75th |
| 20 | 98th |
| 50 | 99.9th |
| 100 | 100th |

The teleport-floor eigentrust *does* starve the sybils themselves (each keeps only
`(1-δ)/n` floor mass), which is the invariant the existing unit test checks. But the
**target** harvests the sybils' redirected baseline mass: each sybil votes for
exactly one author and (row-stochastic) hands over its full unit, while honest
boosters split their trust across many candidates. So single-purpose rings
concentrate more transported trust on their target than diffuse honest support does.

**Implication.** §5's Sybil resistance leans entirely on the identity port's
forge-cost (making *K* identities expensive) — which lives *outside* the §5 math.
The whitepaper presents teleport-floor eigentrust as itself Sybil-starving; on real
signed data that is only true for the sybils, not their target. Options to probe:
degree-aware down-weighting of single-target raters, capping incoming trust per
author from low-λ raters, or making the forge-cost dependency explicit in §5.

### F2. §4 keystone adds no value over naive averaging on Community Notes
`test_community_notes_keystone::test_blcb_recovers_community_notes_helpfulness`

Predicting CN's own `CURRENTLY_RATED_HELPFUL` status on a dense k-core slice
(2,412 notes, 152k ratings, 1,342 raters):

| score | AUC vs CN status |
|---|---|
| naive mean signed rating | **0.9994** |
| `b_p` (scalar note intercept) | 0.9530 |
| `B_LCB` (cross-cluster tested support) | 0.9265 |

`B_LCB` is beaten by both the naive mean **and its own cheap scalar pre-filter**.
The cross-cluster LCB pessimism penalty (§4.2) demotes notes rated by fewer
clusters, which *reduces* agreement with CN's decision relative to plain averaging.

**Caveats.** (a) The slice is 95% helpful (imbalanced), so all AUCs are high — the
*ordering* mean > b_p > B_LCB is the signal, not the absolute values. (b) CN's own
status is itself a bridging-MF output, so "mean rating ≈ status" partly reflects how
CN decides. Still: if the sophisticated keystone can't beat averaging at reproducing
the one deployed bridging system we can compare to, that warrants investigation —
is the LCB penalty miscalibrated, is `d=2`/2-cluster too coarse, or is the exposure
term (`n_cp`) doing harm when set uniformly? Next step: sweep `lcb_beta`, cluster
count, and add propensity-corrected `n_cp` from real exposure proxies.

### F3. §6/C.3b semi-synthetic IPW fails to help on MovieLens
`test_movielens_mnar::test_ipw_recovers_ranking_and_anchor_matters`

On a dense MovieLens slice with a synthetic in-group-alignment logging policy, IPW
(with the *true* logging propensity) did **not** improve — and slightly hurt — the
unbiased ranking: NDCG@5 0.756 (uncorrected) → 0.738 (IPW), Δ = −0.018.

**This is as much a finding about the harness as about IPW.** MovieLens has no
random-exposure block, so the "MAR" holdout is a random slice of observations that
are *already* MNAR (people rate what they watch). Correcting only the injected
logging policy adds variance without touching MovieLens's own selection bias, so it
can't improve a holdout that shares that bias. Coat — which *has* a real MAR block —
shows IPW helping (F-none; it held). **Lesson for the whitepaper:** C.3's
semi-synthetic recipe is only valid on a genuinely MAR/near-complete base matrix;
run on organically-MNAR data it is self-defeating. §6/C.3 should say so explicitly,
and the MovieLens row in the C.1 table should be qualified.

## Validated candidate fixes (prototyped in `validate/experiments/`)

After a literature survey (trust-metric + welfare/risk/robust-statistics math), the two
headline findings (F1, F2) were prototyped against the *same* real-data benchmarks. Both
are fixable, and the winning mechanisms are near-one-liners. The prototypes live in
`validate/experiments/` (exercised by `test_sybil_hardening.py` and
`test_keystone_variants.py`); **both fixes are now promoted into the core** — the F1/F2
finding-tests below (`test_signed_nets_eigentrust.py`, `test_community_notes_keystone.py`)
were flipped from documented `xfail`s into passing regression guards against the shipped
`chord/` code.

> **Shipped:** out-diversity transmit weight in `chord/rater/eigentrust.py`
> (`config.sybil_out_diversity`, default on); exposure-weighted shrinkage + `nash`
> aggregator in `chord/model/bridging.py` (`config.bridging_aggregator="nash"`,
> `bridging_shrinkage_n0`). Set the flags back to reproduce the original findings.

### Fix for F1 — out-diversity down-weighting neutralizes the ring
`test_sybil_hardening.py` re-runs the exact RfA ring K-sweep under three candidate tweaks to
`λ ← (1-δ)/n + δ·Tᵀλ`. Target percentile among real editors:

| defense | K=5 | K=20 | K=50 | K=100 |
|---|---|---|---|---|
| baseline (shipped) | 75 | 98 | 99.9 | **100** |
| **out-diversity** `Tᵀ(w·λ)`, `w`=norm. out-entropy | **0** | **0** | **0** | **0** |
| seeded teleport (TrustRank-style) | 0 | 0 | 0 | 0 |
| per-author clip `min(Tᵀλ, c)` alone | 75 | 98 | 99.9 | 99.9 |

**Out-diversity** is the clear winner: weighting each rater's *transmitted* mass by the
normalized entropy of its outgoing trust row sends a single-target rater (every ring puppet,
out-degree 1) exactly **zero** mass, so the target harvests nothing — at *no* cost to honest
ranking (**Spearman(baseline λ, out-div λ) = 1.000** on the clean graph, because real editors
approve many authors and keep near-full weight). One line in `build_trust_matrix`/`eigentrust`.
Seeded teleport (Kamvar 2003 pre-trusted peers; Gyöngyi 2004 TrustRank; the theoretically
*necessary* asymmetry per Cheng–Friedman 2005/2006) also fully flattens it but needs a seed-
selection step. Per-author clip alone is insufficient as tuned. *Caveat:* an attacker can
raise a puppet's out-entropy by voting for several authors, but that splits its unit away from
the target — the defense forces exactly that trade-off. Blend `w` with a small floor so a
genuine one-vote newcomer isn't zeroed.

### Fix for F2 — drop the σ/√n penalty; shrink instead; prefer Nash/EDE over hard min
`test_keystone_variants.py` sweeps aggregator × penalty on both benchmarks.

**Community Notes** (balanced-subsample AUC, bar to beat = `b_p` = 0.956; naive mean = 0.999):

| variant | balanced AUC |
|---|---|
| `min` + σ/√n_count (≈ shipped B_LCB) | 0.870 |
| `min`, no penalty | 0.888 |
| `min` + James–Stein shrinkage | 0.998 |
| **`mean`/`nash` + James–Stein** | **0.999** |

**Polis** (Spearman with true min-across-*Polis-group* support — genuine bridging, distinct
from the mean):

| variant | mean Spearman |
|---|---|
| `min` + σ/√n_count (shipped) | 0.800 |
| `min`, no penalty | 0.934 |
| `min` + James–Stein | 0.964 |
| **`nash` + James–Stein** | **0.973** |

Two lessons: (1) **the σ/√n_count penalty was actively harmful** — removing it alone lifts
Polis tracking 0.80→0.93 and CN 0.870→0.888, confirming the diagnosis that it subtracts
*noise* (thin-cluster) not *risk*. (2) **Replace subtractive pessimism with empirical-Bayes
shrinkage** (James–Stein / DerSimonian–Laird): shrink each cluster mean toward the note mean
by how well-rated it is, so thin clusters regress to the mean and only *well-rated* dissent
pulls the score down — exactly "subtract risk, not noise." This beats `b_p` on CN (0.998 vs
0.956) and lifts Polis to 0.964. (3) On real multi-group structure (Polis) the **Nash /
geometric-mean aggregator — which is literally Polis's own "group-informed consensus" — beats
hard `min`** (0.973 vs 0.964); on CN it doesn't matter because CRH ≈ mean by construction.
*Honest caveat:* beating `b_p` on CN is partly regression-to-the-mean (CN's label *is* an
intercept threshold, so the mean is near-optimal there); the genuine bridging gain is the
Polis result, where the target differs from the mean.

**Recommended promotion into `chord/`:** (a) `rater/eigentrust.py`: out-diversity transmit-
weight `w` (+ optional seeded teleport bound to the identity port); (b) `model/bridging.py`:
replace `min_c[r̂ − β·σ/√(n+1)]` with `agg_ε(JamesStein-shrunk r̂_cp)`, default aggregator =
Nash/EDE, and make `n_cp` a propensity-corrected *exposure* count (§6), not a rating count.

## Weaker signals worth watching

- **Polis cluster reconstruction is inconsistent.** ARI between CHORD's clusters and
  Polis's validated groups: vTaiwan +0.59 (good), football-concussions +0.04,
  brexit +0.02 (near chance). Mean +0.21 clears the bar but two of three
  conversations barely separate. The `d=2` embedding recovers strong divides and
  misses subtle ones.
- **Polis divisiveness** `D(p)` tracks group spread on brexit (+0.65) and vTaiwan
  (+0.49) but is slightly negative on football-concussions (−0.05) — a
  low-conflict conversation where whitened `‖y‖²` is mostly noise.
