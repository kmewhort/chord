"""Exploration-anchor cap: de-confounding + collusion complement (§6.2/§13.10).

The exploration pool floors uniform-random, unconfounded exposures. Capping each
cluster's reconstructed reception at the upper confidence bound of the author's
reception among those exposures (`config.exploration_anchor_cap`) does two things,
measured in the closed loop:

1. **De-confounds organic reception.** Organic exposure is selection-biased (people
   see what they already like), so organic reception overstates true reception; the
   cap removes that surplus and raises the *delivered* true value.
2. **Complements the loyalty defense against the distributed ring.** The cap alone
   cannot push a ring below parity — a camouflaged ring's near-origin target has the
   same *true* reception as genuine broad content (reception can't see quality) — but
   it removes the ring's common-mode inflation K-independently, so paired with the
   cluster-spread-gated loyalty penalty it contains the ring even at the higher
   exploration rates where the loyalty penalty alone weakens.

(Honest scope: at the small ε floor of the default config the anchor is data-starved
and simply does not bind — it misses safely, never false-positives. These tests
raise ε so the bound has evidence to bite.)
"""
import numpy as np
import pytest

from chord.config import ChordConfig
from chord.simulator import Simulator

SEEDS = (1, 2, 3, 4)


def _no_ring_true_value(cap):
    tv = []
    for s in SEEDS + (5,):
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                          epsilon_min=0.2, exploration_anchor_cap=cap)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=s, adaptive_authors=False)
        tv.append(sim.run("chord", n_windows=8).tail("true_value", 4))
    return float(np.nanmean(tv))


def _inflation(loyalty, cap, q=0.8):
    tgt, uni = [], []
    for s in SEEDS:
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                          epsilon_min=0.2, collusion_loyalty_penalty=loyalty,
                          exploration_anchor_cap=cap)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=s, adaptive_authors=False,
                        sybil_ring_size=30, ring_mode="distributed", ring_target_quality=q)
        r = sim.run("chord", n_windows=8)
        tgt.append(r.tail("ring_target_reach", 4))
        uni.append(np.nanmean([m.reach_by_label.get("universal", 0) for m in r.metrics[2:]]))
    return float(np.nanmean(tgt)) / float(np.nanmean(uni))


def test_exploration_cap_deconfounds_reception(_reporter=None):
    off = _no_ring_true_value(False)
    on = _no_ring_true_value(True)
    print(f"\n[sim anchor] no-ring delivered true_value: cap_off={off:.4f} cap_on={on:.4f}")
    assert on > off + 0.005, (
        f"exploration-anchor cap should de-confound and raise true value "
        f"({off:.4f} -> {on:.4f})"
    )


def test_cap_complements_loyalty_against_the_ring():
    loyalty_only = _inflation(3.0, False)
    loyalty_cap = _inflation(3.0, True)
    print(f"\n[sim anchor] high-Q ring inflation: loyalty={loyalty_only:.2f}x  "
          f"loyalty+cap={loyalty_cap:.2f}x")
    assert loyalty_cap < loyalty_only, (
        f"adding the cap should help contain the ring ({loyalty_only:.2f}x -> {loyalty_cap:.2f}x)"
    )
    assert loyalty_cap < 1.05, (
        f"loyalty + cap should hold the ring near/under parity ({loyalty_cap:.2f}x)"
    )
