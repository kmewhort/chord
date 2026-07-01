"""Dive #2: the distributed (camouflaged) sybil ring, and a partial defense (§5/§10).

The simulator surfaced a real vulnerability the earlier fixes miss. A *naive* ring
(all puppets in one opinion cluster) is contained by B_LCB's min-over-clusters. But
a **distributed** ring — puppets camouflaged into every cluster (each rates genuine
content like its host cluster, then boosts the target) — manufactures fake
cross-cluster support and gets its target promoted; the §5 out-diversity λ weight
does not help (the puppets are not single-target *raters*).

The `coordination_penalty` (config) discounts a post whose approvers co-approve in
lockstep (COCM / pairwise-bounded idea). It is an honest **partial** mitigation: it
reduces the distributed ring's reach but does not fully contain it, because
camouflage dilutes the co-approval signal. Fully neutralizing a camouflaged ring
needs stronger machinery (spectral spike-removal on the co-approval graph, co-approval
community detection, or COCM pairwise-bounded matching) — a documented open problem,
see chord/simulator/SIMULATOR.md.
"""
import numpy as np
import pytest

from chord.config import ChordConfig
from chord.simulator import Simulator

SEEDS = (1, 2, 3)


def _reach(mode, K, coord_pen=0.0):
    vals = []
    for s in SEEDS:
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                          coordination_penalty=coord_pen)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=s, adaptive_authors=False,
                        sybil_ring_size=K, ring_mode=mode)
        vals.append(sim.run("chord", n_windows=8).tail("ring_target_reach", 4))
    return float(np.nanmean(vals))


@pytest.fixture(scope="module")
def reaches():
    out = {
        "naive": _reach("naive", 30),
        "distributed": _reach("distributed", 30),
        "distributed+coord": _reach("distributed", 30, coord_pen=4.0),
    }
    print(f"\n[sim collusion] ring-target reach (K=30): "
          f"naive={out['naive']:.1f}  distributed={out['distributed']:.1f}  "
          f"distributed+coord_penalty={out['distributed+coord']:.1f}")
    return out


def test_distributed_ring_defeats_min_over_clusters(reaches):
    # The vulnerability: a camouflaged cross-cluster ring gets far more reach than a
    # naive one, because it fakes the cross-cluster support min-over-clusters checks.
    assert reaches["distributed"] > 1.6 * reaches["naive"], (
        f"distributed ring ({reaches['distributed']:.1f}) should beat the contained "
        f"naive ring ({reaches['naive']:.1f}) — the vulnerability is not reproduced"
    )


def test_coordination_penalty_partially_contains_the_ring(reaches):
    # The partial defense: coordination discounting measurably reduces the ring's
    # reach (it does NOT fully contain it — see the module docstring).
    assert reaches["distributed+coord"] < reaches["distributed"] - 10.0, (
        f"coordination penalty should reduce the distributed ring's reach "
        f"({reaches['distributed']:.1f} -> {reaches['distributed+coord']:.1f})"
    )
