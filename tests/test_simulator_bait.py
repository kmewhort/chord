"""Dive #1: does the depth handling defeat bridging-bait? (§10, App C.4/C.5)

The simulator surfaced a weakness: B_LCB is quality-blind, so it can't tell genuine
quality from shallow "bridging-bait" (broadly-mildly-liked, low-value) content —
CHORD captured only ~1/3 of the oracle's true value and a bait author kept high
reach. The fix (config `depth_reward` + `depth_gate`, applied in the bridge factor):
promote genuine depth and multiplicatively gate a *shallow* post's positive bridged
support so it can't be crowned. This test shows, in the closed loop, that turning it
on raises delivered true value and reduces the bait author's reach.

Since F4 depth is an **earned** latent (estimated from the vouch channel), not an
author-set oracle feature — so a bait cannot forge it. The honest cost is a *softer*
signal than the old oracle depth: the effect is real and in the right direction, but
smaller (an estimate built from noisy cross-cluster vouches, not ground-truth quality).
The thresholds below reflect the earned signal.
"""
import numpy as np
import pytest

from chord.config import ChordConfig
from chord.simulator import Simulator

SEEDS = (1, 2, 3, 4)


def _run(depth_reward, depth_gate):
    tv, bait = [], []
    for s in SEEDS:
        # Isolate the depth mechanism from E9 (now default-on): the quality prior lifts B_LCB
        # for vouched authors, confounding the depth gate's isolated effect on the bait.
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                          hierarchical_prior=False,
                          depth_reward=depth_reward, depth_gate=depth_gate)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=s, adaptive_authors=False)
        r = sim.run("chord", n_windows=8)
        tv.append(r.tail("true_value", 4))
        bait.append(np.nanmean([m.reach_by_label.get("bridging_bait", 0) for m in r.metrics[2:]]))
    return float(np.nanmean(tv)), float(np.nanmean(bait))


@pytest.fixture(scope="module")
def onoff():
    off_tv, off_bait = _run(0.0, 0.0)
    on_tv, on_bait = _run(0.5, 0.5)
    print(f"\n[sim bait] depth OFF: true_value={off_tv:.4f} bait_reach={off_bait:.1f}")
    print(f"[sim bait] depth ON : true_value={on_tv:.4f} bait_reach={on_bait:.1f}")
    return {"off": (off_tv, off_bait), "on": (on_tv, on_bait)}


def test_depth_handling_raises_delivered_true_value(onoff):
    off_tv, _ = onoff["off"]
    on_tv, _ = onoff["on"]
    assert on_tv > off_tv + 0.015, (
        f"earned-depth handling should raise delivered true value ({off_tv:.4f} -> {on_tv:.4f})"
    )


def test_depth_handling_demotes_the_bait_author(onoff):
    _, off_bait = onoff["off"]
    _, on_bait = onoff["on"]
    assert on_bait < 0.8 * off_bait, (
        f"earned-depth handling should reduce the bait author's reach "
        f"({off_bait:.1f} -> {on_bait:.1f})"
    )
