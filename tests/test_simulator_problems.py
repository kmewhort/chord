"""Capstone: use the simulator to show CHORD actually solves the problems (§1, §8).

Each problem CHORD exists to fix is demonstrated end-to-end as a *counterfactual* —
it appears when the mechanism is off (or under an engagement baseline) and is
solved when CHORD's mechanism is on. This file covers the conserved author budget
(§8); the other problems are proven in sibling suites, indexed here:

  Problem                          | Solved-by                | Test
  ---------------------------------|--------------------------|--------------------------------
  engagement drives polarization   | bridging value (§4,§7)   | test_simulator_welfare.py
  the incentive breeds extremists  | bridging reward (§9.2)   | test_simulator_performativity.py
  a sybil ring promotes bad content| B_LCB min-over-clusters  | test_simulator_adversary.py
  firehosing floods the feed       | conserved budget (§8)    | (this file)
  rater-influence ring (real data) | out-diversity λ (§5)     | validate/test_signed_nets_eigentrust.py
  bridging keystone vs baselines   | shrinkage + nash (§4.2)  | validate/test_community_notes_keystone.py
"""
import numpy as np
import pytest

from chord.config import ChordConfig
from chord.simulator import Simulator

SEEDS = (1, 2, 3, 4)


def _firehose_vs_universal(budget_on):
    fh, uni = [], []
    for seed in SEEDS:
        if budget_on:
            cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0)
        else:  # budget effectively disabled — it never binds
            cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=1000.0, budget_max=1e6)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=seed, adaptive_authors=True)
        r = sim.run("chord", n_windows=8)
        fh.append(np.nanmean([m.reach_by_label.get("firehose", 0) for m in r.metrics[2:]]))
        uni.append(np.nanmean([m.reach_by_label.get("universal", 0) for m in r.metrics[2:]]))
    return float(np.mean(fh)), float(np.mean(uni))


@pytest.fixture(scope="module")
def firehose():
    on_fh, on_uni = _firehose_vs_universal(True)
    off_fh, off_uni = _firehose_vs_universal(False)
    out = {"on": (on_fh, on_uni), "off": (off_fh, off_uni)}
    print("\n[sim problems] firehose vs universal reach-per-post (budget on/off):")
    for k, (f, u) in out.items():
        print(f"[sim problems]   budget {k:<3} firehose={f:.1f} universal={u:.1f} ratio={f/u:.2f}")
    return out


def test_conserved_budget_dilutes_the_firehose(firehose):
    # §8: with the conserved budget binding, a high-volume firehose author earns
    # LESS reach per post than a quality author — and more so than with the budget
    # disabled. Turning the mechanism off narrows the gap (the problem returns).
    on_fh, on_uni = firehose["on"]
    off_fh, off_uni = firehose["off"]
    assert on_fh < on_uni, "budget on: quality author should out-reach the firehose per post"
    assert (on_fh / on_uni) < (off_fh / off_uni), (
        f"the budget should dilute the firehose more than no-budget "
        f"(on ratio {on_fh/on_uni:.2f} vs off {off_fh/off_uni:.2f})"
    )
