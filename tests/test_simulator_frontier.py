"""Dive #3: the bridging↔satisfaction frontier and the M dial (§7.1, App C.4).

CHORD at M=1 (pure bridging) pays a satisfaction cost vs an engagement baseline.
Sweeping the master dial M traces the tradeoff — and shows the cost is partly
self-inflicted: **M=1 is Pareto-dominated by an interior M≈0.7**, which delivers
*more* true value and *less* divisiveness at *no* satisfaction cost, because keeping
a little personalization surfaces content that is both liked and genuinely good.
0.7 is already the product default (`UserKnobs.M`), so the lesson is: don't push the
dial to the pure-bridging extreme. Linear scalarization (the M dial) reaches this
interior optimum here, so a heavier control (a satisfaction-floor ε-constraint in
feed assembly) is not needed for this world — noted as a considered option.
"""
import numpy as np
import pytest

from chord.config import ChordConfig, UserKnobs
from chord.simulator import Simulator

SEEDS = (1, 2, 3, 4, 5)


def _run(M):
    agg = {k: [] for k in ("true_value", "divisiveness", "toxicity", "satisfaction")}
    for s in SEEDS:
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=s,
                        adaptive_authors=False, knobs=UserKnobs(M=M))
        r = sim.run("chord", n_windows=8)
        for k in agg:
            agg[k].append(r.tail(k, 4))
    return {k: float(np.nanmean(v)) for k, v in agg.items()}


@pytest.fixture(scope="module")
def frontier():
    pts = {M: _run(M) for M in (0.5, 0.7, 1.0)}
    print("\n[sim frontier] M    true_value  divisiveness  satisfaction")
    for M, m in pts.items():
        print(f"[sim frontier] {M:.2f}   {m['true_value']:.4f}      {m['divisiveness']:.4f}"
              f"        {m['satisfaction']:.4f}")
    return pts


def test_interior_M_dominates_pure_bridging(frontier):
    # M≈0.7 Pareto-dominates M=1 on the objectives that matter, at no satisfaction cost.
    mid, pure = frontier[0.7], frontier[1.0]
    assert mid["true_value"] > pure["true_value"], (
        f"M=0.7 true value ({mid['true_value']:.4f}) should exceed M=1's "
        f"({pure['true_value']:.4f})"
    )
    assert mid["divisiveness"] <= pure["divisiveness"] + 1e-3, (
        f"M=0.7 divisiveness ({mid['divisiveness']:.4f}) should not exceed M=1's "
        f"({pure['divisiveness']:.4f})"
    )
    assert mid["satisfaction"] >= pure["satisfaction"] - 1e-3, (
        f"M=0.7 satisfaction ({mid['satisfaction']:.4f}) should be at least M=1's "
        f"({pure['satisfaction']:.4f})"
    )
