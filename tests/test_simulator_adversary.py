"""Adversary containment in the loop: can a sybil ring buy promotion? (§10, App C.4)

A ring of single-target sybil raters boosts a mediocre target author's posts every
window — the reactions the ranker then learns from. The question is consequential:
does the ring get its target *shown more* over time?

Under engagement the ring works — inflating the target's predicted personalized
approval buys it reach that holds as the ring grows. Under CHORD it backfires: the
sybils read as an outlier group, so B_LCB's min-over-clusters keeps the target's
bridged support (and thus its reach) low — and reach *falls* as the ring grows.

(This exercises the keystone's resistance to a content-boost ring; the distinct
rater-influence ring that the §5 out-diversity weight fixes is validated on real
signed votes in validate/test_signed_nets_eigentrust.py.)
"""
import numpy as np
import pytest

from chord.config import ChordConfig
from chord.simulator import Simulator

SEEDS = (1, 2, 3)


def _target_reach(ranker, K):
    vals = []
    for seed in SEEDS:
        # Isolate the out-diversity/loyalty ring defense from E9's quality prior (default-on):
        # sim ring puppets vouch *honestly* (vouches track truth.quality), so a bigger ring
        # gives a genuine-quality target more vouch evidence → E9 credits its merit → reach
        # grows with K, masking the approval-inflation defense under test. Real *fake* vouches
        # are contained by the vouch channel's own collusion defenses (§10, §13.11).
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                          hierarchical_prior=False)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=seed,
                        adaptive_authors=False, sybil_ring_size=K)
        r = sim.run(ranker, n_windows=8)
        vals.append(r.tail("ring_target_reach", 4))
    return float(np.nanmean(vals))


@pytest.fixture(scope="module")
def reach():
    grid = {(rk, K): _target_reach(rk, K)
            for rk in ("chord", "engagement") for K in (15, 45)}
    print("\n[sim adv] sybil-ring target reach-per-post:")
    for rk in ("chord", "engagement"):
        print(f"[sim adv]   {rk:<11} K=15 {grid[(rk,15)]:.1f}   K=45 {grid[(rk,45)]:.1f}")
    return grid


def test_chord_contains_the_ring_engagement_does_not(reach):
    # CHORD gives the ring's target far less traction than engagement does.
    assert reach[("chord", 45)] < 0.75 * reach[("engagement", 45)], (
        f"CHORD did not contain the ring: target reach {reach[('chord',45)]:.1f} vs "
        f"engagement {reach[('engagement',45)]:.1f}"
    )


def test_scaling_the_ring_does_not_buy_reach_under_chord(reach):
    # The headline: adding colluders must not buy more promotion under CHORD.
    assert reach[("chord", 45)] <= reach[("chord", 15)] + 1e-6, (
        f"a bigger ring bought MORE reach under CHORD "
        f"({reach[('chord',15)]:.1f} -> {reach[('chord',45)]:.1f})"
    )
