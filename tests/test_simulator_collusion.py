"""Dive #2: the distributed (camouflaged) sybil ring — vulnerability and defense (§5/§10).

A *naive* ring (all puppets in one cluster) is contained by B_LCB's min-over-clusters.
A **distributed** ring — puppets camouflaged into every cluster, all boosting one
target — manufactures fake cross-cluster support and gets its target promoted; the §5
out-diversity λ does not help (the puppets are not single-target *raters*), and a
plain co-approval discount (`coordination_penalty`) is only partial (camouflage
dilutes it).

The working defense (`collusion_loyalty_penalty`, `CollusionTracker`): the one thing
the ring cannot hide is that the *same accounts* approve *every* one of the target's
posts, window after window, while being spread across opinion clusters. We discount an
author's bridged support by the fraction of its support coming from a super-loyal bloc
**gated by that bloc's opinion-cluster spread** — so a camouflaged ring (dispersed) is
penalized while a genuine single-cluster fanbase (coherent) is not. The mechanism it
targets is the ring's rank-1 lift of the shared `b_p`/`b_a` intercepts (see
SIMULATOR.md); the exploration-anchor cap and spectral spike-removal are the
principled complements documented there.
"""
import numpy as np
import pytest

from chord.config import ChordConfig
from chord.simulator import Simulator

SEEDS = (1, 2, 3, 4)


def _inflation(mode, K, q=0.4, loyalty=0.0):
    """Ring target's reach relative to a legitimate high-quality author (>1 ⇒ the
    ring out-promotes genuine content)."""
    tgt, uni = [], []
    for s in SEEDS:
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                          collusion_loyalty_penalty=loyalty)
        sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=s, adaptive_authors=False,
                        sybil_ring_size=K, ring_mode=mode, ring_target_quality=q)
        r = sim.run("chord", n_windows=8)
        tgt.append(r.tail("ring_target_reach", 4))
        uni.append(np.nanmean([m.reach_by_label.get("universal", 0) for m in r.metrics[2:]]))
    return float(np.nanmean(tgt)) / float(np.nanmean(uni))


@pytest.fixture(scope="module")
def inflations():
    out = {
        "naive": _inflation("naive", 30),
        "distributed": _inflation("distributed", 30),
        "distributed+defense": _inflation("distributed", 30, loyalty=3.0),
        "distributed_hiQ": _inflation("distributed", 30, q=0.8),
        "distributed_hiQ+defense": _inflation("distributed", 30, q=0.8, loyalty=3.0),
    }
    print("\n[sim collusion] ring inflation (target reach / legit-author reach), K=30:")
    for k, v in out.items():
        print(f"[sim collusion]   {k:<26} {v:.2f}x")
    return out


def test_distributed_ring_defeats_min_over_clusters(inflations):
    # The vulnerability: a camouflaged cross-cluster ring out-promotes genuine content,
    # far more than a naive single-cluster ring. min-over-clusters contains the naive
    # ring much better than the distributed one. (With empirical reception an *unexposed*
    # cluster abstains to the neutral prior grand=μ rather than being predicted-low from
    # the embedding, so a one-sided post sits nearer parity than before — but still well
    # below the distributed ring, which is the vulnerability this dive is about.)
    assert inflations["distributed"] > 1.4, "distributed ring should out-promote legit content"
    assert inflations["naive"] < inflations["distributed"] - 0.4, (
        "naive ring should be far more contained than the distributed one"
    )


def test_loyalty_defense_contains_the_ring(inflations):
    # The fix: continuous-loyalty × opinion-axis-spread discounting strips the ring's
    # manufactured advantage. For a low-quality target it drives reach below parity; for
    # a HIGH-quality target (which genuinely earns some reach) it removes the manufactured
    # lift, bringing a 1.7x ring back to ~parity.
    assert inflations["distributed+defense"] < 1.0, (
        f"loyalty defense should drive the ring below parity "
        f"({inflations['distributed']:.2f}x -> {inflations['distributed+defense']:.2f}x)"
    )
    assert inflations["distributed_hiQ+defense"] < inflations["distributed_hiQ"] - 0.4, (
        f"loyalty defense should strip the high-quality-target ring's advantage "
        f"({inflations['distributed_hiQ']:.2f}x -> {inflations['distributed_hiQ+defense']:.2f}x)"
    )
    assert inflations["distributed_hiQ+defense"] < 1.2, "high-Q ring reduced to ~parity"
