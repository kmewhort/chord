"""Dynamic-theory validation (§9.2 stability, §9.3 concentration controller).

These exercise the *dynamic* claims that the static tests can't, and are written to
break if the claims don't hold. They surfaced two findings (see the failing tests):
the §9.3 controller is inert, and there is no sharp §9.2 stability threshold — only a
gradual instability increase.
"""
import numpy as np
import pytest

from chord import ChordConfig, UserKnobs
from chord.loop import Chord
from chord.types import Exposure, ExposureSource, Post, Reaction

import tests.test_properties as P  # reuse the random-world generator
from chord.simulator import Simulator


def _world(seed):
    return P._random_world(seed)


def test_concentration_controller_response_is_applied():
    """§9.3 (FIXED): the controller's response now reaches the estimator.

    Previously the loop called ``controller.step(λ)`` but never read ``controller.state``
    — the controller was inert. It is now wired: ``fit_window`` uses the controller's
    δ/ε_min. Forcing a heavy-teleport δ (the controller's tighten response) must flatten
    the rater weights vs a light-teleport δ, on the *same* data and config.
    """
    from chord.monitor import gini
    reactions, posts, exposures = _world(3)
    cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, eigentrust_delta=0.9)

    tight = Chord(cfg, seed=3)
    tight.controller.state.eigentrust_delta = 0.30       # controller says: teleport hard
    lam_tight = tight.fit_window(reactions, posts, exposures).rater_lambda_eff

    loose = Chord(cfg, seed=3)
    loose.controller.state.eigentrust_delta = 0.95       # controller says: barely teleport
    lam_loose = loose.fit_window(reactions, posts, exposures).rater_lambda_eff

    g_tight, g_loose = gini(lam_tight), gini(lam_loose)
    print(f"\n[dynamics] controller δ applied: Gini(λ) tight-δ={g_tight:.3f} "
          f"loose-δ={g_loose:.3f}")
    assert g_tight < g_loose - 1e-3, (
        f"controller δ not applied: forcing a heavy-teleport δ did not flatten λ "
        f"(Gini tight={g_tight:.3f} vs loose={g_loose:.3f})"
    )


def test_performativity_increases_instability():
    """§9.2: chasing the incentive harder should destabilize the content ecosystem.

    Holds *directionally* — steady-state content-divisiveness variance rises with the
    performativity rate — but note there is no sharp phase transition (it is gradual),
    and feed_churn is ~high at all levels, so this is a weaker signature than the
    two-timescale theory's threshold.
    """
    def instability(perf):
        stds = []
        for s in (1, 2, 3):
            sim = Simulator(config=ChordConfig(d=2, n_clusters=2, mf_iters=25,
                                               budget_B0=2.0, budget_max=6.0),
                            n_users=36, n_slots=6, seed=s, adaptive_authors=False,
                            performativity=perf)
            r = sim.run("chord", n_windows=20)
            stds.append(np.std([m.content_divisiveness for m in r.metrics[10:]]))
        return float(np.mean(stds))

    low, high = instability(0.0), instability(0.8)
    print(f"\n[dynamics] content-divisiveness std: perf=0.0 -> {low:.4f}, perf=0.8 -> {high:.4f}")
    assert high > low + 0.01, (
        f"instability should rise with performativity ({low:.4f} -> {high:.4f})"
    )


def test_long_horizon_metrics_do_not_diverge():
    """Over a long horizon the system should not blow up: welfare/concentration stay
    finite and bounded (a minimal stability sanity check)."""
    sim = Simulator(config=ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0,
                                       budget_max=6.0),
                    n_users=36, n_slots=6, seed=1, adaptive_authors=True,
                    performativity=0.2)
    r = sim.run("chord", n_windows=60)
    assert len(r.metrics) == 60
    for m in r.metrics:
        assert np.isfinite(m.true_value) and np.isfinite(m.gini_lambda)
        assert 0.0 <= m.gini_lambda <= 1.0
        assert -1.0 <= m.true_value <= 1.0
