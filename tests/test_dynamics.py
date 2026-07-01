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
    """§9.3: the controller's response to concentration must reach the estimator.

    FOUND GAP (left failing): the loop calls ``controller.step(lambda)`` but never
    reads ``controller.state`` — eigentrust and ``rank`` use the *fixed* config delta/
    epsilon. So the controller is inert: forcing its recommended teleport-delta to a
    very different value has NO effect on the next window's rater weights. We assert
    the *intended* behaviour (the loop applies the controller's delta); it fails.
    """
    reactions, posts, exposures = _world(3)
    cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, eigentrust_delta=0.85)
    chord = Chord(cfg, seed=3)
    chord.fit_window(reactions, posts, exposures)

    # the controller now recommends a much tighter teleport (raise the floor)
    chord.controller.state.eigentrust_delta = 0.30
    lam_after = chord.fit_window(reactions, posts, exposures).rater_lambda_eff

    # what the loop SHOULD produce if it applied the controller: an independent fit
    # whose config delta *is* 0.30
    ref = Chord(ChordConfig(d=2, n_clusters=2, mf_iters=25, eigentrust_delta=0.30), seed=3)
    ref.fit_window(reactions, posts, exposures)
    lam_ref = ref.fit_window(reactions, posts, exposures).rater_lambda_eff

    ids = list(lam_after)
    diff = max(abs(lam_after[u] - lam_ref[u]) for u in ids)
    assert diff < 1e-6, (
        f"controller response not applied: forcing delta=0.30 left rater weights "
        f"unchanged (max diff vs a real delta=0.30 fit = {diff:.4f}). The §9.3 "
        f"controller is inert — the loop discards controller.state."
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
