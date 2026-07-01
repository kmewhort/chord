"""Broader test: the closed-loop simulator (§9, Appendix C.4).

The feedback loop is untestable on fixed data, so these assert the dynamic
properties the simulator exists to exercise:
* the concentration controller holds rater weight bounded (Gini below ceiling,
  N_eff from collapsing);
* the exploration anchor is sustained at the floor over every window (persistent
  excitation, §9.3);
* the conserved author budget dilutes a firehose author's reach-per-post (§8).
"""
import numpy as np
import pytest

from chord import ChordConfig, UserKnobs
from chord.simulator import Simulator


@pytest.fixture(scope="module")
def sim_result():
    sim = Simulator(n_users=30, d=2, knobs=UserKnobs(M=1.0), n_slots=6, seed=3)
    return sim.run(n_windows=8)


def test_closed_loop_runs_all_windows(sim_result):
    assert len(sim_result.metrics) == 8
    assert all(m.n_reactions > 0 for m in sim_result.metrics)


def test_concentration_stays_bounded(sim_result):
    # The §9.3 controller target: rater concentration held in a bounded regime.
    assert max(m.gini_lambda for m in sim_result.metrics) < 0.5


def test_effective_rater_count_does_not_collapse(sim_result):
    # N_eff should remain a healthy fraction of the population, not collapse to 1.
    assert min(m.n_eff for m in sim_result.metrics[1:]) > 5.0


def test_exploration_anchor_sustained(sim_result):
    # Persistent excitation: every steady-state window keeps sampling at the floor.
    for m in sim_result.metrics[1:]:
        assert m.exploration_rate > 0.0


def test_firehose_reach_is_diluted(sim_result):
    # §8: a high-volume firehose author earns LESS reach per post than a quality
    # (universal) author, because the conserved budget spreads thin.
    fh = np.mean([m.firehose_reach_per_post for m in sim_result.metrics[2:]])
    uni = np.mean([m.universal_reach_per_post for m in sim_result.metrics[2:]])
    assert fh < uni


def test_bridging_scores_are_finite(sim_result):
    for m in sim_result.metrics:
        assert np.isfinite(m.mean_bridge_score)


def test_engagement_mode_differs_from_bridging_mode():
    # Two runs with different master dials should produce different feed dynamics;
    # this is a smoke test that the M knob is actually wired through the loop.
    sim_bridge = Simulator(n_users=24, knobs=UserKnobs(M=1.0), n_slots=6, seed=5)
    sim_engage = Simulator(n_users=24, knobs=UserKnobs(M=0.0), n_slots=6, seed=5)
    rb = sim_bridge.run(n_windows=5)
    re = sim_engage.run(n_windows=5)
    # the mean bridge score of what gets shown should differ between regimes
    mb = np.mean([m.mean_bridge_score for m in rb.metrics])
    me = np.mean([m.mean_bridge_score for m in re.metrics])
    assert mb != me
