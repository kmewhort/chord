"""E2: the ε-slice bias calibrator (§6/§13.2).

Unit tests of the mechanism (deterministic). The closed-loop *benefit* is not asserted
here: in the simulator, with only the sparse ε floor, the effect is within seed noise —
E2's clear win is on dense random-exposure data (Coat, where it beat IPW, see
EXPERIMENTS.md), and in the loop it needs a dedicated ε allocation (the randomization-
portfolio meta-point). So we test that the calibrator recovers a known bias and abstains
without evidence, and that calibrated_reception prefers unconfounded ground truth.
"""
import numpy as np

from chord.model import BiasCalibrator, calibrated_reception


def test_calibrator_recovers_linear_bias_and_abstains():
    cal = BiasCalibrator(min_evidence=4.0)
    rng = np.random.default_rng(0)
    # true unconfounded relationship r_exp = 0.2 + 0.5 r_org
    cal.update([(0, x, 0.2 + 0.5 * x, 1.0) for x in rng.uniform(-1, 1, 60)])
    for x in (-0.6, 0.0, 0.8):
        assert abs(cal.predict(0, x) - (0.2 + 0.5 * x)) < 0.05
    # a cluster with no paired evidence returns the organic value unchanged (no guessing)
    assert cal.predict(7, 0.7) == 0.7


def test_calibrated_reception_prefers_exploration_ground_truth():
    cal = BiasCalibrator(min_evidence=4.0)
    reception = {"p": {0: (10.0, 0.9)}}          # confounded organic mean 0.9
    org = {"p": {0: [10.0, 9.0]}}                # organic: mean 0.9
    exp = {"p": {0: [4.0, -0.8]}}                # exploration: mean -0.2 (unconfounded)
    out, pairs = calibrated_reception(reception, org, exp, cal)
    assert out["p"][0][1] == -0.2                # uses the unconfounded exploration mean
    assert pairs == [(0, 0.9, -0.2, 4.0)]        # and emits the training pair
