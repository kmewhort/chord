import numpy as np
import pytest

from chord import ChordConfig
from chord.monitor import (
    ConcentrationController,
    effective_rater_count,
    endogenous_shift,
    gini,
)


def test_effective_rater_count_uniform_equals_n():
    lam = {i: 1.0 for i in range(10)}
    assert abs(effective_rater_count(lam) - 10.0) < 1e-9


def test_effective_rater_count_collapses_under_concentration():
    lam = {0: 100.0, 1: 1.0, 2: 1.0}
    assert effective_rater_count(lam) < 2.0  # one rater dominates


def test_gini_zero_for_equal():
    lam = {i: 1.0 for i in range(5)}
    assert abs(gini(lam)) < 1e-9


def test_gini_high_for_concentrated():
    lam = {0: 100.0, 1: 0.0, 2: 0.0, 3: 0.0}
    assert gini(lam) > 0.6


def test_controller_tightens_when_concentrated():
    cfg = ChordConfig(gini_ceiling=0.3, eigentrust_delta=0.9, epsilon_min=0.05)
    ctrl = ConcentrationController(cfg)
    concentrated = {0: 100.0, 1: 1.0, 2: 1.0}
    st = ctrl.step(concentrated)
    # tightening lowers delta (more teleport) and raises epsilon_min
    assert st.eigentrust_delta < 0.9
    assert st.epsilon_min > 0.05


def test_controller_relaxes_toward_defaults_when_healthy():
    cfg = ChordConfig(gini_ceiling=0.6, eigentrust_delta=0.9, epsilon_min=0.05)
    ctrl = ConcentrationController(cfg)
    # first tighten
    ctrl.step({0: 100.0, 1: 1.0, 2: 1.0})
    tightened = ctrl.state.eigentrust_delta
    # then healthy -> relax back up toward the configured default
    ctrl.step({i: 1.0 for i in range(10)})
    assert ctrl.state.eigentrust_delta >= tightened


def test_controller_never_exceeds_defaults():
    cfg = ChordConfig(gini_ceiling=0.9, eigentrust_delta=0.85, epsilon_min=0.05)
    ctrl = ConcentrationController(cfg)
    for _ in range(20):
        ctrl.step({i: 1.0 for i in range(10)})
    assert ctrl.state.eigentrust_delta <= 0.85 + 1e-9
    assert ctrl.state.epsilon_min >= 0.05 - 1e-9


def test_endogenous_shift_subtracts_exogenous_baseline():
    # personalized drift 0.5, exploration (exogenous) drift 0.2 -> endogenous 0.3
    assert abs(endogenous_shift(0.2, 0.5) - 0.3) < 1e-9
    # when the loop drift is below the news baseline, endogenous is floored at 0
    assert endogenous_shift(0.5, 0.3) == 0.0
