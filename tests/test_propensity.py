import numpy as np
import pytest

from chord import ChordConfig, Exposure, Post, Reaction
from chord.propensity import (
    LoggedPropensityModel,
    PolicyDerivedModel,
    PositionBasedModel,
    UniformExplorationModel,
    compute_ipw_weights,
    doubly_robust_mean,
    doubly_robust_reception,
    snipw_estimate,
)


def test_uniform_exploration_returns_epsilon():
    m = UniformExplorationModel(0.05)
    assert m.propensity("u", "p") == 0.05
    with pytest.raises(ValueError):
        UniformExplorationModel(0.0)


def test_position_model_decays_with_rank():
    m = PositionBasedModel(epsilon=0.01, base=0.5)
    p0 = m.propensity("u", "p", Exposure("u", "p", slot=0))
    p3 = m.propensity("u", "p", Exposure("u", "p", slot=3))
    assert p0 > p3 >= 0.01


def test_position_model_floors_at_epsilon():
    m = PositionBasedModel(epsilon=0.2, base=0.5)
    deep = m.propensity("u", "p", Exposure("u", "p", slot=20))
    assert deep == 0.2  # floored


def test_policy_derived_softmax():
    scores = {"p1": 2.0, "p2": 0.0, "p3": 0.0}
    m = PolicyDerivedModel(epsilon=0.01, scores_for=lambda u: scores)
    p1 = m.propensity("u", "p1")
    p2 = m.propensity("u", "p2")
    assert p1 > p2  # higher logging score -> higher exposure probability
    assert abs((p1 + 2 * p2) - 1.0) < 1e-6  # softmax sums to 1


def test_logged_propensity_reads_event():
    m = LoggedPropensityModel(0.05)
    assert m.propensity("u", "p", Exposure("u", "p", propensity=0.3)) == 0.3
    assert m.propensity("u", "p", None) == 0.05  # fallback to floor


def test_ipw_weight_clip():
    # Tiny propensity would blow up 1/pi; the clip caps it at W_max.
    posts = {"p": Post("p", "a")}
    rx = [Reaction("u", "p", 1.0)]
    cfg = ChordConfig(W_max=10.0)
    m = UniformExplorationModel(1e-6)  # 1/pi = 1e6, clipped to 10
    w = compute_ipw_weights(rx, m, cfg, rater_lambda={"u": 1.0})
    assert w[0] <= 10.0 + 1e-9


def test_ipw_weight_uses_rater_lambda():
    posts = {"p": Post("p", "a")}
    rx = [Reaction("hi", "p", 1.0), Reaction("lo", "p", 1.0)]
    cfg = ChordConfig(W_max=10.0)
    m = UniformExplorationModel(0.5)
    w = compute_ipw_weights(rx, m, cfg, rater_lambda={"hi": 0.9, "lo": 0.1})
    assert w[0] > w[1]  # higher-lambda rater gets more weight


def test_snipw_matches_weighted_mean():
    vals = [1.0, 0.0, -1.0]
    weights = [3.0, 1.0, 1.0]
    est = snipw_estimate(vals, weights)
    assert abs(est - (3 * 1 + 0 - 1) / 5) < 1e-12


def test_doubly_robust_consistent_when_propensity_right():
    # If everyone is exposed and propensity is exact, DR reduces to the IPW mean
    # of the observed reactions.
    users = ["u1", "u2", "u3"]
    observed = {"u1": 1.0, "u2": 0.0, "u3": -1.0}
    impute = lambda u, p: 0.5  # deliberately wrong imputation
    cfg = ChordConfig(W_max=100.0)
    m = UniformExplorationModel(1.0)  # pi = 1 everywhere (all exposed)
    dr = doubly_robust_reception(users, "p", observed, impute, m, cfg)
    assert abs(dr - np.mean([1.0, 0.0, -1.0])) < 1e-9


def test_doubly_robust_consistent_when_imputation_right():
    # If imputation is exact, DR is unbiased even when no one is observed.
    users = ["u1", "u2"]
    truth = {"u1": 0.7, "u2": -0.3}
    impute = lambda u, p: truth[u]  # perfect imputation
    cfg = ChordConfig()
    m = UniformExplorationModel(0.5)
    dr = doubly_robust_reception(users, "p", observed={}, impute=impute,
                                 propensity_model=m, config=cfg)
    assert abs(dr - np.mean([0.7, -0.3])) < 1e-9


def test_doubly_robust_mean_falls_back_to_imputation_for_unexposed():
    pairs = [("u1", "p"), ("u2", "p")]
    observed = {("u1", "p"): 1.0}  # u2 never exposed
    impute = lambda u, p: 0.2
    cfg = ChordConfig(W_max=100.0)
    m = UniformExplorationModel(1.0)
    dr = doubly_robust_mean(pairs, observed, impute, m, cfg)
    # u1 term = 0.2 + 1*(1.0-0.2) = 1.0 ; u2 term = 0.2 (imputed) -> mean 0.6
    assert abs(dr - 0.6) < 1e-9
