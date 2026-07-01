"""B_LCB from empirical per-cluster reception (§4.2).

Reception is now supplied as ``{cluster: (n_cp, r_emp_cp)}`` — the propensity-corrected
evidence weight and the empirical mean of that cluster's signed reactions — so these
test the empirical-Bayes shrinkage and the min/nash aggregation directly.
"""
import dataclasses

import pytest

from chord.model import BridgingScorer


def _authors():
    return {"A": "auth1", "B": "auth2", "C": "auth3"}


def test_universal_outbridges_partisan(fitted, cluster_model, toy_config):
    scorer = BridgingScorer(toy_config)
    reception = {
        "A": {0: (40.0, 0.8), 1: (40.0, 0.8)},     # both clusters approve → bridging
        "B": {0: (40.0, 0.9), 1: (40.0, -0.6)},    # one approves, one dislikes → partisan
        "C": {0: (40.0, -0.6), 1: (40.0, 0.9)},
    }
    s = scorer.score(fitted, cluster_model, _authors(), reception)
    assert s.b_lcb["A"] > s.b_lcb["B"]
    assert s.b_lcb["A"] > s.b_lcb["C"]


def test_min_takes_worst_cluster_shrunk(fitted, cluster_model, toy_config):
    # min aggregator: B_LCB is the worst cluster's reception, empirical-Bayes-shrunk
    # toward the prior grand = μ by weight n_cp/(n_cp+n0).
    cfg = dataclasses.replace(toy_config, bridging_aggregator="min")
    scorer = BridgingScorer(cfg)
    lcb = scorer.score_post("B", "auth2", fitted, cluster_model,
                            {0: (40.0, 0.9), 1: (40.0, -0.6)})
    w = 40.0 / (40.0 + cfg.bridging_shrinkage_n0)
    worst = fitted.mu + w * (-0.6 - fitted.mu)
    assert lcb == pytest.approx(worst, abs=1e-9)
    assert lcb < fitted.mu


def test_untested_hostile_cluster_assumed_average(fitted, cluster_model, toy_config):
    # An *unrated* disagreeing cluster is absent (n_cp = 0) → shrunk to the prior, not
    # penalized. Rating that hostile cluster is what reveals the dissent.
    scorer = BridgingScorer(toy_config)
    only_friendly = scorer.score_post("B", "auth2", fitted, cluster_model, {0: (50.0, 0.9)})
    both = scorer.score_post("B", "auth2", fitted, cluster_model,
                             {0: (50.0, 0.9), 1: (50.0, -0.6)})
    assert both < only_friendly


def test_more_evidence_reveals_dissent(fitted, cluster_model, toy_config):
    # More evidence (n_cp) from the disagreeing cluster drives the score toward its true
    # low reception — surviving contact, not tightening a penalty.
    scorer = BridgingScorer(toy_config)
    low = scorer.score_post("B", "auth2", fitted, cluster_model,
                            {0: (50.0, 0.9), 1: (1.0, -0.6)})
    high = scorer.score_post("B", "auth2", fitted, cluster_model,
                             {0: (50.0, 0.9), 1: (50.0, -0.6)})
    assert high < low


def test_unseen_post_returns_neg_inf(fitted, cluster_model, toy_config):
    scorer = BridgingScorer(toy_config)
    assert scorer.score_post("nonexistent", "auth9", fitted, cluster_model, {}) == float("-inf")


def test_no_reception_scores_at_prior(fitted, cluster_model, toy_config):
    # No reception at all → every cluster regresses to the prior grand = μ (benefit of
    # the doubt), and rating the hostile cluster of a partisan post pulls it below.
    scorer = BridgingScorer(toy_config)
    none = scorer.score_post("B", "auth2", fitted, cluster_model, None)
    exposed = scorer.score_post("B", "auth2", fitted, cluster_model,
                                {0: (50.0, 0.9), 1: (50.0, -0.6)})
    assert none == pytest.approx(fitted.mu, abs=1e-9)
    assert exposed < none
