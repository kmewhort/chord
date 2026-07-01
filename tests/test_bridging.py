import numpy as np
import pytest

from chord.model import BridgingScorer, ClusterModel


def _authors():
    return {"A": "auth1", "B": "auth2", "C": "auth3"}


def test_universal_outbridges_partisan(fitted, cluster_model, toy_config):
    scorer = BridgingScorer(toy_config)
    ec = {"A": {0: 5.0, 1: 5.0}, "B": {0: 5.0, 1: 5.0}, "C": {0: 5.0, 1: 5.0}}
    scores = scorer.score(fitted, cluster_model, _authors(), ec)
    assert scores.b_lcb["A"] > scores.b_lcb["B"]
    assert scores.b_lcb["A"] > scores.b_lcb["C"]


def test_min_over_clusters_is_rawlsian(fitted, cluster_model, toy_config):
    # B_LCB takes the worst cluster: partisan post's disagreeing cluster drags it
    # below its per-cluster mean.
    scorer = BridgingScorer(toy_config)
    ec = {"B": {0: 5.0, 1: 5.0}}
    lcb = scorer.score_post("B", "auth2", fitted, cluster_model, ec["B"])
    per_cluster = scorer.score(fitted, cluster_model, _authors(), ec).per_cluster["B"]
    assert lcb <= per_cluster.min() + 1e-9


def test_untested_cluster_penalized(fitted, cluster_model, toy_config):
    # A post not yet exposed to the disagreeing cluster must NOT be crowned as
    # bridging: n_cp ~ 0 -> large pessimism penalty (§4.2).
    scorer = BridgingScorer(toy_config)
    fully_tested = scorer.score_post("A", "auth1", fitted, cluster_model, {0: 5.0, 1: 5.0})
    half_tested = scorer.score_post("A", "auth1", fitted, cluster_model, {0: 5.0})
    assert half_tested < fully_tested


def test_more_exposure_tightens_bound(fitted, cluster_model, toy_config):
    scorer = BridgingScorer(toy_config)
    low = scorer.score_post("A", "auth1", fitted, cluster_model, {0: 1.0, 1: 1.0})
    high = scorer.score_post("A", "auth1", fitted, cluster_model, {0: 50.0, 1: 50.0})
    # more exposure -> smaller penalty -> higher (less pessimistic) B_LCB
    assert high > low


def test_unseen_post_returns_neg_inf(fitted, cluster_model, toy_config):
    scorer = BridgingScorer(toy_config)
    assert scorer.score_post("nonexistent", "auth9", fitted, cluster_model, {}) == float("-inf")


def test_missing_cluster_counts_treated_as_zero(fitted, cluster_model, toy_config):
    scorer = BridgingScorer(toy_config)
    # No exposure map at all == both clusters untested == max penalty.
    none = scorer.score_post("A", "auth1", fitted, cluster_model, None)
    some = scorer.score_post("A", "auth1", fitted, cluster_model, {0: 5.0, 1: 5.0})
    assert none < some
