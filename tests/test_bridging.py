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


def _hostile_friendly(scorer, fitted, cluster_model, pid, author):
    """Return (hostile_cluster, friendly_cluster) for a post by its per-cluster
    reconstructed reception (the disagreeing cluster has the lowest r_hat)."""
    per = scorer.score(fitted, cluster_model, _authors(), {pid: {0: 5.0, 1: 5.0}}).per_cluster[pid]
    return int(np.argmin(per)), int(np.argmax(per))


def test_rawlsian_min_within_shrinkage_bounds(fitted, cluster_model, toy_config):
    # With the "min" aggregator and equal exposure the score still takes the worst
    # cluster, but the empirical-Bayes shrinkage pulls it toward the population mean:
    # it lies between the raw worst-cluster reception and the mean (never below min).
    import dataclasses
    cfg = dataclasses.replace(toy_config, bridging_aggregator="min")
    scorer = BridgingScorer(cfg)
    ec = {"B": {0: 20.0, 1: 20.0}}
    lcb = scorer.score_post("B", "auth2", fitted, cluster_model, ec["B"])
    per = scorer.score(fitted, cluster_model, _authors(), ec).per_cluster["B"]
    assert per.min() - 1e-9 <= lcb <= per.mean() + 1e-9


def test_untested_hostile_cluster_assumed_average(fitted, cluster_model, toy_config):
    # New semantics (§4.2, App C.5): an *unexposed* disagreeing cluster is assumed
    # average (shrunk to the mean), NOT penalized. Exposing that hostile cluster is
    # what reveals the dissent and lowers a partisan post's bridged support.
    scorer = BridgingScorer(toy_config)
    hostile, friendly = _hostile_friendly(scorer, fitted, cluster_model, "B", "auth2")
    only_friendly = scorer.score_post("B", "auth2", fitted, cluster_model, {friendly: 50.0})
    both_exposed = scorer.score_post("B", "auth2", fitted, cluster_model,
                                     {friendly: 50.0, hostile: 50.0})
    assert both_exposed < only_friendly


def test_more_exposure_reveals_dissent(fitted, cluster_model, toy_config):
    # For a partisan post, MORE exposure of the disagreeing cluster drives the score
    # DOWN toward its true (low) reception — surviving contact, not tightening a
    # penalty. (The reverse of the old subtractive-bound behavior.)
    scorer = BridgingScorer(toy_config)
    hostile, friendly = _hostile_friendly(scorer, fitted, cluster_model, "B", "auth2")
    low_exposure = scorer.score_post("B", "auth2", fitted, cluster_model,
                                     {friendly: 50.0, hostile: 1.0})
    high_exposure = scorer.score_post("B", "auth2", fitted, cluster_model,
                                      {friendly: 50.0, hostile: 50.0})
    assert high_exposure < low_exposure


def test_unseen_post_returns_neg_inf(fitted, cluster_model, toy_config):
    scorer = BridgingScorer(toy_config)
    assert scorer.score_post("nonexistent", "auth9", fitted, cluster_model, {}) == float("-inf")


def test_unexposed_post_scores_near_population_mean(fitted, cluster_model, toy_config):
    # No exposure map => every cluster regresses to the mean, so B_LCB is the
    # population mean reception (benefit of the doubt), and exposing the hostile
    # cluster of a partisan post pulls it below that.
    scorer = BridgingScorer(toy_config)
    per = scorer.score(fitted, cluster_model, _authors(), {"B": {0: 5.0, 1: 5.0}}).per_cluster["B"]
    none = scorer.score_post("B", "auth2", fitted, cluster_model, None)
    exposed = scorer.score_post("B", "auth2", fitted, cluster_model, {0: 50.0, 1: 50.0})
    assert abs(none - per.mean()) < 1e-9
    assert exposed < none
