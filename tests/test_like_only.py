"""Bridging on a one-signed (like-only) network — no dislikes (§4.2).

A network like Bluesky has likes/reposts but no "dislike": the cluster that doesn't like
a post produces *no signal* (it is absent, n_cp=0), not a negative. In that regime the
grand mean μ is high (everything observed is positive), so the default EB-shrinkage form
pulls a silent cluster UP toward μ — and a partisan post (loved in one cluster, silent in
the other) looks just as bridged as a universal one: B_LCB goes flat. The legacy
subtractive lower-confidence bound (``bridging_subtractive_lcb``) instead penalizes the
silent cluster DOWN, recovering cross-cluster contrast from positive-only data.
"""
import dataclasses

import pytest

from chord import ChordConfig, Post, Reaction
from chord.model import BridgingScorer, ClusterModel, MatrixFactorization

AUTHORS = {"U": "a_u", "P0": "a_0", "P1": "a_1"}


def _like_only_world(seed=1):
    """Fit a positive-only world: everyone likes U; each cluster likes only its own post.
    All reactions are +0.5 (FAVORITE), so μ is high — the like-only regime."""
    cfg = ChordConfig(d=4, mf_iters=50, n_clusters=2)
    rx = []
    for u in range(10):
        left = u < 5
        rx.append(Reaction(u, "U", 0.5, timestamp=float(u)))
        rx.append(Reaction(u, "P0" if left else "P1", 0.5, timestamp=float(u)))
    posts = {p: Post(p, AUTHORS[p]) for p in AUTHORS}
    fitted = MatrixFactorization(cfg, seed=seed).fit(rx, posts)
    clusters = ClusterModel.from_factorization(fitted, {u: (0 if u < 5 else 1) for u in range(10)})
    # like-only reception: the non-liking cluster is ABSENT, not negative.
    reception = {
        "U":  {0: (5.0, 0.5), 1: (5.0, 0.5)},   # tested-and-liked across both clusters
        "P0": {0: (5.0, 0.5)},                   # liked in cluster 0; cluster 1 silent
        "P1": {1: (5.0, 0.5)},                   # liked in cluster 1; cluster 0 silent
    }
    return fitted, clusters, cfg, reception


def test_default_shrinkage_is_flat_on_like_only_data():
    fitted, clusters, cfg, reception = _like_only_world()
    assert fitted.mu > 0.25, "μ should be high in a positive-only world"
    s = BridgingScorer(cfg).score(fitted, clusters, AUTHORS, reception)   # subtractive OFF
    # the silent cluster shrinks up to μ, so the partisan posts ≈ the universal one
    assert abs(s.b_lcb["U"] - s.b_lcb["P0"]) < 0.06
    assert abs(s.b_lcb["U"] - s.b_lcb["P1"]) < 0.06


def test_subtractive_lcb_recovers_bridging_from_like_only_data():
    fitted, clusters, cfg, reception = _like_only_world()
    cfg = dataclasses.replace(cfg, bridging_subtractive_lcb=True, bridging_aggregator="min")
    s = BridgingScorer(cfg).score(fitted, clusters, AUTHORS, reception)
    # the silent cluster is penalized down, so the cross-cluster post wins clearly
    assert s.b_lcb["U"] > s.b_lcb["P0"] + 0.3
    assert s.b_lcb["U"] > s.b_lcb["P1"] + 0.3
    # and a post tested in *more* clusters scores higher — the bridging ordering is back
    assert s.b_lcb["P0"] == pytest.approx(s.b_lcb["P1"], abs=1e-9)


def test_subtractive_lcb_still_ranks_universal_over_partisan_with_negatives():
    # the option must not break the negative-rich case it was (by default) retired for:
    # a genuinely disliked cluster still demotes a partisan post below a universal one.
    fitted, clusters, cfg, _ = _like_only_world()
    cfg = dataclasses.replace(cfg, bridging_subtractive_lcb=True, bridging_aggregator="min")
    reception = {
        "U":  {0: (5.0, 0.8), 1: (5.0, 0.8)},
        "P0": {0: (5.0, 0.9), 1: (5.0, -0.6)},   # cluster 1 actively dislikes it
        "P1": {0: (5.0, -0.6), 1: (5.0, 0.9)},
    }
    s = BridgingScorer(cfg).score(fitted, clusters, AUTHORS, reception)
    assert s.b_lcb["U"] > s.b_lcb["P0"] and s.b_lcb["U"] > s.b_lcb["P1"]


def test_subtractive_off_by_default_leaves_the_shrinkage_form_unchanged():
    fitted, clusters, cfg, reception = _like_only_world()
    assert cfg.bridging_subtractive_lcb is False
    base = BridgingScorer(cfg).score_post("U", "a_u", fitted, clusters, reception["U"])
    w = 5.0 / (5.0 + cfg.bridging_shrinkage_n0)
    shrunk = fitted.mu + w * (0.5 - fitted.mu)                # EB form, both clusters equal
    expected = shrunk if cfg.bridging_aggregator == "min" else shrunk   # both clusters equal → =shrunk
    assert base == pytest.approx(expected, abs=1e-9)
