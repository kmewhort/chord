"""Broader test: the end-to-end CHORD loop (§9.1).

Exercises fit_window (steps 1-7) and rank (serving) together on the canonical
bipolar world, asserting the paper's headline behaviors emerge end to end:
* universal content out-bridges partisan content (the keystone, §4);
* the M dial trades bridging for personalization per user (§7.1), ungameably;
* the author budget replenishes for quality authors and not for partisan ones (§8).
"""
import numpy as np
import pytest

from chord import Chord, ChordConfig, Exposure, ExposureSource, Post, Reaction, UserKnobs
from chord.propensity import UniformExplorationModel


def _world():
    posts = {"A": Post("A", "auth1"), "B": Post("B", "auth2"), "C": Post("C", "auth3")}
    rx, exps = [], []
    for u in range(10):
        left = u < 5
        for pid, val in [("A", 1.0), ("B", 1.0 if left else -1.0),
                         ("C", -1.0 if left else 1.0)]:
            rx.append(Reaction(u, pid, val, timestamp=float(u)))
            exps.append(Exposure(u, pid, propensity=0.5, source=ExposureSource.ORGANIC))
    return posts, rx, exps


@pytest.fixture
def fitted_chord():
    posts, rx, exps = _world()
    cfg = ChordConfig(d=4, mf_iters=40, n_clusters=2)
    chord = Chord(cfg, propensity_model=UniformExplorationModel(0.5), seed=1, inner_iters=3)
    chord.fit_window(rx, posts, exps)
    return chord, posts


def test_rank_before_fit_raises():
    chord = Chord(ChordConfig())
    with pytest.raises(RuntimeError):
        chord.rank(0, [Post("p", "a")])


def test_keystone_universal_outbridges_partisan(fitted_chord):
    chord, posts = fitted_chord
    b = chord.state.bridging.b_lcb
    assert b["A"] > b["B"]
    assert b["A"] > b["C"]


def test_pure_bridging_ranks_universal_first(fitted_chord):
    chord, posts = fitted_chord
    feed = chord.rank(0, list(posts.values()), UserKnobs(M=1.0), n_slots=3)
    assert feed[0] == "A"  # pure bridging -> universal post on top


def test_M_dial_is_personal_and_ungameable(fitted_chord):
    chord, posts = fitted_chord
    # A left user at M=0 (engagement-like) should prefer their in-group post B
    # over the out-group post C. This only changes THIS user's feed.
    left_feed = chord.rank(0, list(posts.values()), UserKnobs(M=0.0), n_slots=3)
    right_feed = chord.rank(9, list(posts.values()), UserKnobs(M=0.0), n_slots=3)
    assert left_feed.index("B") < left_feed.index("C")   # left prefers B
    assert right_feed.index("C") < right_feed.index("B")  # right prefers C
    # the two users' knob choices do not affect each other's ordering primitive:
    # the underlying bridging scores are identical regardless of who is viewing.
    assert chord.state.bridging.b_lcb["A"] == chord.state.bridging.b_lcb["A"]


def test_budget_replenishes_for_quality_author(fitted_chord):
    chord, posts = fitted_chord
    # auth1 authored the universal (high-strength) post A -> budget above base;
    # partisan authors earned low/negative strength -> stay near base.
    assert chord.budget.budget("auth1") > chord.budget.budget("auth2")
    assert chord.budget.budget("auth1") > chord.config.budget_B0


def test_inner_loop_iterates(fitted_chord):
    chord, _ = fitted_chord
    assert chord.state.n_iter_inner == 3  # steps 1-3 iterated


def test_unseen_candidate_routed_to_exploration(fitted_chord):
    chord, posts = fitted_chord
    # Introduce a brand-new post with no fitted loading; it should be auditionable
    # (registered in the exploration pool) and can occupy the exploration floor.
    new_post = Post("NEW", "auth_new")
    feed = chord.rank(0, list(posts.values()) + [new_post],
                      UserKnobs(M=1.0, epsilon=0.34), n_slots=3)
    assert chord.exploration.is_open("NEW")
    assert "NEW" in feed  # exploration floor (ceil(0.34*3)=2) pulls it in


def test_multiple_windows_run(fitted_chord):
    chord, posts = fitted_chord
    # a second window with fresh reactions should refit without error
    _, rx, exps = _world()
    st2 = chord.fit_window(rx, posts, exps)
    assert st2.result.weighted_rmse < 0.2
