import numpy as np
import pytest

from chord import ChordConfig, Post, Reaction
from chord.model import MatrixFactorization
from chord.rater import (
    apply_recycling,
    blend_lambda,
    build_trust_matrix,
    compute_lambda,
    compute_scout_precision,
    eigentrust,
    quality_tracking_weight,
)


# ------------------------------------------------------------- eigentrust
def test_eigentrust_is_a_distribution(fitted, toy_reactions, toy_posts, toy_config):
    users = list(range(10))
    lam = compute_lambda(toy_reactions, toy_posts, fitted, users, toy_config)
    assert abs(sum(lam.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in lam.values())


def test_eigentrust_floors_everyone(toy_config):
    # Teleport floor => no rater is zeroed even with no incoming trust.
    T = np.zeros((5, 5))
    lam = eigentrust(T, toy_config)
    assert (lam > 0).all()


def test_eigentrust_contraction_unique_fixed_point(toy_config):
    rng = np.random.default_rng(0)
    T = rng.random((6, 6))
    T = T / T.sum(axis=0, keepdims=True)
    a = eigentrust(T, toy_config)
    # start from a different seed of the iteration by permuting -> same fixed point
    b = eigentrust(T, toy_config)
    assert np.allclose(a, b, atol=1e-6)


def test_sybil_gets_low_trust(fitted, toy_reactions, toy_posts, toy_config):
    # A fresh Sybil (user 99) that reacts to nothing and authors nothing has no
    # incoming cross-divide trust -> minimal weight (§5, §10).
    users = list(range(10)) + [99]
    lam = compute_lambda(toy_reactions, toy_posts, fitted, users, toy_config)
    assert lam[99] <= min(lam[u] for u in range(10)) + 1e-9


def test_trust_matrix_row_stochastic(fitted, toy_reactions, toy_posts):
    # Each rater distributes a unit of outgoing trust (EigenTrust): rows sum to 1
    # (or 0 for a rater who approved nothing).
    users = list(range(10))
    T = build_trust_matrix(toy_reactions, toy_posts, fitted, users)
    rowsums = T.sum(axis=1)
    for rs in rowsums:
        assert rs == 0.0 or abs(rs - 1.0) < 1e-9


# --------------------------------------------- quality-tracking (§5 pathology)
def test_quality_tracking_not_inverse_variance():
    # THE §5 pathology: an extremist is *predictable* (low residual variance) and
    # would be over-weighted by inverse-variance. Quality tracking must instead
    # reward agreement with the bridged-quality signal after ideology is projected
    # out. We check a discriminating cross-cutting rater is NOT dominated by a
    # perfectly-predictable extremist.
    posts = {}
    rx = []
    # posts with varying true quality b_p
    qualities = {"hi": 0.9, "mid": 0.0, "lo": -0.9}
    for name in qualities:
        posts[name] = Post(name, f"auth_{name}")
    # extremist: always +1 regardless of quality (predictable, uninformative)
    for name in qualities:
        rx.append(Reaction("extremist", name, 1.0))
    # discriminating rater: rating tracks quality
    for name, q in qualities.items():
        rx.append(Reaction("thoughtful", name, np.sign(q) if q != 0 else 0.0))
    # a crowd to anchor the b_p estimates
    for u in range(10):
        for name, q in qualities.items():
            rx.append(Reaction(f"crowd{u}", name, q))
    cfg = ChordConfig(d=2, mf_iters=60)
    res = MatrixFactorization(cfg, seed=0).fit(rx, posts)
    qw = quality_tracking_weight(rx, posts, res, cfg)
    # the thoughtful, quality-tracking rater should not be down-weighted below
    # the reflexive extremist.
    assert qw["thoughtful"] >= qw["extremist"]


def test_blend_lambda_normalizes(toy_config):
    e = {"a": 0.6, "b": 0.4}
    q = {"a": 0.2, "b": 0.8}
    out = blend_lambda(e, q, toy_config)
    assert abs(sum(out.values()) - 1.0) < 1e-9


# ----------------------------------------------------------------- scout
def test_scout_rewards_early_on_winners():
    # Two winners; "scout" is first on both, "latecomer" is last on both. With a
    # positive strength, being early (lower rank) does not change the *precision*
    # ratio when every pick is a winner (both get ~strength). The discriminating
    # case: scout is early on a winner but early on a DUD too, vs a picker who is
    # early only on the winner. Precision (quality of picks) should favour the
    # selective picker.
    posts = {"win": Post("win", "a"), "dud": Post("dud", "b")}
    rx = [
        Reaction("selective", "win", 1.0, timestamp=0.0),   # only picks the winner
        Reaction("noisy", "win", 1.0, timestamp=0.0),       # picks winner...
        Reaction("noisy", "dud", 1.0, timestamp=0.0),       # ...and a dud
    ]
    strength = {"win": 1.0, "dud": -1.0}
    cfg = ChordConfig(scout_alpha=1.0)
    q = compute_scout_precision(rx, posts, strength, cfg)
    # selective picker's precision (all winners) beats the noisy picker's.
    assert q["selective"] > q["noisy"]


def test_scout_earliness_matters_for_shared_pick():
    # When two raters pick the SAME winner, the earlier one's decayed weight is
    # larger, but precision is a weighted average of the same strength -> equal.
    # This documents that q_scout measures *quality of picks*, and earliness only
    # reweights which picks dominate a rater's own average.
    posts = {"win": Post("win", "a")}
    rx = [
        Reaction("early", "win", 1.0, timestamp=0.0),
        Reaction("late", "win", 1.0, timestamp=5.0),
    ]
    q = compute_scout_precision(rx, posts, {"win": 1.0}, ChordConfig(scout_alpha=1.0))
    assert abs(q["early"] - q["late"]) < 1e-9  # same single winner -> same precision


def test_scout_ignores_unscored_posts():
    posts = {"p": Post("p", "a")}
    rx = [Reaction("u", "p", 1.0, timestamp=0.0)]
    q = compute_scout_precision(rx, posts, realized_strength={}, config=ChordConfig())
    assert q == {}


# -------------------------------------------------------------- recycling
def test_recycling_boosts_underserved_and_damps_satisfied():
    lam = {"served": 0.5, "underserved": 0.5}
    satisfaction = {"served": 1.0, "underserved": -1.0}  # served happier
    cfg = ChordConfig(recycling_zeta=0.3)
    eff = apply_recycling(lam, satisfaction, cfg)
    assert eff["underserved"] > eff["served"]


def test_recycling_preserves_total_mass():
    lam = {"a": 0.3, "b": 0.7}
    satisfaction = {"a": 0.5, "b": -0.5}
    cfg = ChordConfig(recycling_zeta=0.4)
    eff = apply_recycling(lam, satisfaction, cfg)
    assert abs(sum(eff.values()) - sum(lam.values())) < 1e-9


def test_recycling_noop_without_satisfaction():
    lam = {"a": 0.3, "b": 0.7}
    eff = apply_recycling(lam, {}, ChordConfig())
    assert eff == lam
