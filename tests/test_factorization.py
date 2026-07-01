import numpy as np
import pytest

from chord import ChordConfig, Post, Reaction
from chord.model import MatrixFactorization


def test_empty_reactions_raises(toy_config):
    with pytest.raises(ValueError):
        MatrixFactorization(toy_config).fit([], {})


def test_unknown_post_raises(toy_config):
    with pytest.raises(KeyError):
        MatrixFactorization(toy_config).fit([Reaction(0, "X", 1.0)], {})


def test_weights_must_align(toy_reactions, toy_posts, toy_config):
    with pytest.raises(ValueError):
        MatrixFactorization(toy_config).fit(toy_reactions, toy_posts, weights=[1.0])


def test_negative_weights_rejected(toy_reactions, toy_posts, toy_config):
    w = [-1.0] * len(toy_reactions)
    with pytest.raises(ValueError):
        MatrixFactorization(toy_config).fit(toy_reactions, toy_posts, weights=w)


def test_fit_reduces_error(fitted):
    # A structured toy world should be fit to low weighted RMSE.
    assert fitted.weighted_rmse < 0.1


def test_reconstruction_matches_predict(fitted, toy_posts):
    # predict() should reproduce the reconstruction used internally.
    pred = fitted.predict(0, "A", "auth1")
    manual = (
        fitted.mu
        + fitted.b_user[0]
        + fitted.b_author["auth1"]
        + fitted.b_post["A"]
        + float(np.dot(fitted.x_user[0], fitted.y_post["A"]))
    )
    assert abs(pred - manual) < 1e-9


def test_universal_post_has_higher_intercept(fitted):
    # b_p is the marginal-breadth proxy; the universal post A should exceed the
    # partisan posts B and C on the scalar intercept.
    assert fitted.b_post["A"] > fitted.b_post["B"]
    assert fitted.b_post["A"] > fitted.b_post["C"]


def test_author_term_absorbs_blanket_elevation():
    # A "star" author whose posts are uniformly loved, alongside an ordinary
    # author whose posts draw mixed reception. The star's blanket elevation
    # should land in the *author* term, not leak into per-post intercepts (§4.1).
    posts = {}
    rx = []
    for i in range(6):
        posts[f"s{i}"] = Post(f"s{i}", "star")
        posts[f"n{i}"] = Post(f"n{i}", "normal")
    for u in range(8):
        for i in range(6):
            rx.append(Reaction(u, f"s{i}", 1.0))               # star: all love
            rx.append(Reaction(u, f"n{i}", 1.0 if u % 2 else -1.0))  # normal: split
    cfg = ChordConfig(d=3, mf_iters=80, reg_bias_author=0.01, reg_bias_post=0.5)
    res = MatrixFactorization(cfg, seed=0).fit(rx, posts)
    # star's author baseline clearly exceeds the ordinary author's
    assert res.b_author["star"] > res.b_author["normal"] + 0.5
    # and the star's blanket elevation does not masquerade as per-post breadth:
    star_post_spread = max(abs(res.b_post[f"s{i}"]) for i in range(6))
    assert star_post_spread < res.b_author["star"]


def test_weighting_downweights_observations():
    # Zero-weighting a rater's contradictory reactions should let the majority win.
    posts = {"p": Post("p", "a")}
    rx = [Reaction(u, "p", 1.0) for u in range(9)] + [Reaction(9, "p", -1.0)]
    cfg = ChordConfig(d=2, mf_iters=40)
    weights = [1.0] * 9 + [0.0]  # ignore the dissenter entirely
    res = MatrixFactorization(cfg, seed=0).fit(rx, posts, weights)
    pred = res.predict(0, "p", "a")
    assert pred > 0.7  # pulled toward the majority +1, not diluted by the -1


def test_determinism_with_seed(toy_reactions, toy_posts, toy_config):
    a = MatrixFactorization(toy_config, seed=7).fit(toy_reactions, toy_posts)
    b = MatrixFactorization(toy_config, seed=7).fit(toy_reactions, toy_posts)
    assert abs(a.b_post["A"] - b.b_post["A"]) < 1e-12
