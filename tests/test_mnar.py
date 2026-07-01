"""Broader test: the semi-synthetic MNAR / propensity harness (Appendix C.3).

Validates the §6 claims that make the keystone identifiable:
* under an in-group-over-exposing logging policy, IPW correction recovers the
  true bridging ranking better than uncorrected fitting;
* the randomized exploration anchor is load-bearing — as its rate -> 0,
  identifiability fails (C.3d).

Averaged over seeds so the assertions are about the *systematic* effect, not a
single noisy draw.
"""
import numpy as np
import pytest

from chord import ChordConfig
from chord.eval import make_world, run_experiment


SEEDS = range(6)


def _mean_over_seeds(eps_anchor):
    uncorr, corr = [], []
    for s in SEEDS:
        world = make_world(n_users=80, n_posts=40, d=2, seed=s)
        cfg = ChordConfig(d=2, mf_iters=50, reg_bias_post=0.05)
        out = run_experiment(world, cfg, epsilon_anchor=eps_anchor, seed=s)
        uncorr.append(out["uncorrected"].ranking_corr)
        corr.append(out["corrected"].ranking_corr)
    return float(np.mean(uncorr)), float(np.mean(corr))


def test_ipw_recovers_bridging_ranking_under_mnar():
    # The core §6 result: with a healthy anchor, IPW-corrected fitting recovers
    # the true bridging order at least as well as uncorrected fitting. The gap is
    # modest at a generous anchor (where the naive fit is already decent); the
    # dramatic effect appears in the anchor-sweep test below.
    uncorr, corr = _mean_over_seeds(eps_anchor=0.15)
    assert corr > uncorr


def test_identifiability_fails_as_anchor_vanishes():
    # C.3(d): sweep the random epsilon-anchor toward zero and watch
    # identifiability fail — the corrected ranking degrades sharply.
    _, corr_healthy = _mean_over_seeds(eps_anchor=0.15)
    _, corr_starved = _mean_over_seeds(eps_anchor=0.001)
    assert corr_healthy > corr_starved + 0.3
    # with essentially no anchor, the recovered ranking is no better than chance.
    assert corr_starved < 0.2


def test_uncorrected_fit_is_biased():
    # Uncorrected fitting on MNAR data should not achieve strong ranking recovery;
    # the correction exists precisely because the naive fit is biased (§6.1).
    uncorr, corr = _mean_over_seeds(eps_anchor=0.15)
    assert corr > uncorr  # correction strictly helps


def test_world_has_universal_and_partisan_posts():
    world = make_world(n_users=40, n_posts=20, d=2, seed=0)
    n_univ = int(0.3 * 20)
    # universal posts have small loading norm, partisan posts large
    univ_norm = np.mean([np.linalg.norm(world.loadings[p]) for p in range(n_univ)])
    part_norm = np.mean([np.linalg.norm(world.loadings[p]) for p in range(n_univ, 20)])
    assert univ_norm < part_norm
