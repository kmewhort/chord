"""Broader test: adversarial robustness (§10).

* Sybil / sockpuppets — the budget binds to identity, not accounts, so sharding
  across puppets gains nothing (§8, §10); fresh accounts get ~zero trust (§5).
* Brigading — a brigade of fresh accounts has no cross-divide trust path, AND a
  brigade *creates* a split distribution that the divisiveness term and the
  B_LCB min-over-clusters penalize. Gaming lowers the score.
"""
import numpy as np
import pytest

from chord import Chord, ChordConfig, Exposure, ExposureSource, Post, Reaction, UserKnobs
from chord.economy import AuthorBudgetLedger
from chord.ports import AccountAgeIdentityAdapter
from chord.propensity import UniformExplorationModel
from chord.rater import compute_lambda
from chord.model import MatrixFactorization


def test_sockpuppet_budget_sharding_gains_nothing():
    # Two puppet accounts for one human: replenishment aggregates to a single
    # identity budget, so posting the same content across two accounts does not
    # double the reach budget (§8/§10).
    cfg = ChordConfig(budget_B0=5.0, budget_eta=1.0, budget_max=1000.0)
    idp = AccountAgeIdentityAdapter(aliases={"acc1": "human", "acc2": "human"})

    # Case A: one identity, two puppet posts.
    ledger_shard = AuthorBudgetLedger(cfg)
    ledger_shard.replenish(
        realized_strength={"p1": 1.0, "p2": 1.0},
        exposure={"p1": 2.0, "p2": 2.0},
        post_identity={"p1": idp.identity_of("acc1"), "p2": idp.identity_of("acc2")},
    )
    # Case B: one identity, one post with the same total earned strength*exposure.
    ledger_single = AuthorBudgetLedger(cfg)
    ledger_single.replenish(
        realized_strength={"p": 1.0},
        exposure={"p": 4.0},
        post_identity={"p": "human"},
    )
    # Sharding across puppets yields no more budget than the combined single post.
    assert ledger_shard.budget("human") == ledger_single.budget("human")


def test_fresh_sybil_gets_minimal_trust():
    # An honest author earns approval from MANY *diverse* raters; a Sybil author is
    # boosted only by a single-target colluding puppet. EigenTrust (rater-outgoing
    # normalized, with the §5 out-diversity transmit weight) must give the honest
    # author more credibility than the Sybil: trust flows from many independent
    # raters who spread their approval, not from one single-purpose puppet (§5, §10).
    posts = {"pH1": Post("pH1", "H"), "pH2": Post("pH2", "H2"),
             "pH3": Post("pH3", "H3"), "sybil_post": Post("sybil_post", "S1")}
    rx = []
    for u in range(10):
        # honest raters approve several honest authors → diverse outgoing trust
        rx.append(Reaction(u, "pH1", 1.0))
        rx.append(Reaction(u, "pH2", 1.0))
        rx.append(Reaction(u, "pH3", 1.0))
    rx.append(Reaction("S2", "sybil_post", 1.0))     # lone single-target puppet
    users = list(range(10)) + ["S1", "S2", "H", "H2", "H3"]
    cfg = ChordConfig(d=3, mf_iters=40)
    res = MatrixFactorization(cfg, seed=0).fit(rx, posts)
    lam = compute_lambda(rx, posts, res, users, cfg)
    # the honest author out-accrues the Sybil author
    assert lam["H"] > lam["S1"]


def test_sybil_ring_cannot_harvest_influence():
    # The §5 ring defense (App C.5): a ring of K single-target puppets all boosting
    # one target must NOT lift that target above honest, diversely-approved authors.
    # Out-diversity zeroes each puppet's transmitted mass (out-degree 1), so the
    # target harvests nothing regardless of K.
    posts = {"pH1": Post("pH1", "H"), "pH2": Post("pH2", "H2"),
             "ring_post": Post("ring_post", "TARGET")}
    rx = []
    for u in range(8):                                # diverse honest approval
        rx.append(Reaction(u, "pH1", 1.0))
        rx.append(Reaction(u, "pH2", 1.0))
    K = 50
    for i in range(K):                                # a big single-target ring
        rx.append(Reaction(f"sybil{i}", "ring_post", 1.0))
    users = list(range(8)) + ["H", "H2", "TARGET"] + [f"sybil{i}" for i in range(K)]
    cfg = ChordConfig(d=3, mf_iters=40)
    res = MatrixFactorization(cfg, seed=0).fit(rx, posts)
    lam = compute_lambda(rx, posts, res, users, cfg)
    # even a 50-puppet ring leaves its target below the honestly-approved author
    assert lam["TARGET"] < lam["H"]


def test_brigade_creates_penalized_split():
    # A brigade that piles onto a post from one pole creates exactly the split
    # distribution B_LCB's min-over-clusters penalizes: the post scores LOW
    # bridged support despite high raw approval from the brigading cluster.
    posts = {"bridged": Post("bridged", "a1"), "brigaded": Post("brigaded", "a2")}
    rx, exps = [], []
    for u in range(10):
        left = u < 5
        # 'bridged' genuinely liked across both clusters
        rx.append(Reaction(u, "bridged", 1.0))
        # 'brigaded' loved by left (the brigade), muted by right
        rx.append(Reaction(u, "brigaded", 1.0 if left else -1.0))
        for pid in ("bridged", "brigaded"):
            exps.append(Exposure(u, pid, propensity=0.5, source=ExposureSource.ORGANIC))
    cfg = ChordConfig(d=4, mf_iters=40, n_clusters=2)
    chord = Chord(cfg, propensity_model=UniformExplorationModel(0.5), seed=1, inner_iters=3)
    chord.fit_window(rx, posts, exps)
    b = chord.state.bridging.b_lcb
    # The brigaded post's manufactured approval does not buy bridged support.
    assert b["bridged"] > b["brigaded"]


def test_brigade_gaming_lowers_score_vs_honest_broad_support():
    # Compare a post that earns honest broad support to one that gets the same
    # NUMBER of positive reactions but concentrated in one cluster (a brigade).
    # The concentrated one must not outrank the broad one on B_LCB.
    posts = {"broad": Post("broad", "a1"), "narrow": Post("narrow", "a2")}
    rx, exps = [], []
    for u in range(10):
        left = u < 5
        rx.append(Reaction(u, "broad", 0.6))                    # everyone mildly likes
        rx.append(Reaction(u, "narrow", 1.0 if left else -0.2))  # left loves, right meh
        for pid in ("broad", "narrow"):
            exps.append(Exposure(u, pid, propensity=0.5, source=ExposureSource.ORGANIC))
    cfg = ChordConfig(d=4, mf_iters=40, n_clusters=2)
    chord = Chord(cfg, propensity_model=UniformExplorationModel(0.5), seed=2, inner_iters=3)
    chord.fit_window(rx, posts, exps)
    b = chord.state.bridging.b_lcb
    assert b["broad"] > b["narrow"]
