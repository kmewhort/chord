import numpy as np
import pytest

from chord import ChordConfig
from chord.economy import AuthorBudgetLedger, BetaPosterior, ExplorationPool


# ------------------------------------------------------------------ budget
def test_budget_defaults_to_base():
    ledger = AuthorBudgetLedger(ChordConfig(budget_B0=10.0))
    assert ledger.budget("new_author") == 10.0


def test_budget_replenished_by_strength():
    cfg = ChordConfig(budget_B0=5.0, budget_eta=2.0, budget_max=1000.0)
    ledger = AuthorBudgetLedger(cfg)
    # author "good" earned strength 1.0 over 3 exposures; "bad" earned nothing
    ledger.replenish(
        realized_strength={"pg": 1.0, "pb": -1.0},
        exposure={"pg": 3.0, "pb": 3.0},
        post_identity={"pg": "good", "pb": "bad"},
    )
    assert ledger.budget("good") == 5.0 + 2.0 * (1.0 * 3.0)
    assert ledger.budget("bad") == 5.0  # negative strength does not replenish


def test_budget_binds_to_identity_not_account():
    # Two sockpuppet accounts mapping to one identity share a single budget (§10).
    cfg = ChordConfig(budget_B0=5.0, budget_eta=1.0, budget_max=1000.0)
    ledger = AuthorBudgetLedger(cfg)
    ledger.replenish(
        realized_strength={"p1": 1.0, "p2": 1.0},
        exposure={"p1": 2.0, "p2": 2.0},
        post_identity={"p1": "same_human", "p2": "same_human"},  # both -> one id
    )
    # replenishment aggregates across both puppets: 5 + 1*(1*2 + 1*2) = 9
    assert ledger.budget("same_human") == 9.0


def test_budget_bounded_by_max():
    cfg = ChordConfig(budget_B0=5.0, budget_eta=100.0, budget_max=20.0)
    ledger = AuthorBudgetLedger(cfg)
    ledger.replenish({"p": 10.0}, {"p": 10.0}, {"p": "a"})
    assert ledger.budget("a") == 20.0  # clamped


# ------------------------------------------------------------- exploration
def test_beta_posterior_mean_and_update():
    b = BetaPosterior(2.0, 2.0)
    assert abs(b.mean - 0.5) < 1e-12
    b.update(1.0)  # a success
    assert b.mean > 0.5


def test_base_rate_prior_not_flat():
    # The §8 fix: prior mean = empirical newcomer base rate, NOT 0.5.
    cfg = ChordConfig(newcomer_base_rate=0.1)
    pool = ExplorationPool(cfg, seed=0)
    pool.register("p")
    assert abs(pool.posteriors["p"].mean - 0.1) < 1e-9


def test_audition_saturates_and_closes():
    cfg = ChordConfig(exploration_saturation_var=0.02, newcomer_base_rate=0.3)
    pool = ExplorationPool(cfg, seed=0)
    pool.register("p")
    assert pool.is_open("p")
    # feed many consistent observations -> variance shrinks -> closes
    for _ in range(200):
        pool.observe("p", 0.8)
    assert not pool.is_open("p")


def test_sample_score_in_unit_interval():
    pool = ExplorationPool(ChordConfig(), seed=1)
    pool.register("p")
    for _ in range(50):
        s = pool.sample_score("p")
        assert 0.0 <= s <= 1.0


def test_closed_audition_ignores_further_observations():
    cfg = ChordConfig(exploration_saturation_var=0.02)
    pool = ExplorationPool(cfg, seed=0)
    pool.register("p")
    for _ in range(300):
        pool.observe("p", 0.9)
    assert not pool.is_open("p")
    mean_before = pool.posterior_mean("p")
    pool.observe("p", 0.0)  # should be ignored
    assert pool.posterior_mean("p") == mean_before
