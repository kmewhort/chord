import numpy as np
import pytest

from chord import ChordConfig, Post, UserKnobs
from chord.model import fit_divisiveness
from chord.feed import (
    Candidate,
    FactorContext,
    blended_value,
    greedy_assemble,
    value,
)


# --------------------------------------------------------------- value V(u,p)
def test_M_dial_shifts_between_bridging_and_personalization(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    # left user 0 evaluating the left-partisan post B
    v_person = value(0, "B", b_lcb=0.0, result=fitted, divisiveness=dm,
                     knobs=UserKnobs(M=0.0, rho=1.0))
    v_bridge = value(0, "B", b_lcb=0.0, result=fitted, divisiveness=dm,
                     knobs=UserKnobs(M=1.0, rho=1.0))
    # M=0 keeps the personalization (aligned) term and no divisiveness penalty;
    # M=1 drops personalization and applies the divisiveness penalty -> lower.
    assert v_person > v_bridge


def test_value_falls_back_to_blcb_for_unseen_post(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    v = value(0, "unseen", b_lcb=0.42, result=fitted, divisiveness=dm, knobs=UserKnobs())
    assert v == 0.42


def test_rho_zero_removes_divisiveness_penalty(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    v = value(0, "B", 0.0, fitted, dm, UserKnobs(M=1.0, rho=0.0))
    # with rho=0 and M=1, personalization is off and penalty is off -> just b_lcb
    assert abs(v - 0.0) < 1e-9


def test_blended_value_respects_theta(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    post = Post("A", "auth1")
    ctx = FactorContext(0, post, b_lcb=1.0, result=fitted, divisiveness=dm,
                        knobs=UserKnobs(M=1.0, theta={"bridge": 1.0, "recency": 1.0}),
                        extras={"recency": 3.0})
    v = blended_value(ctx)
    # bridge factor ~ value(...), recency factor = 3.0, equally weighted -> average
    bridge_only = value(0, "A", 1.0, fitted, dm, UserKnobs(M=1.0))
    assert min(bridge_only, 3.0) <= v <= max(bridge_only, 3.0)


# ----------------------------------------------------------- greedy assembly
def test_greedy_selects_highest_value_first():
    cands = [
        Candidate("p1", "a1", base_value=1.0),
        Candidate("p2", "a2", base_value=3.0),
        Candidate("p3", "a3", base_value=2.0),
    ]
    res = greedy_assemble(cands, n_slots=2, epsilon=0.0)
    assert res.selected[0] == "p2"


def test_author_budget_caps_exposure():
    # One author floods with high-value posts; budget caps how many appear.
    cands = [Candidate(f"p{i}", "spammer", base_value=10.0) for i in range(5)]
    cands.append(Candidate("other", "b", base_value=1.0))
    res = greedy_assemble(cands, n_slots=4, epsilon=0.0,
                          author_budgets={"spammer": 2.0})
    picked_spam = [p for p in res.selected if p != "other"]
    assert len(picked_spam) <= 2  # budget of 2 (cost 1 each)


def test_exploration_floor_reserves_slots():
    # High-value seen posts vs low-value exploration posts; the floor still
    # reserves exploration slots.
    cands = [Candidate(f"seen{i}", f"a{i}", base_value=10.0, is_exploration=False)
             for i in range(8)]
    cands += [Candidate(f"new{i}", f"n{i}", base_value=0.0, exploration_value=0.1,
                        is_exploration=True) for i in range(4)]
    res = greedy_assemble(cands, n_slots=10, epsilon=0.2)  # floor = ceil(2) = 2
    assert res.exploration_count >= 2


def test_diverse_approval_is_submodular():
    # Two posts covering the SAME cluster give diminishing returns vs two posts
    # covering DIFFERENT clusters.
    same = [
        Candidate("p1", "a", base_value=1.0, approval_coverage=np.array([1.0, 0.0])),
        Candidate("p2", "b", base_value=1.0, approval_coverage=np.array([1.0, 0.0])),
    ]
    diff = [
        Candidate("q1", "a", base_value=1.0, approval_coverage=np.array([1.0, 0.0])),
        Candidate("q2", "b", base_value=1.0, approval_coverage=np.array([0.0, 1.0])),
    ]
    r_same = greedy_assemble(same, n_slots=2, epsilon=0.0, coverage_weight=1.0, n_clusters=2)
    r_diff = greedy_assemble(diff, n_slots=2, epsilon=0.0, coverage_weight=1.0, n_clusters=2)
    # covering two distinct regions yields higher total objective (diverse approval)
    assert r_diff.objective > r_same.objective


def test_empty_or_zero_slots():
    assert greedy_assemble([], 5, 0.0).selected == []
    assert greedy_assemble([Candidate("p", "a", 1.0)], 0, 0.0).selected == []


def test_greedy_one_minus_1_over_e_on_coverage():
    # Sanity: greedy submodular coverage is within (1-1/e) of the optimum. Build a
    # small instance where the optimum is known.
    cands = [
        Candidate("p1", "a", base_value=0.0, approval_coverage=np.array([1.0, 1.0, 0.0])),
        Candidate("p2", "b", base_value=0.0, approval_coverage=np.array([1.0, 0.0, 1.0])),
        Candidate("p3", "c", base_value=0.0, approval_coverage=np.array([0.0, 1.0, 1.0])),
    ]
    res = greedy_assemble(cands, n_slots=2, epsilon=0.0, coverage_weight=1.0, n_clusters=3)
    # optimum coverage value for 2 items = sqrt over union; greedy must reach
    # at least (1-1/e) of the best pair.
    from chord.feed.assembly import _coverage_value
    best = max(
        _coverage_value(a.approval_coverage + b.approval_coverage)
        for a in cands for b in cands if a.post_id != b.post_id
    )
    assert res.objective >= (1 - 1 / np.e) * best - 1e-9
