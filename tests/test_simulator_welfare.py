"""Counterfactual welfare: does bridging beat engagement end-to-end? (§1, App C.4)

Runs CHORD against the engagement / oracle / random baselines on the *same* seeded
worlds and compares ground-truth welfare the rankers cannot see: the true bridged
value delivered, the affective divisiveness exposed, viewer satisfaction, and how
well the estimator recovered the true opinion geometry. Averaged over seeds to test
the systematic effect, per the simulator/MNAR convention.

The world is deliberately non-circular (a hidden opinion axis the d=2 model can't
represent; toxicity drives engagement while quality — genuine value — barely moves
reactions), so these are not self-fulfilling.
"""
import numpy as np
import pytest

from chord.simulator import Simulator

RANKERS = ["oracle", "chord", "engagement", "random"]
SEEDS = (1, 2, 3)
KEYS = ("true_value", "divisiveness", "toxicity", "satisfaction", "recovery")


@pytest.fixture(scope="module")
def comparison():
    agg = {r: {k: [] for k in KEYS} for r in RANKERS}
    for seed in SEEDS:
        sim = Simulator(n_users=36, d=2, n_slots=6, seed=seed, adaptive_authors=False)
        res = sim.compare(RANKERS, n_windows=7)
        for r in RANKERS:
            for k in KEYS:
                agg[r][k].append(res[r].tail(k, n=4))
    def safe_mean(v):
        finite = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(finite)) if finite else float("nan")
    means = {r: {k: safe_mean(v) for k, v in d.items()} for r, d in agg.items()}
    print("\n[sim welfare] mean over seeds", SEEDS)
    print(f"[sim welfare] {'ranker':<12} " + "  ".join(f"{k:>12}" for k in KEYS))
    for r in RANKERS:
        print(f"[sim welfare] {r:<12} " + "  ".join(f"{means[r][k]:>12.4f}" for k in KEYS))
    return means


def test_chord_delivers_more_true_value_than_engagement(comparison):
    # The headline §1 claim: optimizing tested cross-cluster value delivers more
    # genuine (quality × bridged) value than optimizing engagement.
    assert comparison["chord"]["true_value"] > comparison["engagement"]["true_value"]


def test_chord_exposes_less_divisiveness_than_engagement(comparison):
    # Bridging shows people less affectively-divisive content than engagement does.
    assert comparison["chord"]["divisiveness"] < comparison["engagement"]["divisiveness"]


def test_engagement_wins_satisfaction_the_expected_tradeoff(comparison):
    # Honest tradeoff, not a bug: engagement maximizes *individual* approval, so it
    # should win raw satisfaction — CHORD trades some of it for bridged value.
    assert comparison["engagement"]["satisfaction"] >= comparison["chord"]["satisfaction"] - 0.02


def test_exploration_anchor_improves_recovery(comparison):
    # §6.2 in-loop: CHORD's floored random exposure keeps the opinion estimate
    # identifiable, so it recovers the true geometry better than engagement, whose
    # feedback is fully confounded (it never explores).
    assert comparison["chord"]["recovery"] > comparison["engagement"]["recovery"]


def test_value_ordering_oracle_best_random_weak(comparison):
    # Sanity on the ground-truth scale: the cheating oracle tops true value and
    # beats the uninformed random baseline by a clear margin.
    assert comparison["oracle"]["true_value"] > comparison["chord"]["true_value"]
    assert comparison["oracle"]["true_value"] > comparison["random"]["true_value"] + 0.05
