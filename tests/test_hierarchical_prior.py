"""E9: the hierarchical author×cluster prior closes the §9 leniency gap.

With the global-mean prior μ, an untested one-sided firehose post regresses to neutral, so
§8's budget has to bound its reach. The hierarchical prior (author history → cluster mean →
μ) makes B_LCB itself predict-low on a firehose *before* the budget bites — raising
delivered true value. Deterministic (author history is a function of the data), and gated
(`config.hierarchical_prior`, default off) so it doesn't disturb the μ-tuned results.
"""
import numpy as np

from chord.config import ChordConfig
from chord.model.priors import AuthorClusterReception, hierarchical_priors
from chord.simulator import Simulator


def _run(hier):
    ratio, tv = [], []
    for s in (1, 2, 3, 4):
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                          hierarchical_prior=hier)
        r = Simulator(config=cfg, n_users=36, n_slots=6, seed=s).run("chord", n_windows=10)
        fh = np.mean([m.firehose_reach_per_post for m in r.metrics[3:]])
        uni = np.mean([m.universal_reach_per_post for m in r.metrics[3:]])
        ratio.append(fh / uni)
        tv.append(r.tail("true_value", 4))
    return float(np.mean(ratio)), float(np.mean(tv))


def test_hierarchical_prior_suppresses_firehose_and_raises_value():
    off_ratio, off_tv = _run(False)
    on_ratio, on_tv = _run(True)
    print(f"\n[E9] firehose/quality reach ratio {off_ratio:.2f}->{on_ratio:.2f}; "
          f"true_value {off_tv:.4f}->{on_tv:.4f}")
    # The headline: B_LCB itself now suppresses untested one-sided content, so more true
    # (quality × bridged) value is delivered.
    assert on_tv > off_tv + 0.02
    # and the firehose is suppressed relative to quality (directional; the effect is
    # larger in aggregate reach than per-seed ratio).
    assert on_ratio < off_ratio


def test_quality_basis_denies_the_broadly_approved_bait_but_keeps_firehose_demotion():
    """The rebase (§13.11): the author lift is asymmetric — approval can only *lower* the
    prior, and *raising* it above the cluster baseline is licensed only by earned vouches.
    So a broadly-approved-but-unvouched BAIT is no longer propped up (as it was under the
    approval basis), a genuinely-vouched MERIT author still gets the head-start, and a
    disliked FIREHOSE is still demoted below the baseline."""
    n_clusters, mu, n0, n0a = 2, 0.0, 8.0, 8.0
    approval, vouch = AuthorClusterReception(decay=1.0), AuthorClusterReception(decay=1.0)
    for _ in range(3):                                  # accumulate history
        approval.update({"pb": {0: (20.0, 0.8), 1: (20.0, 0.8)},    # BAIT: broad approval
                         "pm": {0: (20.0, 0.8), 1: (20.0, 0.8)},    # MERIT: same approval...
                         "pf": {0: (20.0, -0.7), 1: (20.0, -0.7)}}, # FIREHOSE: disliked
                        {"pb": "BAIT", "pm": "MERIT", "pf": "FIRE"})
        vouch.update({"pm": {0: (10.0, 0.9), 1: (10.0, 0.9)}},      # ...but only MERIT is vouched
                     {"pm": "MERIT"})
    reception = {"b": {}, "m": {}, "f": {}}             # untested new posts → prior only
    authors = {"b": "BAIT", "m": "MERIT", "f": "FIRE"}

    appr = hierarchical_priors(reception, authors, approval, mu, n0, n0a, n_clusters)
    qual = hierarchical_priors(reception, authors, approval, mu, n0, n0a, n_clusters,
                               vouch_tracker=vouch)
    # approval basis props up BOTH the bait and the merit author (indistinguishable by approval)
    assert appr["b"][0] > 0.3 and appr["m"][0] > 0.3
    # quality basis: the bait is capped at the cluster baseline (no vouches → no lift)...
    assert qual["b"][0] <= mu + 1e-9
    # ...the merit author keeps the lift (vouches license it)...
    assert qual["m"][0] > qual["b"][0] + 0.2
    # ...and the disliked firehose is still demoted below the baseline under both bases.
    assert appr["f"][0] < mu and qual["f"][0] < mu
