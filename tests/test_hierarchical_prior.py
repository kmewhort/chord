"""E9: the hierarchical author×cluster prior closes the §9 leniency gap.

With the global-mean prior μ, an untested one-sided firehose post regresses to neutral, so
§8's budget has to bound its reach. The hierarchical prior (author history → cluster mean →
μ) makes B_LCB itself predict-low on a firehose *before* the budget bites — raising
delivered true value. Deterministic (author history is a function of the data), and gated
(`config.hierarchical_prior`, default off) so it doesn't disturb the μ-tuned results.
"""
import numpy as np

from chord.config import ChordConfig
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
