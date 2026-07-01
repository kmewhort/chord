"""Does a candidate fix flatten the §5 Sybil-ring curve? (prototype, on RfA)

Companion to the F1 finding in test_signed_nets_eigentrust.py. Re-runs the exact
ring K-sweep under the three research-proposed λ-iteration tweaks and asserts that
at least one turns the linear-in-K influence harvest into a flat curve — i.e. the
finding is fixable, and by which mechanism.
"""
from __future__ import annotations

import numpy as np
import pytest

from chord.config import ChordConfig

from chord.rater.eigentrust import build_trust_matrix

from . import _modeling as M
from ._common import require
from .datasets import signed_nets as sn
from .experiments import sybil_hardening as sh
from .metrics import spearman


@pytest.fixture(scope="module")
def rfa():
    require(sn.NAME, *sn.REQUIRED)
    votes = sn.load_votes()
    reactions, posts, users = sn.to_reactions(votes, min_src_votes=15, min_tgt_votes=30)
    cfg = ChordConfig(d=2, mf_iters=30, reg_embedding=0.08, affective_weighting=False)
    result = M.fit(reactions, posts, cfg, seed=0)
    return dict(reactions=reactions, posts=posts, users=users, result=result, cfg=cfg)


def test_candidate_defenses_flatten_the_ring(rfa):
    Ks = (5, 20, 50, 100)
    curves = sh.sweep(rfa["reactions"], rfa["posts"], rfa["users"], rfa["result"],
                      rfa["cfg"], Ks=Ks)

    header = "  ".join(f"K={k:<4}" for k in Ks)
    print(f"\n[rfa §5 fix] target percentile among real editors, by defense:")
    print(f"[rfa §5 fix] {'variant':<16} {header}")
    for name, curve in curves.items():
        row = "  ".join(f"{curve[k]:5.1f}" for k in Ks)
        print(f"[rfa §5 fix] {name:<16} {row}")

    baseline_worst = max(curves["baseline"].values())
    # A defense "works" if the target never cracks the top decile even at K=100.
    best_name, best_worst = min(
        ((n, max(c.values())) for n, c in curves.items() if n != "baseline"),
        key=lambda t: t[1],
    )
    print(f"[rfa §5 fix] baseline worst percentile={baseline_worst:.1f}; "
          f"best defense '{best_name}' worst={best_worst:.1f}")

    assert baseline_worst >= 90.0, "expected the baseline ring to reach top decile"
    assert best_worst < 90.0, (
        f"No candidate defense flattened the ring (best '{best_name}' still reached "
        f"{best_worst:.0f}th percentile). The §5 fix does not work as predicted."
    )

    # Legitimacy: a defense that flattens the ring but scrambles honest influence
    # is useless. On the *clean* graph (no ring), out-diversity λ must still rank
    # real editors much like the baseline.
    T_clean = build_trust_matrix(rfa["reactions"], rfa["posts"], rfa["result"], rfa["users"])
    lam_base = sh.eigentrust_variant(T_clean, rfa["cfg"])
    lam_odiv = sh.eigentrust_variant(T_clean, rfa["cfg"], transmit_w=sh.row_entropy_weights(T_clean))
    rho = spearman(lam_base, lam_odiv)
    print(f"[rfa §5 fix] out-diversity preserves honest ranking: "
          f"Spearman(baseline λ, out-div λ) = {rho:.3f} on the clean graph")
    assert rho > 0.9, (
        f"out-diversity distorts honest influence (Spearman={rho:.3f}); it flattens "
        f"the ring but at too high a cost to legitimate raters."
    )
