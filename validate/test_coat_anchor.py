"""Exploration-anchor de-confounding, on Coat's real random-exposure block (§6.2/§13.10).

The exploration-anchor cap bounds organic reception at the upper confidence bound of
reception among *unconfounded* (uniform-random) exposures. Coat is the one real
dataset that carries such a block: ``train`` is self-selected (MNAR — people rate what
they chose), ``test`` is uniformly random (MAR). So we can validate the cap's core
claim directly, with no simulator: the MAR block is the anchor and the ground truth.

Per item, the self-selected (MNAR) mean *overstates* the true mean (selection bias);
capping it at the UCB of a noisy subsample of the random (MAR) block should pull the
over-stated items back toward the truth. Ranking is expected to be roughly unchanged —
the cap corrects the *level*, and organic selection bias is fairly uniform; its sharp
edge is against *manipulation* (a ring lifting one item), which Coat does not contain.
"""
from __future__ import annotations

import numpy as np
import pytest

from ._common import require
from .datasets import coat

Z, SIGMA, PRIOR_N = 1.0, 0.5, 3.0


def _signed(m):
    return (m - 3.0) / 2.0


@pytest.fixture(scope="module")
def deconfounding():
    require(coat.NAME, *coat.REQUIRED)
    d = coat.load()
    rng = np.random.default_rng(0)
    mnar, ucb, true = [], [], []
    for j in range(d.n_items):
        tr = d.train[:, j]; tr = tr[tr > 0]
        te = d.test[:, j]; te = te[te > 0]
        if len(tr) < 3 or len(te) < 6:
            continue
        perm = rng.permutation(len(te)); h = len(te) // 2
        anc, ev = te[perm[:h]], te[perm[h:]]          # anchor half, eval half (= truth)
        mnar.append(float(_signed(tr).mean()))
        m = float(_signed(anc).sum() / (len(anc) + PRIOR_N))   # shrunk toward 0
        ucb.append(m + Z * SIGMA / np.sqrt(len(anc) + PRIOR_N))
        true.append(float(_signed(ev).mean()))
    mnar, ucb, true = map(np.array, (mnar, ucb, true))
    capped = np.minimum(mnar, ucb)
    out = dict(
        n=len(true),
        mnar_bias=float(np.mean(mnar - true)),
        capped_bias=float(np.mean(capped - true)),
        mnar_err=float(np.abs(mnar - true).mean()),
        capped_err=float(np.abs(capped - true).mean()),
        binds=float(np.mean(capped < mnar - 1e-9)),
    )
    print(f"\n[coat anchor] {out['n']} items | signed bias MNAR={out['mnar_bias']:+.3f} "
          f"capped={out['capped_bias']:+.3f} | |err| {out['mnar_err']:.3f}->{out['capped_err']:.3f} "
          f"| cap binds on {out['binds']:.0%} of items")
    return out


def test_self_selection_biases_reception_upward(deconfounding):
    # The premise: on real data, the self-selected (organic) mean overstates the true
    # (random-exposure) mean.
    assert deconfounding["mnar_bias"] > 0.05, (
        f"expected upward selection bias, got {deconfounding['mnar_bias']:+.3f}"
    )


def test_exploration_anchor_cap_de_confounds(deconfounding):
    # The cap pulls the over-stated items back toward the unconfounded truth: less
    # signed bias and less absolute error against the held-out MAR block.
    assert 0 <= deconfounding["capped_bias"] < deconfounding["mnar_bias"], (
        f"cap should reduce the upward bias ({deconfounding['mnar_bias']:+.3f} -> "
        f"{deconfounding['capped_bias']:+.3f})"
    )
    assert deconfounding["capped_err"] < deconfounding["mnar_err"], (
        f"cap should reduce error toward truth ({deconfounding['mnar_err']:.3f} -> "
        f"{deconfounding['capped_err']:.3f})"
    )
