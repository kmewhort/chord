"""§6 propensity / MNAR, validated on Coat's real MAR holdout (Appendix C.3).

Coat is the cleanest real test of the propensity layer: the *train* block is
missing-not-at-random (users rated self-selected items) and the *test* block is
missing-completely-at-random (uniformly exposed items). The whitepaper's §6.1
claim is that fitting the keystone on MNAR data without correction biases the
estimate, and that inverse-propensity weighting (§6.2) recovers a ranking that
generalizes to the unconfounded holdout.

The honest test: fit on the MNAR train block *with* and *without* IPW (using
Coat's own learned propensities as the logging model), then score both against the
MAR test block with unbiased NDCG@k / AUC. If IPW does **not** beat the
uncorrected fit here, that is a genuine finding about the propensity layer — the
assertions are written to surface it, not to hide it.
"""
from __future__ import annotations

import numpy as np
import pytest

from chord.propensity import compute_ipw_weights
from chord.propensity.base import PropensityModel

from . import _modeling as M
from ._common import require
from .datasets import coat
from .metrics import auc, ndcg_at_k


def _evaluate(result, data: coat.CoatData, k: int = 5):
    """Unbiased NDCG@k and AUC of ``result`` against the MAR test block.

    For each user we rank their MAR-test items by predicted rating and score the
    ranking against the held-out true ratings (relevance = star rating; AUC label
    = rating >= 4). Items unseen in train (no embedding) are skipped.
    """
    ndcgs, aucs = [], []
    for u in range(data.n_users):
        items = np.nonzero(data.test[u])[0]
        cand = [j for j in items if f"i{j}" in result.y_post]
        if len(cand) < 2:
            continue
        preds = np.array([M.predict(result, f"u{u}", f"i{j}", f"a{j}") for j in cand])
        rels = np.array([data.test[u, j] for j in cand], dtype=float)
        ndcgs.append(ndcg_at_k(preds, rels, k))
        labels = (rels >= 4).astype(int)
        if 0 < labels.sum() < len(labels):
            aucs.append(auc(preds, labels))
    return float(np.mean(ndcgs)), float(np.mean(aucs))


class _TruePropensity(PropensityModel):
    def __init__(self, table):
        self.table = table

    def propensity(self, user_id, post_id, exposure=None):
        return max(self.table.get((user_id, post_id), 0.05), 1e-4)


def test_ipw_improves_unbiased_ranking_on_mar_holdout(base_config):
    require(coat.NAME, *coat.REQUIRED)
    data = coat.load()
    reactions, posts = coat.to_reactions(data.train)
    prop_table = coat.propensity_lookup(data)
    pmodel = _TruePropensity(prop_table)

    cfg = base_config

    unc_ndcg, unc_auc, cor_ndcg, cor_auc = [], [], [], []
    for seed in range(4):  # average over MF inits: test the systematic effect
        unc = M.fit(reactions, posts, cfg, weights=None, seed=seed)
        w = compute_ipw_weights(reactions, pmodel, cfg)
        cor = M.fit(reactions, posts, cfg, weights=w, seed=seed)
        n0, a0 = _evaluate(unc, data)
        n1, a1 = _evaluate(cor, data)
        unc_ndcg.append(n0); unc_auc.append(a0)
        cor_ndcg.append(n1); cor_auc.append(a1)

    un, uc = np.mean(unc_ndcg), np.mean(cor_ndcg)
    au, ac = np.mean(unc_auc), np.mean(cor_auc)
    print(f"\n[coat §6] MAR-holdout NDCG@5  uncorrected={un:.4f}  IPW={uc:.4f}  Δ={uc-un:+.4f}")
    print(f"[coat §6] MAR-holdout AUC     uncorrected={au:.4f}  IPW={ac:.4f}  Δ={ac-au:+.4f}")

    # Sanity: the fitted model must beat random on the unconfounded holdout at all.
    assert au > 0.5, f"uncorrected AUC {au:.3f} no better than chance — model broken"

    # The §6.1/§6.2 claim: IPW correction improves the unbiased holdout ranking.
    # A tiny tolerance guards against pure MF-seed noise; a real regression fails.
    assert uc >= un - 1e-3, (
        f"FINDING: IPW did not improve unbiased NDCG on Coat "
        f"(uncorrected={un:.4f}, IPW={uc:.4f}). The §6 propensity correction "
        f"does not pan out on this dataset."
    )
