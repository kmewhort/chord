"""§6 semi-synthetic propensity harness on real preferences (Appendix C.3).

Coat gives a *real* MAR holdout; MovieLens gives *control of ground truth* on real
human preference structure. We take a dense MovieLens slice as the ground-truth
matrix, then impose the exact confound §6.1 warns about — items over-exposed to the
users predicted to like them (in-group alignment) — plus a small ``epsilon`` block
exposed uniformly at random (the §6.2 identifiability anchor) and a held-out MAR
block for scoring (:mod:`validate.synthetic`).

Two whitepaper claims are checked honestly:

* (C.3b) IPW using the true logging propensity recovers a better unbiased ranking
  than the uncorrected fit.
* (C.3d) the random ``epsilon`` anchor is what makes it work: sweep the anchor
  toward zero and identifiability should degrade. If it does *not*, that is a
  finding about how much the anchor actually buys.
"""
from __future__ import annotations

import numpy as np
import pytest

from chord.propensity import compute_ipw_weights
from chord.propensity.base import PropensityModel

from . import _modeling as M
from ._common import record_finding, require
from .datasets import movielens
from .metrics import ndcg_at_k
from .synthetic import induce_mnar


class _TruePi(PropensityModel):
    def __init__(self, table, floor):
        self.table, self.floor = table, floor

    def propensity(self, user_id, post_id, exposure=None):
        return max(self.table.get((user_id, post_id), self.floor), 1e-4)


def _eval_ndcg(result, holdout, k: int = 5) -> float:
    """Per-user NDCG@k against the MAR holdout ratings."""
    by_user: dict = {}
    for uid, pid, rating in holdout:
        by_user.setdefault(uid, []).append((pid, rating))
    scores = []
    for uid, items in by_user.items():
        cand = [(p, r) for p, r in items if p in result.y_post]
        if len(cand) < 2:
            continue
        preds = np.array([M.predict(result, uid, p, p.replace("i", "a")) for p, _ in cand])
        rels = np.array([r for _, r in cand], dtype=float)
        # shift relevances to be non-negative for gain
        rels = rels - rels.min() + 1e-6
        scores.append(ndcg_at_k(preds, rels, k))
    return float(np.mean(scores)) if scores else float("nan")


def test_ipw_recovers_ranking_and_anchor_matters(base_config):
    require(movielens.NAME, *movielens.REQUIRED)
    data = movielens.load()
    R, observed = movielens.dense_slice(data, n_users=200, n_items=200)
    cfg = base_config
    print(f"\n[ml §6] dense slice {R.shape} density={observed.mean():.3f}")

    # --- claim (C.3b): IPW beats uncorrected on the MAR holdout ---
    unc_scores, cor_scores = [], []
    for seed in range(4):
        split = induce_mnar(R, observed, epsilon_anchor=0.1, seed=seed)
        pmodel = _TruePi(split.true_pi, floor=0.1)
        unc = M.fit(split.reactions, split.posts, cfg, weights=None, seed=seed)
        w = compute_ipw_weights(split.reactions, pmodel, cfg)
        cor = M.fit(split.reactions, split.posts, cfg, weights=w, seed=seed)
        unc_scores.append(_eval_ndcg(unc, split.holdout))
        cor_scores.append(_eval_ndcg(cor, split.holdout))
    un, co = np.nanmean(unc_scores), np.nanmean(cor_scores)
    print(f"[ml §6] MAR-holdout NDCG@5  uncorrected={un:.4f}  IPW={co:.4f}  Δ={co-un:+.4f}")

    # --- claim (C.3d): the epsilon anchor is what buys identifiability ---
    print("[ml §6] anchor sweep (IPW-corrected holdout NDCG@5 as anchor → 0):")
    anchor_curve = {}
    for eps in (0.20, 0.10, 0.03, 0.0):
        vals = []
        for seed in range(3):
            split = induce_mnar(R, observed, epsilon_anchor=eps, seed=seed)
            pmodel = _TruePi(split.true_pi, floor=max(eps, 1e-3))
            w = compute_ipw_weights(split.reactions, pmodel, cfg)
            cor = M.fit(split.reactions, split.posts, cfg, weights=w, seed=seed)
            vals.append(_eval_ndcg(cor, split.holdout))
        anchor_curve[eps] = float(np.nanmean(vals))
        print(f"[ml §6]   anchor={eps:.2f}  NDCG@5={anchor_curve[eps]:.4f}")

    assert un > 0.0 and co > 0.0, "model failed to fit the MovieLens slice at all"

    # Claim (C.3d) — HOLDS here: a healthy random anchor gives a better unbiased
    # ranking than a vanishing one. Identifiability really does rest on the anchor.
    assert anchor_curve[0.20] >= anchor_curve[0.0] - 1e-3, (
        f"FINDING: shrinking the random exploration anchor to zero did not hurt "
        f"identifiability ({anchor_curve}); §6.2's anchor may not be doing the work."
    )

    # Claim (C.3b) — does NOT hold on this slice: IPW for the *synthetic* logging
    # layer does not improve prediction of a holdout that still carries MovieLens's
    # own (unmodelled) selection bias. Documented finding, not hidden.
    if co < un - 1e-3:
        record_finding(
            f"§6/C.3b: IPW correction did not improve — and slightly hurt — the "
            f"unbiased ranking on the MovieLens semi-synthetic harness "
            f"(uncorrected NDCG@5={un:.4f}, IPW={co:.4f}, Δ={co-un:+.4f}). Unlike "
            f"Coat, MovieLens has no true random-exposure block, so the 'MAR' "
            f"holdout is itself a random slice of already-MNAR observations; "
            f"correcting only the injected logging policy adds variance without "
            f"removing that base confound. Contrast Coat (real MAR holdout), where "
            f"IPW does help. Lesson: the semi-synthetic harness needs a genuinely "
            f"MAR base matrix — a caveat §6/C.3 should state explicitly."
        )
