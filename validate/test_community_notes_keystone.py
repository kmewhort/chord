"""§4 keystone vs the deployed Community Notes model (Appendix C.1/C.2 "start here").

Community Notes decides note status with its own bridging matrix factorization; the
published ``currentStatus`` (CURRENTLY_RATED_HELPFUL vs NOT) is that deployed
model's output — a strong external baseline. The honest question: does CHORD's
``B_LCB``, fit from the same rater x note signed ratings, rank the notes X itself
found helpful above the ones it found not-helpful?

We report the AUC of three scores against the CN label:

* ``B_LCB`` — CHORD's tested cross-cluster bridged support (§4.2),
* ``b_p``   — the plain note intercept (the cheap scalar pre-filter),
* mean rating — a naive baseline ignoring who rated.

If ``B_LCB`` does not beat chance, the keystone fails against the deployed model.
If it does not beat the *naive mean*, bridging is not adding value over averaging —
either way a real finding, surfaced rather than hidden.
"""
from __future__ import annotations

import numpy as np

from . import _modeling as M
from ._common import record_finding, require
from .datasets import community_notes as cn
from .metrics import auc


def test_blcb_recovers_community_notes_helpfulness(base_config):
    require(cn.NAME, *cn.SLICE_FILES)
    sl = cn.load_slice()
    reactions, posts, labels = cn.to_reactions(sl)

    n_pos = int(sum(labels.values()))
    print(f"\n[cn §4] slice: {len(labels):,} notes ({n_pos:,} helpful / "
          f"{len(labels)-n_pos:,} not), {len(reactions):,} ratings, "
          f"{len({r.user_id for r in reactions}):,} raters")

    cfg = base_config
    result = M.fit(reactions, posts, cfg, seed=0)
    clusters = M.cluster(result, cfg, seed=0)
    post_authors = {pid: p.author_id for pid, p in posts.items()}
    scores = M.bridging(result, clusters, post_authors, cfg)

    # naive mean signed rating per note
    mean_rating: dict = {}
    counts: dict = {}
    for r in reactions:
        mean_rating[r.post_id] = mean_rating.get(r.post_id, 0.0) + r.value
        counts[r.post_id] = counts.get(r.post_id, 0) + 1
    for pid in mean_rating:
        mean_rating[pid] /= counts[pid]

    ids = [n for n in labels if n in scores.b_lcb]
    y = np.array([labels[n] for n in ids])
    auc_lcb = auc(np.array([scores.b_lcb[n] for n in ids]), y)
    auc_bp = auc(np.array([scores.b_scalar[n] for n in ids]), y)
    auc_mean = auc(np.array([mean_rating.get(n, 0.0) for n in ids]), y)

    print(f"[cn §4] AUC vs CN status  B_LCB={auc_lcb:.4f}  b_p={auc_bp:.4f}  "
          f"mean-rating={auc_mean:.4f}  (n={len(ids)})")

    # Weak claim — HOLDS: the keystone at least recovers the helpful/not decision
    # far better than chance.
    assert auc_lcb > 0.55, (
        f"B_LCB barely predicts Community Notes status (AUC={auc_lcb:.3f}); the §4 "
        f"keystone does not reconstruct the deployed bridging model at all here."
    )

    # Strong claim — the point of bridging: B_LCB should add value over naively
    # averaging helpfulness. On this slice it does NOT — it is beaten by both the
    # naive mean and even its own scalar pre-filter b_p. Documented finding.
    if not (auc_lcb >= auc_mean - 0.01 and auc_lcb >= auc_bp - 0.01):
        record_finding(
            f"§4 keystone adds no value over naive averaging on Community Notes: "
            f"AUC(B_LCB)={auc_lcb:.4f} is below AUC(mean-rating)={auc_mean:.4f} and "
            f"AUC(b_p)={auc_bp:.4f}. The cross-cluster LCB pessimism penalty (§4.2) "
            f"demotes notes seen by fewer clusters, which *reduces* agreement with "
            f"CN's own decision relative to just averaging signed helpfulness. Note "
            f"the slice is {n_pos}/{len(labels)} helpful (imbalanced), so all AUCs "
            f"are high; the ranking mean > b_p > B_LCB is the signal. Worth probing "
            f"whether the LCB penalty / clustering is miscalibrated on real data."
        )
