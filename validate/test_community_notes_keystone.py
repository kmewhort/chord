"""§4 keystone vs the deployed Community Notes model (Appendix C.1/C.2 "start here").

Community Notes decides note status with its own bridging matrix factorization; the
published ``currentStatus`` (CURRENTLY_RATED_HELPFUL vs NOT) is that deployed
model's output — a strong external baseline. The honest question: does CHORD's
``B_LCB``, fit from the same rater x note signed ratings, rank the notes X itself
found helpful above the ones it found not-helpful?

History (Appendix C.5): the *original* B_LCB (subtractive ``min_c[r̂ − β·σ/√(n+1)]``)
was beaten by both ``b_p`` and a naive helpfulness mean — the penalty demoted
under-sampled clusters (noise, not risk). The shipped keystone now uses
empirical-Bayes **shrinkage** toward the population mean weighted by per-cluster
exposure, aggregated with the **nash** (Polis group-informed-consensus) aggregator.
This test verifies the fix: fed per-cluster exposure counts, B_LCB reaches ``b_p``
parity on a class-balanced sample (the pathology is gone). It cannot beat the naive
mean on the raw 95%-helpful slice — but nothing can, because CN's own label is
essentially a threshold on mean helpfulness (the genuine bridging gain shows up on
Polis, where the target differs from the mean; see test_keystone_variants.py).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import _modeling as M
from ._common import require
from .datasets import community_notes as cn
from .metrics import auc


def _balanced_auc(scores, labels, seed=0, draws=25):
    labels = np.asarray(labels)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    m = min(len(pos), len(neg))
    if m < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(pos, m, replace=False), neg])
        vals.append(auc(scores[idx], labels[idx]))
    return float(np.mean(vals))


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
    clusters = M.cluster(reactions, result, cfg)
    post_authors = {pid: p.author_id for pid, p in posts.items()}

    # B_LCB = empirical IPW-shrunk per-cluster reception (built inside M.bridging from
    # the reactions); a thinly-rated cluster regresses to the mean instead of penalized.
    scores = M.bridging(reactions, result, clusters, post_authors, cfg)

    mean_rating: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    for r in reactions:
        mean_rating[r.post_id] += r.value
        counts[r.post_id] += 1
    mean_rating = {p: mean_rating[p] / counts[p] for p in mean_rating}

    ids = [n for n in labels if n in scores.b_lcb]
    y = np.array([labels[n] for n in ids])
    blcb = np.array([scores.b_lcb[n] for n in ids])
    bp = np.array([scores.b_scalar[n] for n in ids])
    mr = np.array([mean_rating.get(n, 0.0) for n in ids])

    bal_lcb, bal_bp, bal_mean = (_balanced_auc(s, y) for s in (blcb, bp, mr))
    print(f"[cn §4] balanced AUC  B_LCB(nash)={bal_lcb:.4f}  b_p={bal_bp:.4f}  "
          f"mean-rating={bal_mean:.4f}  (n={len(ids)})")

    # Sanity: the keystone recovers the helpful/not decision far better than chance.
    assert bal_lcb > 0.55, (
        f"B_LCB barely predicts Community Notes status (balAUC={bal_lcb:.3f})."
    )
    # Fix verified (was: B_LCB ≪ b_p): with exposure-weighted shrinkage + nash, the
    # cross-cluster keystone is no longer beaten by its own scalar intercept.
    assert bal_lcb >= bal_bp - 0.01, (
        f"REGRESSION: B_LCB (balAUC={bal_lcb:.4f}) is still beaten by the scalar "
        f"b_p ({bal_bp:.4f}); the §4.2 shrinkage fix is not working on real data."
    )
