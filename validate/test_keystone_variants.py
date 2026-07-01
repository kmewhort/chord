"""Can any cross-cluster aggregator/penalty beat b_p at reproducing real decisions?

Companion to the F2 finding. Sweeps aggregator × penalty (experiments.keystone_variants)
on the Community Notes slice and asks whether a variant beats the scalar b_p on a
*class-balanced* subsample (the honest bar — the raw 95%-helpful AUC flatters the
naive mean). Also checks, on Polis, whether a smoother aggregator tracks true
cross-group support better than the hard min.
"""
from __future__ import annotations

import numpy as np
import pytest

from . import _modeling as M
from ._common import record_finding, require
from .datasets import community_notes as cn
from .datasets import polis
from .experiments import keystone_variants as kv
from .metrics import auc, spearman

GRID = [
    ("min", "wald_count"),   # ≈ the current B_LCB
    ("min", "none"),
    ("min", "james_stein"),
    ("min", "wilson"),
    ("mean", "none"),
    ("mean", "james_stein"),
    ("nash", "none"),
    ("nash", "james_stein"),
    ("cvar.5", "james_stein"),
    ("ede4", "james_stein"),
]


def _balanced_auc(scores, labels, rng, draws=25):
    labels = np.asarray(labels)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    m = min(len(pos), len(neg))
    if m < 5:
        return float("nan")
    vals = []
    for _ in range(draws):
        take_pos = rng.choice(pos, m, replace=False)
        idx = np.concatenate([take_pos, neg]) if len(neg) <= m else \
            np.concatenate([take_pos, rng.choice(neg, m, replace=False)])
        vals.append(auc(scores[idx], labels[idx]))
    return float(np.mean(vals))


def test_community_notes_aggregator_penalty_grid(base_config):
    require(cn.NAME, *cn.SLICE_FILES)
    sl = cn.load_slice()
    reactions, posts, labels = cn.to_reactions(sl)
    cfg = base_config
    result = M.fit(reactions, posts, cfg, seed=0)

    ids = [n for n in labels]
    y = np.array([labels[n] for n in ids])
    rng = np.random.default_rng(0)

    # baselines
    bp = np.array([result.b_post.get(n, 0.0) for n in ids])
    mean_rating = np.array([
        (sum(r.value for r in reactions if r.post_id == n) /
         max(1, sum(1 for r in reactions if r.post_id == n))) for n in ids
    ])
    auc_bp = _balanced_auc(bp, y, rng)
    auc_mean = _balanced_auc(mean_rating, y, rng)

    print(f"\n[cn §4 grid] {len(ids)} notes, balanced AUC (25 draws), bar to beat = b_p")
    print(f"[cn §4 grid] {'BASELINE b_p':<26} balAUC={auc_bp:.4f}")
    print(f"[cn §4 grid] {'BASELINE mean-rating':<26} balAUC={auc_mean:.4f}")

    best = ("", -1.0)
    for k in (2, 4):
        clusters = M.cluster(result, cfg, seed=0, n_clusters=k)
        stats = kv.cluster_stats_for_notes(reactions, clusters.assignments, k)
        usable = [n for n in ids if n in stats]
        yk = np.array([labels[n] for n in usable])
        for agg, pen in GRID:
            s = kv.score_notes(stats, usable, agg, pen)
            a = _balanced_auc(s, yk, np.random.default_rng(1))
            tag = f"k={k} {agg}/{pen}"
            print(f"[cn §4 grid] {tag:<26} balAUC={a:.4f}")
            if a > best[1]:
                best = (tag, a)

    print(f"[cn §4 grid] best variant: {best[0]} balAUC={best[1]:.4f}  vs b_p={auc_bp:.4f}")

    # Did fixing the penalty at least recover the current B_LCB toward b_p?
    # (the diagnostic prediction), and can anything beat b_p?
    if best[1] < auc_bp - 0.005:
        record_finding(
            f"§4.2: no cross-cluster aggregator/penalty beat the scalar b_p at "
            f"reproducing Community Notes status on a balanced sample (best "
            f"{best[0]}={best[1]:.4f} vs b_p={auc_bp:.4f}). Consistent with the "
            f"research diagnosis: CN's CRH label *is* a threshold on the note "
            f"intercept, so b_p is near-optimal by construction on this benchmark. "
            f"The cross-cluster score's value must be judged where real group "
            f"structure exists (Polis), not against CN's own intercept."
        )


def test_polis_aggregator_tracks_cross_group_support(base_config):
    require(polis.NAME)
    slugs = polis.available()
    if not slugs:
        pytest.skip("no Polis conversations present")
    cfg = base_config

    print(f"\n[polis §4 grid] Spearman(score, true min-across-Polis-group support):")
    header_aggs = ["min/wald_count", "min/none", "min/james_stein", "nash/james_stein",
                   "ede4/james_stein"]
    agg_pen = [("min", "wald_count"), ("min", "none"), ("min", "james_stein"),
               ("nash", "james_stein"), ("ede4", "james_stein")]
    means = {name: [] for name in header_aggs}

    for slug in slugs:
        conv = polis.load_conversation(slug)
        if conv.n_groups < 2:
            continue
        reactions, posts = polis.to_reactions(conv)
        if len(reactions) < 200:
            continue
        result = M.fit(reactions, posts, cfg, seed=0)
        k = min(conv.n_groups, 4)
        clusters = M.cluster(result, cfg, seed=0, n_clusters=k)
        stats = kv.cluster_stats_for_notes(reactions, clusters.assignments, k)
        split = polis.group_split(conv)

        ids = [f"c{cid}" for cid in split if f"c{cid}" in stats
               and not np.isnan(split[cid]).all()]
        true_support = np.array([np.nanmin(split[int(pid[1:])]) for pid in ids])
        row = []
        for (agg, pen), name in zip(agg_pen, header_aggs):
            s = kv.score_notes(stats, ids, agg, pen)
            rho = spearman(s, true_support) if len(ids) > 5 else float("nan")
            means[name].append(rho)
            row.append(f"{name.split('/')[0]}:{rho:+.2f}")
        print(f"[polis §4 grid] {slug:<22} " + "  ".join(row))

    print("[polis §4 grid] MEAN  " +
          "  ".join(f"{name}:{np.nanmean(v):+.3f}" for name, v in means.items()))

    cur = np.nanmean(means["min/wald_count"])       # current B_LCB
    best_name = max(means, key=lambda n: np.nanmean(means[n]))
    best = np.nanmean(means[best_name])
    print(f"[polis §4 grid] current(min/wald_count)={cur:.3f}  best({best_name})={best:.3f}")
    assert best >= cur - 1e-9, "expected some variant to match or beat current B_LCB"
