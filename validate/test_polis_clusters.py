"""§4 keystone on real deliberation: Polis clusters + B_LCB (Appendix C.1).

Polis conversations ship their own *validated opinion groups* (``group-id``),
computed by Polis's PCA + k-means over the vote matrix. That gives two honest
checks of CHORD §4:

1. **Cluster reconstruction (§4.2).** CHORD fits an opinion embedding ``x_u`` from
   the same agree/disagree votes and clusters it with the default Partition
   adapter. Do the recovered clusters match Polis's groups? Measured by Adjusted
   Rand Index (chance-corrected). Low ARI is a real finding about the embedding.

2. **B_LCB tracks cross-group support (§4.2).** A genuinely bridging comment earns
   agreement in *every* group; a divisive one splits them. We compute each
   comment's minimum mean-vote across Polis groups (its true cross-group support)
   and its group spread (true divisiveness), then ask whether CHORD's ``B_LCB``
   ranks by cross-group support and whether ``D(p)`` tracks the spread.

Metrics are printed per conversation and asserted on the aggregate so the test
measures the *systematic* effect rather than one noisy conversation.
"""
from __future__ import annotations

import numpy as np
import pytest

from . import _modeling as M
from ._common import require
from .datasets import polis
from .metrics import adjusted_rand_index, spearman


def _conversation_metrics(slug, cfg):
    conv = polis.load_conversation(slug)
    if conv.n_groups < 2:
        return None
    reactions, posts = polis.to_reactions(conv)
    if len(reactions) < 200:
        return None

    result = M.fit(reactions, posts, cfg, seed=0)

    # --- (1) cluster reconstruction vs Polis groups ---
    k = min(conv.n_groups, 4)
    clusters = M.cluster(reactions, result, cfg, n_clusters=k)
    shared = [p for p in conv.groups if f"p{p}" in clusters.assignments]
    ari = float("nan")
    if len(shared) >= 10:
        chord_lab = [clusters.assignments[f"p{p}"] for p in shared]
        polis_lab = [conv.groups[p] for p in shared]
        ari = adjusted_rand_index(chord_lab, polis_lab)

    # --- (2) B_LCB vs true cross-group support; D(p) vs true spread ---
    post_authors = {pid: post.author_id for pid, post in posts.items()}
    scores = M.bridging(reactions, result, clusters, post_authors, cfg)
    div = M.divisiveness_of(result, cfg)
    split = polis.group_split(conv)  # comment-id -> per-group mean vote

    blcb, cross_support, dvals, spreads = [], [], [], []
    for cid, gmeans in split.items():
        pid = f"c{cid}"
        if pid not in scores.b_lcb or np.isnan(gmeans).all():
            continue
        gm = gmeans[~np.isnan(gmeans)]
        if len(gm) < 2:
            continue
        blcb.append(scores.b_lcb[pid])
        cross_support.append(float(np.min(gm)))   # worst-group reception
        dvals.append(div.get(pid, np.nan))
        spreads.append(float(np.max(gm) - np.min(gm)))

    corr_support = spearman(np.array(blcb), np.array(cross_support)) if len(blcb) > 5 else float("nan")
    corr_div = spearman(np.array(dvals), np.array(spreads)) if len(dvals) > 5 else float("nan")
    return dict(slug=slug, n_groups=conv.n_groups, n_shared=len(shared),
                ari=ari, corr_support=corr_support, corr_div=corr_div, n_comments=len(blcb))


def test_polis_keystone_on_real_divides(base_config):
    require(polis.NAME)
    slugs = polis.available()
    if not slugs:
        pytest.skip("no Polis conversations present")

    cfg = base_config
    rows = [m for s in slugs if (m := _conversation_metrics(s, cfg)) is not None]
    assert rows, "no usable Polis conversations (need >=2 groups and enough votes)"

    print("\n[polis §4]  conversation            groups  ARI    corr(B_LCB,support)  corr(D,spread)")
    for r in rows:
        print(f"[polis §4]  {r['slug']:<24} {r['n_groups']:>4}   {r['ari']:+.3f}   "
              f"{r['corr_support']:+.3f}              {r['corr_div']:+.3f}")

    mean_ari = float(np.nanmean([r["ari"] for r in rows]))
    mean_support = float(np.nanmean([r["corr_support"] for r in rows]))
    mean_div = float(np.nanmean([r["corr_div"] for r in rows]))
    print(f"[polis §4]  MEAN  ARI={mean_ari:+.3f}  corr(B_LCB,support)={mean_support:+.3f}  "
          f"corr(D,spread)={mean_div:+.3f}")

    # Claim 1 (§4.2): the embedding recovers the real divide better than chance.
    assert mean_ari > 0.1, (
        f"FINDING: CHORD clusters barely match Polis groups (mean ARI={mean_ari:.3f}). "
        f"The §4.2 opinion embedding does not reconstruct real divides here."
    )
    # Claim 2 (§4.2): B_LCB ranks by genuine cross-group support.
    assert mean_support > 0.2, (
        f"FINDING: B_LCB does not track cross-group support on Polis "
        f"(mean Spearman={mean_support:.3f}). The bridged-support keystone is weak here."
    )
    # Claim 3 (§4.1): D(p) tracks the true group spread.
    assert mean_div > 0.0, (
        f"FINDING: divisiveness D(p) does not track real group spread "
        f"(mean Spearman={mean_div:.3f})."
    )
