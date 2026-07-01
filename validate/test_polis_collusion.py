"""Dive #2 on Polis: dense deliberation is intrinsically ring-resistant (§13.10).

Cross-checking the camouflaged content-boost ring on Polis surfaced an instructive
*difference* from Community Notes rather than a copy of it. Polis voting is **dense**
(a comment collects hundreds of votes from a large fraction of participants), so a
distributed ring is a small minority of a divisive comment's genuine support: to fake
cross-cluster agreement it would need a ring larger than the genuine *opposing* group.

Concretely, even a ring of 100 sybils (~5% of the voters) lifts a genuinely divisive
comment's B_LCB only slightly, and nowhere near the median genuinely-consensus
comment — the ring cannot crown its pick. Dense engagement is therefore itself a
strong defense; the content-boost ring (and the loyalty penalty that counters it,
`test_community_notes_collusion.py`) is a *sparse*-data phenomenon. The loyalty
signal correspondingly dilutes here (the ring is a small fraction of dense support) —
which is the right behaviour: on dense data there is little to defend against.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from chord.config import ChordConfig
from chord.types import Reaction

from . import _modeling as M
from ._common import require
from .datasets import polis

RING_SIZE = 100
CONVERSATION = "vtaiwan.uberx"


def _cfg():
    return ChordConfig(d=2, mf_iters=40, reg_embedding=0.08, reg_bias_post=0.05,
                       n_clusters=2, affective_weighting=False)


def _score(reactions, posts, post_authors, cfg):
    res = M.fit(reactions, posts, cfg, seed=0)
    clusters = M.cluster(reactions, res, cfg)
    return M.bridging(reactions, res, clusters, post_authors, cfg), res, clusters


def test_dense_polis_resists_the_content_boost_ring():
    require(polis.NAME)
    if CONVERSATION not in polis.available():
        pytest.skip(f"{CONVERSATION} not present")
    conv = polis.load_conversation(CONVERSATION)
    reactions, posts = polis.to_reactions(conv)
    post_authors = {pid: p.author_id for pid, p in posts.items()}
    cfg = _cfg()

    sc, res, cl = _score(reactions, posts, post_authors, cfg)
    author_posts = defaultdict(list)
    for pid, a in post_authors.items():
        author_posts[a].append(pid)
    counts = Counter(post_authors[pid] for pid in posts)
    multi = [a for a, n in counts.items() if n >= 3 and a != "auth?"]
    if not multi:
        pytest.skip("no multi-comment author")

    # most divisive multi-comment author = lowest mean min-over-cluster reception
    def minclu(a):
        pc = [sc.per_cluster[p] for p in author_posts[a] if p in sc.per_cluster]
        return float(np.mean([x.min() for x in pc])) if pc else 1.0
    target = min(multi, key=minclu)
    target_posts = author_posts[target]
    base = float(np.nanmean([sc.b_lcb.get(p, np.nan) for p in target_posts]))

    genuine = [v for p, v in sc.b_lcb.items()
               if p not in set(target_posts) and np.isfinite(v)]
    median_genuine = float(np.median(genuine))

    # inject a large camouflaged ring across the opinion clusters
    hosts = defaultdict(list)
    for u, c in cl.assignments.items():
        hosts[c].append(u)
    host_rx = defaultdict(list)
    for r in reactions:
        host_rx[r.user_id].append(r)
    rng = np.random.default_rng(0)
    aug = list(reactions)
    for i in range(RING_SIZE):
        c = i % cl.n_clusters
        if not hosts.get(c):
            continue
        host = hosts[c][rng.integers(len(hosts[c]))]
        for r in host_rx[host][:25]:
            aug.append(Reaction(f"SYBIL{i}", r.post_id, r.value))
        for tp in target_posts:
            aug.append(Reaction(f"SYBIL{i}", tp, 1.0))

    sc2, _, _ = _score(aug, posts, post_authors, cfg)
    ring = float(np.nanmean([sc2.b_lcb.get(p, np.nan) for p in target_posts]))
    n_voters = len({r.user_id for r in reactions})

    print(f"\n[polis collusion] {CONVERSATION}: divisive target B_LCB baseline={base:.3f} "
          f"-> ring(K={RING_SIZE}, {100*RING_SIZE/n_voters:.0f}% of voters)={ring:.3f}; "
          f"median genuine comment={median_genuine:.3f}")

    # Dense engagement resists the ring: even a big ring cannot lift its divisive pick
    # anywhere near a genuinely-consensus comment.
    assert ring < median_genuine, (
        f"a K={RING_SIZE} ring lifted the target to {ring:.3f}, at/above the median "
        f"genuine comment {median_genuine:.3f} — dense deliberation did not resist it"
    )
