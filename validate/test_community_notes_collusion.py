"""Dive #2 on real data: distributed sybil ring + loyalty defense on Community Notes.

The distributed-ring attack and its cluster-spread-gated loyalty defense were found
and fixed in the simulator (§13.10). This checks both on *real* rater×note data: pick
a genuinely mediocre real author (several not-helpful notes), inject a distributed
ring of camouflaged sybils — each copies a real cluster-member's ratings to embed in
that opinion cluster, then boosts every one of the target's notes — and confirm

1. the ring inflates the target's B_LCB (the attack transfers from sim to reality), and
2. the loyalty defense detects it (a large opinion-dispersed super-loyal bloc) and
   removes the inflation.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from chord.config import ChordConfig
from chord.model import CollusionTracker
from chord.types import Reaction

from . import _modeling as M
from ._common import require
from .datasets import community_notes as cn

RING_SIZE = 40
LOYALTY_PENALTY = 3.0


def _cfg():
    return ChordConfig(d=2, mf_iters=60, reg_embedding=0.08, reg_bias_post=0.05,
                       n_clusters=2, affective_weighting=False)


def _target_blcb(reactions, posts, post_authors, target, target_notes, cfg,
                 defense=False):
    res = M.fit(reactions, posts, cfg, seed=0)
    clusters = M.cluster(reactions, res, cfg)
    sc = M.bridging(reactions, res, clusters, post_authors, cfg)
    vals = np.array([sc.b_lcb.get(pid, np.nan) for pid in target_notes])
    frac = 0.0
    if defense:
        ct = CollusionTracker()
        ct.update(reactions, post_authors)
        frac = ct.manufactured_fraction(target, opinion_coord=clusters.opinion_coord)
        vals = vals - LOYALTY_PENALTY * frac
    return float(np.nanmean(vals)), frac


def test_ring_attack_and_loyalty_defense_on_community_notes():
    require(cn.NAME, *cn.SLICE_FILES)
    sl = cn.load_slice()
    reactions, posts, labels = cn.to_reactions(sl)
    post_authors = {pid: p.author_id for pid, p in posts.items()}
    cfg = _cfg()

    # a genuinely mediocre real author: >=3 notes, mostly NOT helpful
    by_auth = sl.notes.groupby("author").agg(n=("noteId", "size"),
                                             helpful=("statusBinary", "mean"))
    cand = by_auth[(by_auth.n >= 3) & (by_auth.helpful < 0.4)].sort_values("n", ascending=False)
    if cand.empty:
        pytest.skip("no mediocre multi-note author in this slice")
    target = cand.index[0]
    target_notes = list(sl.notes[sl.notes.author == target].noteId)

    base, _ = _target_blcb(reactions, posts, post_authors, target, target_notes, cfg)

    # place camouflaged sybils across the real opinion clusters
    res0 = M.fit(reactions, posts, cfg, seed=0)
    cl0 = M.cluster(reactions, res0, cfg)
    hosts: dict = defaultdict(list)
    for u, c in cl0.assignments.items():
        hosts[c].append(u)
    host_rx: dict = defaultdict(list)
    for r in reactions:
        host_rx[r.user_id].append(r)

    rng = np.random.default_rng(0)
    aug = list(reactions)
    for i in range(RING_SIZE):
        c = i % cl0.n_clusters
        host = hosts[c][rng.integers(len(hosts[c]))]
        sid = f"SYBIL{i}"
        for r in host_rx[host][:15]:          # camouflage: embed in the host's cluster
            aug.append(Reaction(sid, r.post_id, r.value))
        for tn in target_notes:               # the attack: boost every target note
            aug.append(Reaction(sid, tn, 1.0))

    ring, _ = _target_blcb(aug, posts, post_authors, target, target_notes, cfg)
    defended, frac = _target_blcb(aug, posts, post_authors, target, target_notes, cfg,
                                  defense=True)

    print(f"\n[cn collusion] target B_LCB  baseline={base:.3f}  ring(K={RING_SIZE})={ring:.3f}  "
          f"ring+defense={defended:.3f}  (manufactured_fraction={frac:.3f})")

    # 1. the attack transfers to real data: the ring inflates the target's bridged support
    assert ring > base + 0.2, (
        f"ring should inflate the target's B_LCB on real data ({base:.3f} -> {ring:.3f})"
    )
    # 2. the loyalty defense detects the dispersed super-loyal bloc
    assert frac > 0.5, f"loyalty detector should fire on the ring (fraction={frac:.3f})"
    # 3. and removes the inflation (back to at most the honest baseline)
    assert defended < ring and defended <= base + 1e-6, (
        f"loyalty defense should remove the inflation ({ring:.3f} -> {defended:.3f}, "
        f"baseline {base:.3f})"
    )
