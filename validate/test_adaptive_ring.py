"""Adaptive red-team: can a ring evade the loyalty defense on real CN data? (§13.10)

A defense that only beats a naive attacker is weak. The loyalty defense fires on a
*super-loyal* bloc (accounts approving nearly every one of a target's notes) that is
*opinion-dispersed*. The obvious evasion is **partial approval**: each puppet approves
only a fraction f of the target's notes, staying under the super-loyal cutoff.

We sweep f on the real Community Notes slice and check the attacker's *best* outcome
under the defense. The result is a genuine tension: at high f the ring inflates but the
detector fires and the defense removes it; at low f the detector is evaded but the
attack no longer inflates (scattered partial co-approval doesn't manufacture consistent
cross-cluster support). So there is no evasion sweet spot — the ring never beats its
honest baseline under the defense.
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
FRACTIONS = (1.0, 0.7, 0.5, 0.3)


def _cfg():
    return ChordConfig(d=2, mf_iters=60, reg_embedding=0.08, reg_bias_post=0.05,
                       n_clusters=2, affective_weighting=False)


def test_loyalty_defense_is_adaptively_robust():
    require(cn.NAME, *cn.SLICE_FILES)
    sl = cn.load_slice()
    reactions, posts, labels = cn.to_reactions(sl)
    post_authors = {pid: p.author_id for pid, p in posts.items()}
    cfg = _cfg()

    by = sl.notes.groupby("author").agg(n=("noteId", "size"), h=("statusBinary", "mean"))
    cand = by[(by.n >= 3) & (by.h < 0.4)].sort_values("n", ascending=False)
    if cand.empty:
        pytest.skip("no mediocre multi-note author")
    target = cand.index[0]
    tnotes = list(sl.notes[sl.notes.author == target].noteId)

    res0 = M.fit(reactions, posts, cfg, seed=0)
    cl0 = M.cluster(res0, cfg, seed=0)
    base = float(np.nanmean([M.bridging(res0, cl0, post_authors, cfg).b_lcb.get(p, np.nan)
                             for p in tnotes]))
    hosts = defaultdict(list)
    for u, c in cl0.assignments.items():
        hosts[c].append(u)
    host_rx = defaultdict(list)
    for r in reactions:
        host_rx[r.user_id].append(r)

    def defended_blcb(frac):
        rng = np.random.default_rng(0)
        aug = list(reactions)
        for i in range(RING_SIZE):
            c = i % 2
            h = hosts[c][rng.integers(len(hosts[c]))]
            sid = f"S{i}"
            for r in host_rx[h][:15]:
                aug.append(Reaction(sid, r.post_id, r.value))
            m = max(1, int(round(frac * len(tnotes))))
            for j in rng.choice(len(tnotes), m, replace=False):   # adaptive: partial approval
                aug.append(Reaction(sid, tnotes[j], 1.0))
        res = M.fit(aug, posts, cfg, seed=0)
        clu = M.cluster(res, cfg, seed=0)
        ec = defaultdict(lambda: defaultdict(float))
        for r in aug:
            cc = clu.assignments.get(r.user_id)
            if cc is not None:
                ec[r.post_id][cc] += 1.0
        sc = M.bridging(res, clu, post_authors, cfg,
                        exposure_counts={p: dict(d) for p, d in ec.items()})
        ct = CollusionTracker()
        ct.update(aug, post_authors)
        det = ct.manufactured_fraction(target, clu.assignments, clu.n_clusters)
        blcb = float(np.nanmean([sc.b_lcb.get(p, np.nan) for p in tnotes]))
        return blcb - LOYALTY_PENALTY * det

    best = max(defended_blcb(f) for f in FRACTIONS)   # the attacker's best evasion
    print(f"\n[cn adaptive] baseline B_LCB={base:.3f}; attacker's best defended B_LCB "
          f"over f in {FRACTIONS} = {best:.3f}")
    # Under the defense, no partial-approval strategy lets the ring beat its honest
    # baseline: evasion (low f) kills the attack, effectiveness (high f) is detected.
    assert best <= base + 0.05, (
        f"an adaptive partial-approval ring beat its baseline under the defense "
        f"({base:.3f} -> {best:.3f}) — the loyalty defense is evadable"
    )
