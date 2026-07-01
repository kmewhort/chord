"""Scout precision q_scout (§5) — reward being early on eventual winners.

    q_scout(u) = sum_{p in P_u^+} e^{-alpha rank_t(u,p)} Phi_inf(p)
                 -----------------------------------------------------
                       sum_{p in P_u^+} e^{-alpha rank_t(u,p)}

where P_u^+ is the set of posts u reacted to positively, ``rank_t(u,p)`` is u's
temporal rank among positive reactors of p (0 = first), and ``Phi_inf(p)`` is the
post's *eventual* realized strength. Self-correcting: it is graded against future
outcomes, not current consensus. Being early (low rank) on posts that later score
high strength earns scout precision; being early on duds does not.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..config import ChordConfig
from ..types import Id, Post, Reaction


def compute_scout_precision(
    reactions: Sequence[Reaction],
    posts: Mapping[Id, Post],
    realized_strength: Mapping[Id, float],
    config: ChordConfig,
) -> Dict[Id, float]:
    """Per-user scout precision (§5).

    Parameters
    ----------
    reactions : all reaction events; only positive ones (``value > 0``) count as
        the rater "picking" the post.
    realized_strength : Phi_inf(p), the post's eventual strength (typically the
        final B_LCB or Phi). Posts absent from this map contribute nothing.
    """
    # Gather positive reactions per post, ordered by time, to assign ranks.
    pos_by_post: Dict[Id, List[Tuple[float, Id]]] = defaultdict(list)
    for rx in reactions:
        if rx.value > 0:
            pos_by_post[rx.post_id].append((rx.timestamp, rx.user_id))

    # For each post, rank its positive reactors by time (stable, ties share order).
    num: Dict[Id, float] = defaultdict(float)
    den: Dict[Id, float] = defaultdict(float)
    alpha = config.scout_alpha
    for pid, events in pos_by_post.items():
        phi = realized_strength.get(pid)
        if phi is None:
            continue
        events.sort(key=lambda e: e[0])
        for rank, (_, uid) in enumerate(events):
            decay = float(np.exp(-alpha * rank))
            num[uid] += decay * phi
            den[uid] += decay

    out: Dict[Id, float] = {}
    for uid in den:
        out[uid] = num[uid] / den[uid] if den[uid] > 0 else 0.0
    return out
