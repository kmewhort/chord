"""Influence recycling (§8) — the anti-ossification governor.

Damp the consistently-satisfied, boost the under-served, so the system listens
hardest to whoever it serves worst:

    lambda_eff_u = lambda_u * (1 + zeta * (S_bar - S_bar(u)))

with ``S_bar(u)`` the user's model-estimated realized value over what they were
shown (hard to fake by "acting dissatisfied", since it is the model's estimate,
not self-report) and ``S_bar`` the population mean. This is the governor that
keeps §5's rater-weighting from calcifying into a taste aristocracy — the one
mechanism most ranking systems lack.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from ..config import ChordConfig
from ..types import Id


def apply_recycling(
    rater_lambda: Mapping[Id, float],
    realized_satisfaction: Mapping[Id, float],
    config: ChordConfig,
) -> Dict[Id, float]:
    """Compute lambda_eff (§8).

    Parameters
    ----------
    rater_lambda : base rater influence lambda_u (§5).
    realized_satisfaction : S_bar(u), the model-estimated realized value over
        what user u was shown. Users absent from this map are treated as
        exactly average (no adjustment).
    """
    if not realized_satisfaction:
        return dict(rater_lambda)
    vals = np.array(list(realized_satisfaction.values()), dtype=float)
    S_bar = float(vals.mean())
    zeta = config.recycling_zeta

    out: Dict[Id, float] = {}
    for u, lam in rater_lambda.items():
        s_u = realized_satisfaction.get(u, S_bar)
        factor = 1.0 + zeta * (S_bar - s_u)
        # Keep the multiplier non-negative: an extremely well-served user is
        # damped but never flipped to negative influence.
        factor = max(0.0, factor)
        out[u] = lam * factor

    # renormalize to preserve total influence mass
    total = sum(out.values())
    base_total = sum(rater_lambda.values())
    if total > 0 and base_total > 0:
        out = {u: w * base_total / total for u, w in out.items()}
    return out
