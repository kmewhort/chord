"""Bias-model calibration of organic reception against the ε-slice (§6/§13.2, E2).

For every post that receives ε-exposure, CHORD observes *both* confounded organic
reception and unconfounded randomized reception on the same content. The gap is a direct
estimate of the total confounding bias — including the unobserved part no propensity model
can touch (proximal-causal in spirit: the ε-slice is a negative-control exposure). So
instead of only *bounding* confounding (Rosenbaum) or capping it (the exploration anchor),
fit a **bias model** per cluster, `r_exp ≈ a_c + b_c · r_org`, and predict the unconfounded
reception from the confounded one everywhere — converting §13.2 from "bounded" to
"calibrated." The Coat MAR block validated this: the bias is structured (corr 0.5 with
level) and a model fit on ε-covered items transports to uncovered ones, beating IPW.

The ε-slice is scarce per window, so paired (organic, exploration) observations are
accumulated across windows with decay before the (weighted) least-squares fit; a cluster
with too little paired evidence returns the organic reception unchanged (no correction).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


class BiasCalibrator:
    """Rolling per-cluster linear bias model `r_exp ≈ a_c + b_c·r_org` from ε pairs."""

    def __init__(self, decay: float = 0.8, min_evidence: float = 8.0):
        self.decay = decay
        self.min_evidence = min_evidence
        # per cluster: weighted sufficient statistics [Sw, Swx, Swy, Swxx, Swxy]
        self._stat: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(5))

    def update(self, pairs: Sequence[Tuple[int, float, float, float]]) -> None:
        """Decay, then fold in (cluster, r_org, r_exp, weight) paired observations."""
        for c in self._stat:
            self._stat[c] *= self.decay
        for c, x, y, w in pairs:
            self._stat[c] += w * np.array([1.0, x, y, x * x, x * y])

    def predict(self, cluster: int, r_org: float) -> float:
        """Predicted unconfounded reception; identity if too little paired evidence."""
        s = self._stat.get(cluster)
        if s is None or s[0] < self.min_evidence:
            return r_org
        Sw, Swx, Swy, Swxx, Swxy = s
        det = Sw * Swxx - Swx * Swx
        if abs(det) < 1e-9:
            return r_org
        b = (Sw * Swxy - Swx * Swy) / det
        a = (Swy - b * Swx) / Sw
        return float(a + b * r_org)


def split_reception_by_source(reactions, weights, assignments, exposure_index,
                              exploration_source):
    """Per (post, cluster): ((n_org, r_org), (n_exp, r_exp)) split by exposure source."""
    org: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    exp: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r, w in zip(reactions, weights):
        c = assignments.get(r.user_id)
        if c is None:
            continue
        e = exposure_index.get((r.user_id, r.post_id))
        d = exp if (e is not None and e.source is exploration_source) else org
        d[r.post_id][c][0] += float(w)
        d[r.post_id][c][1] += float(w) * r.value
    return org, exp


def calibrated_reception(reception, org, exp, calibrator):
    """Return E2-calibrated reception {pid:{c:(n_cp, r_debiased)}} and the ε training pairs.

    Uses actual exploration reception where present (unconfounded ground truth), else the
    bias-model prediction from organic reception. ``reception`` supplies the evidence
    counts ``n_cp`` (unchanged); only the per-cluster mean is de-biased.
    """
    pairs: List[Tuple[int, float, float, float]] = []
    out: Dict = {}
    for pid, rec in reception.items():
        cal = {}
        for c, (n_cp, _mean) in rec.items():
            ow, ox = org.get(pid, {}).get(c, [0.0, 0.0])
            ew, ex = exp.get(pid, {}).get(c, [0.0, 0.0])
            r_org = (ox / ow) if ow > 0 else _mean
            if ew > 0:
                r_exp = ex / ew
                if ow > 0:
                    pairs.append((c, r_org, r_exp, ew))
                debiased = r_exp                       # unconfounded ground truth
            else:
                debiased = calibrator.predict(c, r_org)  # predict from confounded
            cal[c] = (n_cp, debiased)
        out[pid] = cal
    return out, pairs
