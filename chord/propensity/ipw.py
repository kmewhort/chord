"""Inverse-propensity weighting for the observation weights omega_up (§6.2).

Reactions are observed only where a user was *shown* a post, and every historical
policy shows people mostly in-group content — the negatives are missing, not
negative, and the missingness depends on the alignment we are estimating (MNAR).
Weighting each observation by inverse propensity ``1/pi_up`` yields an unbiased
objective provided propensities are accurate and non-zero for every relevant pair
[Joachims–Swaminathan–Schnabel 2017; Schnabel et al. 2016].

The per-observation weight (§6.2):

    omega_up = lambda_u * min(1/pi_hat_up, W_max) * s_up

* ``lambda_u`` — the rater's quality-tracking influence (§5).
* ``min(1/pi, W_max)`` — the clip. Tie ``W_max = 1/epsilon`` to the exploration
  floor (§6.2): with pi >= epsilon guaranteed for audited items this is a natural
  ceiling on inverse weights and hence on gradient variance. This is a
  variance-for-bias trade — clipping slightly under-corrects the hardest deep-MNAR
  pairs, which we adopt knowingly.
* ``s_up`` — a per-observation reliability factor (default 1). The silent-
  disagreement handling (exposed-but-no-reaction as weak negative) lives in the
  reaction *value* itself (``-c``), not here.

Self-normalization (SNIPW): the MF loss is a weighted *average* (it divides by
sum of omega), which is exactly the self-normalized estimator that controls IPW
variance (§6.2).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import ChordConfig
from ..types import Exposure, Id, Reaction
from .base import PropensityModel


def compute_ipw_weights(
    reactions: Sequence[Reaction],
    propensity_model: PropensityModel,
    config: ChordConfig,
    rater_lambda: Optional[Mapping[Id, float]] = None,
    exposures: Optional[Mapping[Tuple[Id, Id], Exposure]] = None,
    reliability: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute omega_up for each reaction (§6.2).

    Parameters
    ----------
    reactions : the observation set E.
    propensity_model : any §6.3 estimator.
    rater_lambda : per-user influence lambda_u (§5). Missing users default to a
        small floor so brand-new raters contribute little (Sybil starvation, §5).
    exposures : optional map (user, post) -> Exposure providing slot/source and
        any logged propensity to the model.
    reliability : optional per-observation s_up (default all 1).
    normalize : rescale the returned weights to mean 1. This is on by default and
        matters: ``rater_lambda`` is typically a normalized distribution (sums to
        1), so raw weights would be O(1/n) and the *fixed* embedding
        regularization in the MF would swamp the data term and collapse the
        embeddings. SNIPW / the weighted-average loss is invariant to a global
        weight scale, so mean-1 normalization preserves every estimate while
        keeping the ALS solve well-conditioned.
    """
    n = len(reactions)
    W_max = config.W_max
    # A floor for unknown raters: the eigentrust teleport floor scaled small.
    lam_floor = (1.0 - config.eigentrust_delta) / max(1, n)

    weights = np.empty(n, dtype=float)
    for i, rx in enumerate(reactions):
        exp = None
        if exposures is not None:
            exp = exposures.get((rx.user_id, rx.post_id))
        pi = propensity_model.propensity(rx.user_id, rx.post_id, exp)
        pi = max(pi, 1e-12)
        inv = min(1.0 / pi, W_max)
        lam = lam_floor if rater_lambda is None else rater_lambda.get(rx.user_id, lam_floor)
        s = 1.0 if reliability is None else float(reliability[i])
        weights[i] = lam * inv * s

    if normalize:
        total = weights.sum()
        if total > 0:
            weights = weights * (n / total)  # mean weight -> 1
    return weights


def snipw_estimate(values: Sequence[float], weights: Sequence[float]) -> float:
    """Self-normalized inverse-propensity estimate (§6.2).

    hat = sum_i w_i v_i / sum_i w_i. Controls the variance of vanilla IPW at the
    cost of a small bias, and is the form embedded in the MF's weighted loss.
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    denom = w.sum()
    if denom <= 0:
        return float("nan")
    return float(np.dot(w, v) / denom)
