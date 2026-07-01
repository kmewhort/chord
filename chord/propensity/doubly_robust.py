"""Doubly-robust estimation (§6.3a) — the default posture.

Pair *any* propensity estimate with a reception imputation model; the estimator
is consistent if *either* the propensity *or* the imputation is right
[Dudík et al. 2014; Saito 2020]. This is the safest default and should wrap
whatever else is chosen.

The DR estimate of a target functional over a (user, post) population is

    hat_DR = mean_p [ imputed(p) ] + IPW-correction on the observed residuals

Here we expose the core building blocks: a doubly-robust *reception estimate* for
a single (cluster/user, post) reception used by the bridging reconstruction, and
a DR mean over a population. The imputation model is any callable
``impute(user_id, post_id) -> r_hat`` — in CHORD this is the factorization's own
reconstruction, so DR uses the model to fill the counterfactual outgroup
reactions the propensity layer cannot observe.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import ChordConfig
from ..types import Exposure, Id, Reaction
from .base import PropensityModel


ImputeFn = Callable[[Id, Id], float]


def doubly_robust_reception(
    user_ids: Sequence[Id],
    post_id: Id,
    observed: Mapping[Id, float],
    impute: ImputeFn,
    propensity_model: PropensityModel,
    config: ChordConfig,
    exposures: Optional[Mapping[Tuple[Id, Id], Exposure]] = None,
) -> float:
    """DR estimate of a post's mean reception over a target user population.

    hat_DR = (1/|U|) sum_u [ imp_u + 1{exposed} * clip(1/pi) * (r_u - imp_u) ]

    If everyone is exposed and propensities are exact, this reduces to the IPW
    mean; if no one is exposed it falls back to the pure imputation — which is
    exactly the graceful degradation the DR wrapper buys (§6.3a).
    """
    if len(user_ids) == 0:
        return float("nan")
    W_max = config.W_max
    total = 0.0
    for u in user_ids:
        imp = impute(u, post_id)
        r = observed.get(u)
        if r is None:
            total += imp  # counterfactual: rely on imputation
        else:
            exp = None if exposures is None else exposures.get((u, post_id))
            pi = max(propensity_model.propensity(u, post_id, exp), 1e-12)
            inv = min(1.0 / pi, W_max)
            total += imp + inv * (r - imp)
    return float(total / len(user_ids))


def doubly_robust_mean(
    pairs: Sequence[Tuple[Id, Id]],
    observed: Mapping[Tuple[Id, Id], float],
    impute: ImputeFn,
    propensity_model: PropensityModel,
    config: ChordConfig,
    exposures: Optional[Mapping[Tuple[Id, Id], Exposure]] = None,
) -> float:
    """Doubly-robust mean of a target functional over arbitrary (user, post) pairs."""
    if not pairs:
        return float("nan")
    W_max = config.W_max
    total = 0.0
    for (u, p) in pairs:
        imp = impute(u, p)
        r = observed.get((u, p))
        if r is None:
            total += imp
        else:
            exp = None if exposures is None else exposures.get((u, p))
            pi = max(propensity_model.propensity(u, p, exp), 1e-12)
            inv = min(1.0 / pi, W_max)
            total += imp + inv * (r - imp)
    return float(total / len(pairs))
