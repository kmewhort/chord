"""Identifiability and the propensity model (§6).

The counterfactual learning-to-rank layer that makes the keystone identifiable:
inverse-propensity weighting (§6.2), the pluggable propensity menu (§6.3), and the
doubly-robust wrapper that is consistent if *either* the propensity or the
imputation is right (§6.3a). The exploration pool's known epsilon-exposure is the
unconfounded anchor across all options.
"""
from .base import PropensityModel
from .models import (
    LoggedPropensityModel,
    PolicyDerivedModel,
    PositionBasedModel,
    UniformExplorationModel,
)
from .ipw import compute_ipw_weights, snipw_estimate
from .doubly_robust import doubly_robust_mean, doubly_robust_reception

__all__ = [
    "PropensityModel",
    "UniformExplorationModel",
    "PositionBasedModel",
    "PolicyDerivedModel",
    "LoggedPropensityModel",
    "compute_ipw_weights",
    "snipw_estimate",
    "doubly_robust_reception",
    "doubly_robust_mean",
]
