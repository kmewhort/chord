"""Rater weighting (§5) and influence recycling (§8).

Quality-tracking, not variance: discriminating cross-cutting raters dominate the
estimate; indiscriminate scrollers barely move it. Two earned quantities feed off
the same geometry — scout precision and the recursive cross-divide credibility
lambda — and influence recycling keeps them from calcifying.
"""
from .eigentrust import (
    build_trust_matrix,
    compute_lambda,
    eigentrust,
    outgoing_diversity_weights,
)
from .quality import blend_lambda, quality_tracking_weight
from .scout import compute_scout_precision
from .recycling import apply_recycling

__all__ = [
    "build_trust_matrix",
    "eigentrust",
    "outgoing_diversity_weights",
    "compute_lambda",
    "quality_tracking_weight",
    "blend_lambda",
    "compute_scout_precision",
    "apply_recycling",
]
