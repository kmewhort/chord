"""The relation model (§4) — the scoring keystone.

Weighted biased matrix factorization (§4.1), the divide-weighting metric and
divisiveness D(p) (§4.1), and the per-cluster B_LCB tested bridged support (§4.2).
"""
from .factorization import FactorizationResult, MatrixFactorization
from .divisiveness import DivisivenessModel, fit_divisiveness
from .bridging import BridgingScorer, BridgingScores, ClusterModel, cluster_reception
from .calibration import (
    BiasCalibrator,
    calibrated_reception,
    split_reception_by_source,
)
from .depth import estimate_depth
from .priors import AuthorClusterReception, hierarchical_priors
from .spectral import spectral_opinion_clusters
from .coordination import CollusionTracker, coordination_scores
from .reception_anchor import ExplorationAnchor

__all__ = [
    "MatrixFactorization",
    "FactorizationResult",
    "DivisivenessModel",
    "fit_divisiveness",
    "BridgingScorer",
    "cluster_reception",
    "estimate_depth",
    "AuthorClusterReception",
    "hierarchical_priors",
    "BiasCalibrator",
    "calibrated_reception",
    "split_reception_by_source",
    "spectral_opinion_clusters",
    "BridgingScores",
    "ClusterModel",
    "coordination_scores",
    "CollusionTracker",
    "ExplorationAnchor",
]
