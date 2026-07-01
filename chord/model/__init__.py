"""The relation model (§4) — the scoring keystone.

Weighted biased matrix factorization (§4.1), the divide-weighting metric and
divisiveness D(p) (§4.1), and the per-cluster B_LCB tested bridged support (§4.2).
"""
from .factorization import FactorizationResult, MatrixFactorization
from .divisiveness import DivisivenessModel, fit_divisiveness
from .bridging import BridgingScorer, BridgingScores, ClusterModel
from .coordination import CollusionTracker, coordination_scores

__all__ = [
    "MatrixFactorization",
    "FactorizationResult",
    "DivisivenessModel",
    "fit_divisiveness",
    "BridgingScorer",
    "BridgingScores",
    "ClusterModel",
    "coordination_scores",
    "CollusionTracker",
]
