"""CHORD — Cross-cluster Harmonized Optimization of Reception and Dissonance.

A bridging, attention-economy feed-ranking algorithm for federated social
networks. This package implements the whitepaper's valuation-and-allocation
layer: it values attention (which raters count, §5), values content (strength,
§4), and allocates scarce visibility (budget + constrained selection, §7–8).

One-line architecture (Appendix B): rank by tested cross-cluster support net of
weighted divisiveness; weight raters by quality-tracking not variance; price
authors by a strength-replenished conserved budget; audition the unproven from a
floored commons pool that doubles as the identifiability anchor; correct exposure
MNAR with doubly-robust propensity weighting; stabilize the coupled estimator as
two-timescale stochastic approximation held in a monitored bounded regime; and
expose M/rho/theta/epsilon as knobs while keeping all authority earned.
"""
from .config import ChordConfig, UserKnobs
from .types import (
    Exposure,
    ExposureSource,
    Post,
    Reaction,
    ReactionKind,
)
from .loop import Chord, WindowState

__version__ = "0.1.0"

__all__ = [
    "ChordConfig",
    "UserKnobs",
    "Post",
    "Reaction",
    "ReactionKind",
    "Exposure",
    "ExposureSource",
    "Chord",
    "WindowState",
    "__version__",
]
