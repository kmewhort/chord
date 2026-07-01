"""Retroactive evaluation harnesses (Appendix C).

The semi-synthetic MNAR / propensity harness (C.3): the only component here that
can validate the debiasing layer, because it gives us control of ground truth.
"""
from .mnar_harness import (
    FitDiagnostics,
    SyntheticWorld,
    logging_policy_exposures,
    make_world,
    run_experiment,
)

__all__ = [
    "SyntheticWorld",
    "make_world",
    "logging_policy_exposures",
    "run_experiment",
    "FitDiagnostics",
]
