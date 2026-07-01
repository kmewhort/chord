"""Value model and feed assembly (§7).

Personalized value V(u,p) (§7.1), the factor vector (§7.3), and the greedy
submodular constrained selection that realizes the feed (§7.2).
"""
from .value import (
    FactorContext,
    FactorFn,
    DEFAULT_FACTORS,
    blended_value,
    bridge_factor,
    value,
)
from .assembly import Candidate, AssemblyResult, greedy_assemble

__all__ = [
    "value",
    "FactorContext",
    "FactorFn",
    "DEFAULT_FACTORS",
    "blended_value",
    "bridge_factor",
    "Candidate",
    "AssemblyResult",
    "greedy_assemble",
]
