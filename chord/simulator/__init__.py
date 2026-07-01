"""Agent-based closed-loop simulator (Appendix C.4).

Exercises the dynamic components of §9 that no fixed dataset can: performative
stability, the concentration controller, and the exploration anchor over time.
"""
from .population import Agent, Population, make_bipolar_population
from .content import AuthorAgent, make_authors, true_loading, reset_truth
from .response import react
from .engine import Simulator, SimulationResult, WindowMetrics

__all__ = [
    "Agent",
    "Population",
    "make_bipolar_population",
    "AuthorAgent",
    "make_authors",
    "true_loading",
    "reset_truth",
    "react",
    "Simulator",
    "SimulationResult",
    "WindowMetrics",
]
