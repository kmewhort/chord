"""Agent-based closed-loop simulator (Appendix C.4).

Exercises the dynamic components of §9 that no fixed dataset can — performative
stability, the concentration controller, the exploration anchor over time — and,
because the world is synthetic, the **ground-truth welfare** each ranker delivers.
Running several rankers (CHORD vs engagement/chronological/random/oracle) on the
same seeded world is the counterfactual that shows what bridging actually buys.
"""
from .population import Agent, Population, make_bipolar_population
from .content import (
    AuthorAgent, PostTruth, make_authors, true_loading, true_post, reset_truth,
)
from .response import react, expected_approval, reaction_pull
from .metrics import Welfare, embedding_recovery
from .rankers import (
    Ranker, ChordRanker, EngagementRanker, ChronologicalRanker, RandomRanker, OracleRanker,
)
from .engine import Simulator, SimulationResult, WindowMetrics

__all__ = [
    "Agent", "Population", "make_bipolar_population",
    "AuthorAgent", "PostTruth", "make_authors", "true_loading", "true_post", "reset_truth",
    "react", "expected_approval", "reaction_pull",
    "Welfare", "embedding_recovery",
    "Ranker", "ChordRanker", "EngagementRanker", "ChronologicalRanker",
    "RandomRanker", "OracleRanker",
    "Simulator", "SimulationResult", "WindowMetrics",
]
