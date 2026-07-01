"""Attention-economy mechanisms (§8).

The conserved, strength-replenished author visibility budget; the base-rate-
calibrated Thompson exploration pool; and (in :mod:`chord.rater.recycling`) the
anti-ossification influence recycling governor.
"""
from .budget import AuthorBudgetLedger
from .exploration import BetaPosterior, ExplorationPool

__all__ = ["AuthorBudgetLedger", "ExplorationPool", "BetaPosterior"]
