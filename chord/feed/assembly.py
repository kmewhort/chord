"""Feed as constrained selection (§7.2) — greedy submodular top-k.

Per viewer, choose N slots to maximize value under constraints:

    max_{S:|S|=N} sum_{p in S} [ (1-eps) sum_f theta_f V_f(u,p) + eps * Phi_tilde(p) ]
                               * posdisc(p)

subject to
  * per-author cap and budget  sum_{p in S, a(p)=a} E(p) <= B(a)   (§8)
  * diverse-*approval* coverage (submodular — diminishing returns for re-covering
    an already-covered region of opinion space)
  * exploration floor >= eps * N

Greedy gives the 1 - 1/e guarantee; run it over the top few hundred candidates,
not the corpus. Crucially the coverage constraint rewards *diverse approval*, not
diverse exposure: content that *earns* cross-cluster support, never forced
exposure (§7.2 note; indiscriminate outgroup exposure can increase polarization).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..types import Id


@dataclass
class Candidate:
    """A scored candidate for the feed assembler."""

    post_id: Id
    author_id: Id
    base_value: float                    # sum_f theta_f V_f(u,p)
    exploration_value: float = 0.0       # Phi_tilde(p): audition score for new items
    approval_coverage: Optional[np.ndarray] = None  # per-cluster earned approval (>=0)
    exposure_cost: float = 1.0           # E(p) charged against the author budget
    posdisc: float = 1.0                 # positive-discrimination multiplier
    is_exploration: bool = False         # eligible to satisfy the exploration floor


@dataclass
class AssemblyResult:
    selected: List[Id] = field(default_factory=list)
    order: List[Candidate] = field(default_factory=list)
    objective: float = 0.0
    exploration_count: int = 0


def _coverage_value(mass: np.ndarray) -> float:
    """Concave, submodular coverage of accumulated per-region approval mass.

    Using sqrt gives strictly diminishing returns: re-covering an already-covered
    opinion region yields less marginal gain than covering a fresh one.
    """
    return float(np.sqrt(np.maximum(mass, 0.0)).sum())


def greedy_assemble(
    candidates: Sequence[Candidate],
    n_slots: int,
    epsilon: float,
    author_budgets: Optional[Mapping[Id, float]] = None,
    coverage_weight: float = 1.0,
    n_clusters: Optional[int] = None,
) -> AssemblyResult:
    """Greedy submodular constrained selection (§7.2).

    Parameters
    ----------
    candidates : scored candidates (typically the top few hundred).
    n_slots : N, the number of attention slots to fill.
    epsilon : exploration appetite; the objective mixes value with the audition
        score and the floor reserves ceil(eps*N) slots for exploration items.
    author_budgets : per-author remaining budget B(a) (§8). Selecting a post
        charges ``exposure_cost`` against its author's budget; a post that would
        overflow the budget is skipped. Authors absent from the map are
        unconstrained.
    coverage_weight : multiplier on the submodular diverse-approval bonus.
    """
    if n_slots <= 0 or not candidates:
        return AssemblyResult()
    eps = max(0.0, min(1.0, epsilon))
    floor = ceil(eps * n_slots)

    # Determine coverage dimensionality.
    if n_clusters is None:
        for c in candidates:
            if c.approval_coverage is not None:
                n_clusters = len(c.approval_coverage)
                break
    cov_mass = np.zeros(n_clusters) if n_clusters else None

    remaining = {c.post_id: c for c in candidates}
    budgets: Dict[Id, float] = dict(author_budgets) if author_budgets else {}
    spent: Dict[Id, float] = defaultdict(float)

    result = AssemblyResult()

    def point_score(c: Candidate) -> float:
        return ((1.0 - eps) * c.base_value + eps * c.exploration_value) * c.posdisc

    def marginal(c: Candidate) -> float:
        base = point_score(c)
        if cov_mass is not None and c.approval_coverage is not None and coverage_weight:
            gain = _coverage_value(cov_mass + c.approval_coverage) - _coverage_value(cov_mass)
            base += coverage_weight * gain
        return base

    def budget_ok(c: Candidate) -> bool:
        if c.author_id not in budgets:
            return True
        return spent[c.author_id] + c.exposure_cost <= budgets[c.author_id] + 1e-9

    for slot in range(n_slots):
        slots_left = n_slots - slot
        need_exploration = (floor - result.exploration_count) >= slots_left

        best_c = None
        best_gain = -np.inf
        for c in remaining.values():
            if not budget_ok(c):
                continue
            if need_exploration and not c.is_exploration:
                continue
            g = marginal(c)
            if g > best_gain:
                best_gain = g
                best_c = c

        if best_c is None:
            # No budget-feasible candidate (possibly the exploration floor cannot
            # be met); relax the exploration requirement and retry once.
            for c in remaining.values():
                if not budget_ok(c):
                    continue
                g = marginal(c)
                if g > best_gain:
                    best_gain = g
                    best_c = c
            if best_c is None:
                break

        # commit the pick
        result.selected.append(best_c.post_id)
        result.order.append(best_c)
        result.objective += best_gain
        if best_c.is_exploration:
            result.exploration_count += 1
        if cov_mass is not None and best_c.approval_coverage is not None:
            cov_mass = cov_mass + best_c.approval_coverage
        spent[best_c.author_id] += best_c.exposure_cost
        del remaining[best_c.post_id]

    return result
