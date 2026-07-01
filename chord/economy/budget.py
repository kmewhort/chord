"""Author visibility budget — conserved + earned (§8).

Per author, per window, exposure is capped and replenished by realized strength:

    sum_{p: a(p)=a, p in W} E(p) <= B(a)
    B_{t+1}(a) = B_0 + eta * sum_{p: a(p)=a, p in W_t} Phi(p) E(p)

Firehose posting spreads a fixed budget thin; quality regenerates it. This
inverts the engagement logic (each post an independent virality lottery ticket,
so volume is rational) into one where posting more *dilutes you* unless it earns.
The budget binds to the **identity** port, not the raw account, so it cannot be
sharded across sockpuppets (§8, §10).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Mapping

from ..config import ChordConfig
from ..types import Id


@dataclass
class AuthorBudgetLedger:
    """Per-identity visibility budgets (§8).

    Keys are **identity** ids (from the Identity port), not raw accounts, so
    sharding across sockpuppets gains nothing. ``budgets`` holds the current
    window's B(a); it is replenished at window boundaries by realized strength.
    """

    config: ChordConfig
    budgets: Dict[Id, float] = field(default_factory=dict)

    def budget(self, identity_id: Id) -> float:
        """Current B(a); defaults to the base budget B_0 for new identities."""
        return self.budgets.get(identity_id, self.config.budget_B0)

    def replenish(
        self,
        realized_strength: Mapping[Id, float],
        exposure: Mapping[Id, float],
        post_identity: Mapping[Id, Id],
    ) -> None:
        """Apply the window replenishment rule (§8).

        B_{t+1}(a) = B_0 + eta * sum_p Phi(p) E(p), summed over the identity's
        posts in the window, then clamped to [0, budget_max] so budgets stay
        bounded (a stability requirement of §9).
        """
        earned: Dict[Id, float] = defaultdict(float)
        for pid, ident in post_identity.items():
            phi = realized_strength.get(pid, 0.0)
            e = exposure.get(pid, 0.0)
            # Only positive realized strength regenerates budget; a post that
            # earned net negative reception does not *cost* prior budget (that
            # would double-punish), it simply fails to replenish.
            earned[ident] += max(0.0, phi) * e

        cfg = self.config
        new_budgets: Dict[Id, float] = {}
        idents = set(self.budgets) | set(earned)
        for ident in idents:
            b = cfg.budget_B0 + cfg.budget_eta * earned.get(ident, 0.0)
            new_budgets[ident] = float(min(cfg.budget_max, max(0.0, b)))
        self.budgets = new_budgets
