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
        """Apply the window replenishment rule (§8) — a floored, capped **replicator**
        update: reach reproduces in proportion to earned cross-cluster strength.

        Default (per-author, memoryless):
            B_{t+1}(a) = clip[ B_0 + γ·(B_t(a) − B_0) + η·Σ_p Φ(p) E(p) ]
        The γ carry term (``budget_memory``, #1) keeps an irregular cadence from being reset
        to the floor. With ``budget_share_based`` (#4), the η·ΣΦE earnings term is replaced by
        a share of a *fixed aggregate pool* (so total issuance is conserved system-wide, not
        procyclical): each identity gets ``surplus · earned(a)/Σearned``.
        """
        earned: Dict[Id, float] = defaultdict(float)
        for pid, ident in post_identity.items():
            phi = realized_strength.get(pid, 0.0)
            e = exposure.get(pid, 0.0)
            # Only positive realized strength regenerates budget; a post that earned net
            # negative reception does not *cost* prior budget (that would double-punish), it
            # simply fails to replenish.
            earned[ident] += max(0.0, phi) * e

        cfg = self.config
        idents = set(self.budgets) | set(earned)
        n = max(len(idents), 1)

        if cfg.budget_share_based:
            # #4: a fixed pool per window, distributed by *relative* earned strength.
            pool = cfg.budget_aggregate_factor * n * cfg.budget_B0
            surplus = max(0.0, pool - n * cfg.budget_B0)
            total_earned = sum(earned.values())
            def earn_component(ident: Id) -> float:
                share = (earned[ident] / total_earned) if total_earned > 0 else (1.0 / n)
                return surplus * share
        else:
            def earn_component(ident: Id) -> float:
                return cfg.budget_eta * earned.get(ident, 0.0)

        new_budgets: Dict[Id, float] = {}
        for ident in idents:
            carry = cfg.budget_memory * (self.budget(ident) - cfg.budget_B0)   # #1
            b = cfg.budget_B0 + carry + earn_component(ident)
            new_budgets[ident] = float(min(cfg.budget_max, max(0.0, b)))
        self.budgets = new_budgets

    def replicator_gain(
        self,
        realized_strength: Mapping[Id, float],
        exposure: Mapping[Id, float],
        post_identity: Mapping[Id, Id],
    ) -> float:
        """Effective linear gain γ + η·Φ̄ of the replicator recursion (#2 diagnostic).

        When a fully-spending author's earnings ≈ Φ̄·B_t, the recursion is linear with this
        gain; its fixed point B* = B_0·(1−γ)/(1 − γ − η·Φ̄) blows up as the gain → 1. A gain
        ≥ 1 is a *phase transition* (authors above the critical strength 1/η run to the cap),
        not a smooth knob — so η should be set to keep this comfortably below 1."""
        cfg = self.config
        if cfg.budget_share_based:
            return cfg.budget_memory            # share-based earnings do not compound B_t
        phis = [max(0.0, realized_strength.get(pid, 0.0)) for pid in post_identity]
        phi_bar = float(sum(phis) / len(phis)) if phis else 0.0
        return cfg.budget_memory + cfg.budget_eta * phi_bar
