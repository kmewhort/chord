"""DID-backed Identity port (§3/§10).

On Bluesky the account *is* a DID, which is already the sybil-resistant handle the
author-visibility budget (§8) and Sybil defenses (§5) want to bind to — you cannot
shard a budget across sockpuppets without minting new DIDs, and a fresh DID has no
history. So ``identity_of`` is 1:1 on DIDs (with an optional alias map for accounts
a host has verified are the same human), and ``forge_cost`` grows with the account's
observed age — a brand-new DID is Sybil-cheap, an old one is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from typing import Dict, Optional

from chord.types import Id

_SECONDS_PER_DAY = 86_400.0


@dataclass
class DidIdentityPort:
    """Implements :class:`chord.ports.base.IdentityPort` over DIDs.

    ``first_seen`` maps a DID → the epoch second we first observed it (the feed
    generator fills this as the firehose flows); ``aliases`` maps an account DID →
    a canonical identity DID when a host has proven two accounts are one human.
    """

    first_seen: Dict[Id, float] = field(default_factory=dict)
    aliases: Dict[Id, Id] = field(default_factory=dict)
    now: float = 0.0    # current epoch second; set by the feed generator each window

    def observe(self, did: Id, at_epoch: float) -> None:
        """Record first-seen for age-based forge-cost (idempotent: keeps the earliest)."""
        prev = self.first_seen.get(did)
        if prev is None or at_epoch < prev:
            self.first_seen[did] = at_epoch

    def identity_of(self, account_id: Id) -> Id:
        return self.aliases.get(account_id, account_id)

    def forge_cost(self, account_id: Id) -> float:
        """Monotone in observed account age (days). Unknown/new DIDs ≈ 0."""
        seen = self.first_seen.get(self.identity_of(account_id))
        if seen is None or self.now <= seen:
            return 0.0
        age_days = (self.now - seen) / _SECONDS_PER_DAY
        return float(log1p(max(0.0, age_days)))

    def identity_map(self, account_ids) -> Optional[Dict[Id, Id]]:
        """The ``identity_of`` mapping fit_window wants — None when it is a pure 1:1
        (no aliases), so the core can skip the indirection."""
        if not self.aliases:
            return None
        return {a: self.identity_of(a) for a in account_ids}
