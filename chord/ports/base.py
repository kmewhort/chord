"""Port protocols (§3).

CHORD enters the world through six ports, each with a crude built-in default
adapter (so the core runs standalone on a single Mastodon instance) and an
optional rich adapter slot (so a capable host upgrades it). The load-bearing
observation for federation (§3): identity, preference, signal, and candidate all
have *some* native fediverse answer; opinion-clustering (Partition) does not,
which is why the core ships its own factorization as the default.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from ..config import UserKnobs
from ..types import Exposure, Id, Post, Reaction


@runtime_checkable
class IdentityPort(Protocol):
    """Stable handle + forge-cost (§3, §10). Budgets/Sybil resistance bind here."""

    def identity_of(self, account_id: Id) -> Id: ...
    def forge_cost(self, account_id: Id) -> float: ...


@runtime_checkable
class PreferencePort(Protocol):
    """Influent function + history — the user's portable profile (§3, D.3)."""

    def knobs(self, user_id: Id) -> UserKnobs: ...
    def set_knobs(self, user_id: Id, knobs: UserKnobs) -> None: ...


@runtime_checkable
class SignalPort(Protocol):
    """Attention-event stream (§3). Default = native reactions."""

    def reactions(self) -> Sequence[Reaction]: ...
    def exposures(self) -> Sequence[Exposure]: ...


@runtime_checkable
class CandidatePort(Protocol):
    """Posts to re-rank (§3). Retrieval stays upstream; CHORD only re-ranks."""

    def candidates(self, user_id: Id) -> Sequence[Post]: ...


@runtime_checkable
class PartitionPort(Protocol):
    """Opinion clusters (§3). The one port with no native fediverse answer."""

    def assign(self, user_ids: Sequence[Id], embeddings: Mapping[Id, "object"]) -> Dict[Id, int]: ...
