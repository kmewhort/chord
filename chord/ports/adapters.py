"""Default (crude) port adapters (§3).

Each ships so the core runs standalone; a capable host swaps in a rich adapter
(verified-human/ZK identity, portable Solid/ATProto pods, external Polis
clustering, richer telemetry) without touching the core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..config import ChordConfig, UserKnobs
from ..types import (
    DEFAULT_REACTION_VALUES,
    Exposure,
    Id,
    Post,
    Reaction,
    ReactionKind,
)


# --------------------------------------------------------------- Identity
@dataclass
class AccountAgeIdentityAdapter:
    """Default Identity adapter: account-age / heuristic forge-cost (§3, D.3).

    Maps each account to an identity (here 1:1 unless an explicit alias map is
    given, e.g. a host that has verified two accounts are the same human) and
    derives a forge-cost from account age — old accounts are costlier to forge.
    """

    ages: Dict[Id, float] = field(default_factory=dict)  # account age in windows
    aliases: Dict[Id, Id] = field(default_factory=dict)  # account -> identity

    def identity_of(self, account_id: Id) -> Id:
        return self.aliases.get(account_id, account_id)

    def forge_cost(self, account_id: Id) -> float:
        # Monotone in age; brand-new accounts are near-free to forge (Sybil-cheap).
        age = self.ages.get(account_id, 0.0)
        return float(np.log1p(max(0.0, age)))


# ------------------------------------------------------------- Preference
@dataclass
class LocalPreferenceAdapter:
    """Default Preference adapter: a local store of per-user knobs (§3, D.3).

    A rich adapter would back this with a portable data pod (Solid/ATProto) so
    the profile survives an instance move.
    """

    default_knobs: UserKnobs = field(default_factory=UserKnobs)
    store: Dict[Id, UserKnobs] = field(default_factory=dict)

    def knobs(self, user_id: Id) -> UserKnobs:
        return self.store.get(user_id, self.default_knobs)

    def set_knobs(self, user_id: Id, knobs: UserKnobs) -> None:
        knobs.validate()
        self.store[user_id] = knobs


# ----------------------------------------------------------------- Signal
@dataclass
class NativeSignalAdapter:
    """Default Signal adapter: native reactions + exposures (§3).

    Reaction *kinds* are mapped to signed values (§4.1); the exposed-no-reaction
    weak negative is scaled by ``config.exposed_no_reaction_c``.
    """

    config: ChordConfig
    _reactions: List[Reaction] = field(default_factory=list)
    _exposures: List[Exposure] = field(default_factory=list)

    def record_reaction(self, user_id: Id, post_id: Id, kind: ReactionKind,
                        timestamp: float = 0.0) -> None:
        base = DEFAULT_REACTION_VALUES[kind]
        if kind is ReactionKind.EXPOSED_NO_REACTION:
            base = -abs(self.config.exposed_no_reaction_c)
        self._reactions.append(
            Reaction(user_id, post_id, float(base), kind=kind, timestamp=timestamp)
        )

    def record_exposure(self, exposure: Exposure) -> None:
        self._exposures.append(exposure)

    def reactions(self) -> Sequence[Reaction]:
        return list(self._reactions)

    def exposures(self) -> Sequence[Exposure]:
        return list(self._exposures)


# -------------------------------------------------------------- Candidate
@dataclass
class TimelineCandidateAdapter:
    """Default Candidate adapter: a per-user home/instance timeline (§3, D.2)."""

    timelines: Dict[Id, List[Post]] = field(default_factory=dict)
    shared: List[Post] = field(default_factory=list)

    def candidates(self, user_id: Id) -> Sequence[Post]:
        return list(self.timelines.get(user_id, [])) + list(self.shared)


# -------------------------------------------------------------- Partition
@dataclass
class KMeansPartitionAdapter:
    """Default Partition adapter: built-in k-means on opinion embeddings (§3, §4).

    The one port with no native fediverse answer, so the core must ship it. A
    rich adapter can delegate to an external Polis / bridging service. This is a
    dependency-free, deterministic (seeded) k-means over the fitted x_u.
    """

    n_clusters: int = 2
    seed: int = 0
    iters: int = 50

    def assign(self, user_ids: Sequence[Id], embeddings: Mapping[Id, np.ndarray]) -> Dict[Id, int]:
        pts = [embeddings[u] for u in user_ids if u in embeddings]
        keys = [u for u in user_ids if u in embeddings]
        if not pts:
            return {}
        X = np.stack(pts)
        k = min(self.n_clusters, len(X))
        if k <= 1:
            return {u: 0 for u in keys}
        rng = np.random.default_rng(self.seed)
        # k-means++ style seeding for stability.
        centers = _kmeans_pp_init(X, k, rng)
        labels = np.zeros(len(X), dtype=int)
        for _ in range(self.iters):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = d.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = X[mask].mean(axis=0)
        return {u: int(labels[i]) for i, u in enumerate(keys)}


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    first = rng.integers(0, n)
    centers = [X[first]]
    for _ in range(1, k):
        d = np.min(
            np.stack([((X - c) ** 2).sum(axis=1) for c in centers], axis=1), axis=1
        )
        total = d.sum()
        if total <= 0:
            centers.append(X[rng.integers(0, n)])
            continue
        probs = d / total
        idx = rng.choice(n, p=probs)
        centers.append(X[idx])
    return np.stack(centers)
