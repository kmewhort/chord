"""Core data types for CHORD.

These are the lingua franca that flows through the ports (§3) and between the
learning and serving planes (Appendix D). Everything is a plain, hashable-ish
dataclass so it can be logged to an append-only event log and replayed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Hashable, Optional

# A user / post / author identifier. Kept opaque (any hashable) so callers can
# use ints, strings, or handles.
Id = Hashable


class ReactionKind(Enum):
    """Native fediverse reactions, mapped to the signed values of §4.1.

    ``r_up`` (e.g. boost +1, favorite +0.5, exposed-no-reaction -c, mute -1).
    The exposed-no-reaction value is a *weak* negative (a "silent disagreement"
    signal, §6.2) and its magnitude ``c`` is configurable; here we store the
    canonical unit and let the signal adapter scale it.
    """

    BOOST = "boost"
    FAVORITE = "favorite"
    EXPOSED_NO_REACTION = "exposed_no_reaction"
    MUTE = "mute"


# Canonical signed values. ``EXPOSED_NO_REACTION`` uses the unit weak-negative
# ``-c``; the Signal port scales it by the configured ``c``.
DEFAULT_REACTION_VALUES: Dict[ReactionKind, float] = {
    ReactionKind.BOOST: 1.0,
    ReactionKind.FAVORITE: 0.5,
    ReactionKind.EXPOSED_NO_REACTION: -1.0,  # scaled by config.c (0<c<1)
    ReactionKind.MUTE: -1.0,
}


class ExposureSource(Enum):
    """Where an exposure came from.

    ``ORGANIC`` exposures are chosen by the personalized ranker and therefore
    confounded with alignment (§6.1). ``EXPLORATION`` exposures come from the
    floored commons pool (§8) and are alignment-*independent* — the unconfounded
    anchor that makes propensities estimable (§6.2, §9.3).
    """

    ORGANIC = "organic"
    EXPLORATION = "exploration"


@dataclass(frozen=True)
class Post:
    """A candidate item to be ranked."""

    id: Id
    author_id: Id
    created_at: float = 0.0
    # Optional intrinsic content features used by e.g. the depth factor (§7.3)
    # and by the simulator's response model. Free-form.
    features: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Exposure:
    """An event: user ``user_id`` was *shown* post ``post_id``.

    Logging exposure — not just reactions — is what makes the propensity layer
    possible (Appendix D.2). ``propensity`` is the (known or estimated)
    probability the pair was exposed under the logging policy.
    """

    user_id: Id
    post_id: Id
    timestamp: float = 0.0
    slot: int = 0
    source: ExposureSource = ExposureSource.ORGANIC
    propensity: Optional[float] = None


@dataclass(frozen=True)
class Reaction:
    """A signed reaction event (§4.1)."""

    user_id: Id
    post_id: Id
    value: float
    kind: Optional[ReactionKind] = None
    timestamp: float = 0.0
