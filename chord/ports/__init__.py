"""Ports and default adapters (§3, Appendix D.3)."""
from .base import (
    CandidatePort,
    IdentityPort,
    PartitionPort,
    PreferencePort,
    SignalPort,
)
from .adapters import (
    AccountAgeIdentityAdapter,
    KMeansPartitionAdapter,
    LocalPreferenceAdapter,
    NativeSignalAdapter,
    TimelineCandidateAdapter,
)

__all__ = [
    "IdentityPort",
    "PreferencePort",
    "SignalPort",
    "CandidatePort",
    "PartitionPort",
    "AccountAgeIdentityAdapter",
    "LocalPreferenceAdapter",
    "NativeSignalAdapter",
    "TimelineCandidateAdapter",
    "KMeansPartitionAdapter",
]
