"""ChordFeed — the learning+serving loop wrapped for Bluesky (§9.1).

Ingest events into the store; every ``window_seconds`` run ``Chord.fit_window`` on
the window; per ``getFeedSkeleton`` request run ``Chord.rank`` and return a feed
skeleton (a list of post AT-URIs). The ranker owns the **ε-exploration slice**: it
floors a uniformly-random, unconfounded fraction of every served feed and logs it
with known propensity, so CHORD's §6.2 anchor is real on Bluesky and the served
skeletons become the logged policy the IPW layer corrects against.

The one genuinely-absent Bluesky signal is the merit/vouch channel (§10). It is a
plug: :class:`VouchSource`. With the default (none), the quality-based E9 prior
(default-on in the core) only *lowers* an untested firehose's prior — its *raising*
requires vouches — and depth stays neutral; both correct, just not earning.
"""
from __future__ import annotations

import math
from typing import List, Optional, Protocol, Sequence

import numpy as np

from chord.config import UserKnobs
from chord.loop import Chord
from chord.types import Exposure, ExposureSource, Id, Post, Reaction

from .config import BlueskyConfig
from .identity import DidIdentityPort
from .store import RollingStore


class VouchSource(Protocol):
    """Merit-channel slot (§10). A host wires a real one — a custom ``app.chord.vouch``
    lexicon record, a trusted labeler, or a model score — returning VOUCH reactions."""

    def vouches(self, posts: Sequence[Post]) -> Sequence[Reaction]: ...


class NoVouchSource:
    """Default: Bluesky has no native merit vote, so none. (E9 still demotes the
    firehose; it just cannot *promote* on unearned merit — which is the safe side.)"""

    def vouches(self, posts: Sequence[Post]) -> Sequence[Reaction]:
        return []


P_ORGANIC = 0.5    # nominal logging propensity for a rank-selected (organic) exposure


class ChordFeed:
    def __init__(self, config: Optional[BlueskyConfig] = None,
                 vouch_source: Optional[VouchSource] = None, seed: int = 0):
        self.config = config or BlueskyConfig()
        self.chord = Chord(self.config.chord, seed=seed)
        self.store = RollingStore(self.config)
        self.identity = DidIdentityPort()
        self.vouch_source = vouch_source or NoVouchSource()
        self._rng = np.random.default_rng(seed)
        self._last_fit: Optional[float] = None
        self.fitted = False
        self.windows_fit = 0

    # ------------------------------------------------------------ ingestion
    def ingest(self, post: Optional[Post] = None, reaction: Optional[Reaction] = None,
               now: float = 0.0) -> None:
        """Route one mapped event into the store and update identity first-seen."""
        if post is not None:
            self.store.add_post(post)
            self.identity.observe(post.author_id, post.created_at or now)
        if reaction is not None:
            self.store.add_reaction(reaction)
            self.identity.observe(reaction.user_id, reaction.timestamp or now)

    # ------------------------------------------------------------ learning
    def maybe_fit(self, now: float) -> bool:
        """Fit a window if ``window_seconds`` have elapsed. Returns whether it fit."""
        if self._last_fit is None:
            self._last_fit = now
            return False
        if now - self._last_fit < self.config.window_seconds:
            return False
        self.fit(now)
        return True

    def fit(self, now: float) -> None:
        """Run one §9.1 learning window over the accumulated events, then roll."""
        reactions, exposures, posts = self.store.build_window()
        # merit channel (§10): fold in any vouches for this window's candidates
        vouches = list(self.vouch_source.vouches(list(posts.values())))
        reactions = reactions + list(vouches)
        self.identity.now = now
        idmap = self.identity.identity_map(
            {r.user_id for r in reactions} | {p.author_id for p in posts.values()})
        if reactions and posts:
            self.chord.fit_window(reactions, posts, exposures, identity_of=idmap)
            self.fitted = True
            self.windows_fit += 1
        self.store.roll(now)
        self._last_fit = now

    # ------------------------------------------------------------ serving
    def serve(self, user_did: Id, limit: int, now: float) -> List[Id]:
        """Return a feed skeleton (list of post AT-URIs) and log what we served.

        Before the first fit we serve reverse-chronological; after, we serve
        ``Chord.rank`` plus a floored uniformly-random ε slice, and log both as
        exposures with known propensity (ORGANIC p≈0.5, EXPLORATION p=ε)."""
        candidates = self.store.candidates(now)
        if not candidates:
            return []
        limit = max(1, min(limit, self.config.default_slots))

        if not self.fitted:
            feed = [p.id for p in candidates[:limit]]
            self._log(user_did, feed, set(), now)          # all organic, cold-start
            return feed

        knobs = self.config.default_knobs
        organic = self.chord.rank(user_did, candidates, knobs, n_slots=limit)
        # ε slice: uniformly-random fresh candidates NOT already ranked in — the
        # unconfounded, known-π anchor (§6.2). ceil so it never rounds to zero (§8).
        eps = float(min(self.config.chord.epsilon_max,
                        max(self.chord.controller.state.epsilon_min, knobs.epsilon)))
        n_expl = min(math.ceil(eps * limit), max(0, len(candidates) - len(organic)))
        chosen = set(organic)
        pool = [p.id for p in candidates if p.id not in chosen]
        explore: List[Id] = []
        if n_expl > 0 and pool:
            idx = self._rng.choice(len(pool), size=min(n_expl, len(pool)), replace=False)
            explore = [pool[int(i)] for i in np.atleast_1d(idx)]

        feed = self._interleave(organic, explore, limit)
        self._log(user_did, feed, set(explore), now)
        return feed

    # ------------------------------------------------------------ internals
    def _interleave(self, organic: List[Id], explore: List[Id], limit: int) -> List[Id]:
        """Place the ε posts at roughly even positions so they are actually shown."""
        if not explore:
            return organic[:limit]
        out, ei = [], 0
        step = max(1, len(organic) // (len(explore) + 1))
        for i, pid in enumerate(organic):
            if ei < len(explore) and i > 0 and i % step == 0:
                out.append(explore[ei]); ei += 1
            out.append(pid)
        out.extend(explore[ei:])
        return out[:limit]

    def _log(self, user_did: Id, feed: List[Id], explore: set, now: float) -> None:
        eps = float(min(self.config.chord.epsilon_max,
                        max(self.chord.controller.state.epsilon_min,
                            self.config.default_knobs.epsilon)))
        exposures = []
        for slot, pid in enumerate(feed):
            is_expl = pid in explore
            exposures.append(Exposure(
                user_id=user_did, post_id=pid, timestamp=now, slot=slot,
                source=ExposureSource.EXPLORATION if is_expl else ExposureSource.ORGANIC,
                propensity=eps if is_expl else P_ORGANIC,
            ))
        self.store.record_served(exposures)
