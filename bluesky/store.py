"""Rolling-window event store (the Signal port's memory, §3/§9.1).

Holds the live candidate posts, this window's observed approval reactions, and —
crucially — the exposures the feed generator *itself served*. That served set is
what makes CHORD faithful on Bluesky: a feed generator IS a logging policy, so the
skeletons it returns (with the ε-exploration slice floored in) are exposures with
**known propensity** — exactly the anchor §6.2 identifiability needs. Two signals
are derived here:

* **exposed-no-reaction** (§4.1 weak negative ``-c``): a (user, post) we served but
  the user did not like/repost this window.
* known-π **exposures**: the served skeletons, ε slice flagged EXPLORATION.

Likes we ingest for posts we did *not* serve have no exposure and fall through to
the core as out-of-band (E5a §13#10 down-weights them for the authority signal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from chord.types import Exposure, ExposureSource, Id, Post, Reaction, ReactionKind

from .config import BlueskyConfig


@dataclass
class RollingStore:
    config: BlueskyConfig
    posts: Dict[Id, Post] = field(default_factory=dict)              # uri -> live candidate
    _reactions: List[Reaction] = field(default_factory=list)         # approval + vouch this window
    _served: List[Exposure] = field(default_factory=list)            # our skeletons this window
    _reacted: Set[Tuple[Id, Id]] = field(default_factory=set)        # (user, post) that reacted
    _served_pairs: Set[Tuple[Id, Id]] = field(default_factory=set)   # (user, post) we served

    # ---- ingestion (called by the Jetstream consumer) ----
    def add_post(self, post: Post) -> None:
        self.posts[post.id] = post

    def add_reaction(self, r: Reaction) -> None:
        """An observed like/repost (approval channel). Kept even if the target post
        isn't in our candidate set — it still informs the factorization."""
        self._reactions.append(r)
        if r.value > 0.0:
            self._reacted.add((r.user_id, r.post_id))

    def add_vouch(self, r: Reaction) -> None:
        """A merit-channel vouch from a VouchSource (§10 depth). kind=VOUCH."""
        self._reactions.append(r)

    # ---- serving (called by the ranker after each getFeedSkeleton) ----
    def record_served(self, exposures: Sequence[Exposure]) -> None:
        for e in exposures:
            self._served.append(e)
            self._served_pairs.add((e.user_id, e.post_id))

    # ---- candidates for a request ----
    def candidates(self, now: float) -> List[Post]:
        """Live posts within the candidate horizon, newest-first, capped."""
        horizon = now - self.config.candidate_horizon_seconds
        live = [p for p in self.posts.values() if p.created_at >= horizon]
        live.sort(key=lambda p: p.created_at, reverse=True)
        return live[: self.config.max_candidates]

    # ---- window assembly for fit_window ----
    def build_window(self) -> Tuple[List[Reaction], List[Exposure], Dict[Id, Post]]:
        """Assemble (reactions, exposures, posts) for one ``Chord.fit_window`` call.

        reactions = observed approval + vouches + derived exposed-no-reaction for every
        served (user, post) the user did not react to. exposures = the served skeletons.
        """
        reactions = list(self._reactions)
        neg = -abs(self.config.chord.exposed_no_reaction_c)
        for (user, post) in self._served_pairs:
            if (user, post) not in self._reacted:
                reactions.append(Reaction(user, post, neg,
                                          kind=ReactionKind.EXPOSED_NO_REACTION))
        posts = {p.id: p for p in self.posts.values()}
        return reactions, list(self._served), posts

    def roll(self, now: float) -> None:
        """Close the window: drop this window's reactions/served buffers and prune posts
        that have aged out of the candidate horizon."""
        self._reactions.clear()
        self._served.clear()
        self._reacted.clear()
        self._served_pairs.clear()
        horizon = now - self.config.candidate_horizon_seconds
        stale = [uri for uri, p in self.posts.items() if p.created_at < horizon]
        for uri in stale:
            del self.posts[uri]
