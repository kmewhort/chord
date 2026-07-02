"""Configuration for the CHORD Bluesky feed generator.

Two layers: :class:`BlueskyConfig` holds the *service* wiring (who we are on the
network, what we ingest, how long a window is) and carries a plain
:class:`chord.config.ChordConfig` for the ranking core. Nothing here changes the
core — the feed generator is built strictly on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from chord.config import ChordConfig, UserKnobs


@dataclass
class BlueskyConfig:
    """Service wiring for one CHORD feed.

    ``hostname`` is the public host the feed generator is served from; the service
    identity is ``did:web:<hostname>`` (no PLC registration needed for a feedgen).
    ``publisher_did``/``feed_rkey`` identify the ``app.bsky.feed.generator`` record
    that points at us (created once via ``bluesky.publish``).
    """

    # --- network identity (§3 Identity port lives on DIDs) ---
    hostname: str = "feed.example.com"          # public https host → did:web:<hostname>
    publisher_did: str = ""                     # the account that owns the feed record
    feed_rkey: str = "chord"                    # rkey of the app.bsky.feed.generator record
    display_name: str = "CHORD"
    description: str = "Bridging feed: tested cross-cluster support net of divisiveness."

    # --- ingestion (Jetstream: a JSON view of the firehose) ---
    jetstream_url: str = "wss://jetstream2.us-east.bsky.network/subscribe"
    # collections we consume; posts feed candidates, likes/reposts feed the signed
    # approval channel (§4.1). Follows/blocks are available but unused in v1.
    wanted_collections: List[str] = field(default_factory=lambda: [
        "app.bsky.feed.post",
        "app.bsky.feed.like",
        "app.bsky.feed.repost",
    ])

    # --- scoping (keep a dev run — or a topical feed — from ingesting the whole network) ---
    # The global firehose is far too much to rank in one window; scope it. Any of:
    wanted_dids: List[str] = field(default_factory=list)   # Jetstream: only these repos (≤10k)
    sample_rate: float = 1.0                    # keep this fraction of the firehose (by actor DID)
    max_posts: int = 20_000                     # hard cap on stored candidates (evict oldest)

    # --- windowing (§9.1 learning plane cadence) ---
    window_seconds: float = 900.0               # a learning window = 15 min of events
    candidate_horizon_seconds: float = 6 * 3600.0   # how far back a post stays a candidate
    max_candidates: int = 4000                  # cap scored per request (newest first)

    # --- serving ---
    default_slots: int = 50                     # feed skeleton length per getFeedSkeleton
    require_auth: bool = False                   # verify the requester JWT (per-user feeds)

    # --- the ranking core ---
    chord: ChordConfig = field(default_factory=lambda: ChordConfig(
        d=16, n_clusters=2, mf_iters=20,
        # a real network is a firehose; keep the budget binding so §8 dilutes volume.
        budget_B0=8.0, budget_max=80.0,
    ))
    default_knobs: UserKnobs = field(default_factory=lambda: UserKnobs(M=0.8))

    @property
    def service_did(self) -> str:
        return f"did:web:{self.hostname}"

    @property
    def feed_uri(self) -> str:
        """The at:// URI of the feed record clients subscribe to."""
        did = self.publisher_did or self.service_did
        return f"at://{did}/app.bsky.feed.generator/{self.feed_rkey}"
