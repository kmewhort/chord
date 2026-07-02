"""Firehose ingestion via Jetstream (the Signal port, §3).

Jetstream is a JSON view of the ATProto firehose (no CBOR/CAR decoding needed). We
subscribe to the post/like/repost collections, map each commit to ``chord.types``
(``mapping``), and feed it to the ``ChordFeed``, advancing the learning window as
time passes. ``handle_event`` is pure and sync so it is testable offline against
recorded events; ``run_ingestion`` is the live async loop (needs the ``[bluesky]``
extra: websockets).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional
from urllib.parse import urlencode

from .config import BlueskyConfig
from .mapping import event_to_post, event_to_reaction
from .ranker import ChordFeed


def handle_event(feed: ChordFeed, event: dict, now: float) -> bool:
    """Map one Jetstream event into the feed. Returns whether it was consumed."""
    post = event_to_post(event)
    if post is not None:
        feed.ingest(post=post, now=now)
        return True
    reaction = event_to_reaction(event)
    if reaction is not None:
        feed.ingest(reaction=reaction, now=now)
        return True
    return False


def subscribe_url(config: BlueskyConfig) -> str:
    params = [("wantedCollections", c) for c in config.wanted_collections]
    return f"{config.jetstream_url}?{urlencode(params)}"


async def run_ingestion(feed: ChordFeed, config: Optional[BlueskyConfig] = None,
                        clock: Callable[[], float] = time.time,
                        stop: Optional[asyncio.Event] = None,
                        max_events: Optional[int] = None) -> int:
    """Consume the live Jetstream, ingesting and advancing windows. Reconnects on
    drop. Returns the number of events consumed (bounded by ``max_events``)."""
    import websockets  # deferred so the core/tests don't need the [bluesky] extra

    config = config or feed.config
    url = subscribe_url(config)
    seen = 0
    async for ws in websockets.connect(url, max_size=None):
        try:
            async for raw in ws:
                event = json.loads(raw)
                now = clock()
                handle_event(feed, event, now)
                feed.maybe_fit(now)
                seen += 1
                if (max_events is not None and seen >= max_events) or (stop and stop.is_set()):
                    return seen
        except Exception:
            if stop and stop.is_set():
                return seen
            continue    # reconnect
    return seen
