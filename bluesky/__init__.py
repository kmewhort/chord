"""CHORD as a Bluesky (ATProto) feed generator — built strictly on top of the core.

This package maps ATProto/Jetstream data into ``chord.types``, runs the CHORD
learning+serving loop (``chord.loop.Chord``), and serves
``app.bsky.feed.getFeedSkeleton``. The pure ``chord`` core is untouched; everything
here is an implementation of its ports (§3) plus the network plumbing the core
deliberately omits. See ``bluesky/README.md``.
"""
from .config import BlueskyConfig
from .ranker import ChordFeed, NoVouchSource, VouchSource

__all__ = ["BlueskyConfig", "ChordFeed", "VouchSource", "NoVouchSource"]
