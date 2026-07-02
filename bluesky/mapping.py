"""ATProto ⇄ CHORD type mapping (the Signal + Candidate ports, §3/§4.1).

Jetstream delivers a JSON view of the firehose: each event is a repo *commit*
creating/updating/deleting a record. We translate the three collections we care
about into ``chord.types``:

* ``app.bsky.feed.post``   → :class:`~chord.types.Post` (a candidate to rank)
* ``app.bsky.feed.like``   → a ``FAVORITE`` :class:`~chord.types.Reaction` (+0.5)
* ``app.bsky.feed.repost`` → a ``BOOST``   :class:`~chord.types.Reaction` (+1.0)

Ids are opaque hashables (CHORD convention): a **user/author id is a DID**, a
**post id is the record's AT-URI** ``at://<did>/app.bsky.feed.post/<rkey>``.

Two channels CHORD wants that Bluesky does not natively provide:
  * the *exposed-no-reaction* weak negative (§4.1/§6.2) — sourced not here but from
    the feed generator's own served-but-not-liked skeletons (see ``ranker`` /
    ``store``), which is the only place impression data exists;
  * the *vouch / merit* channel (§10) — no native signal; see ``ranker.VouchSource``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from chord.types import Post, Reaction, ReactionKind

# canonical signed magnitudes (§4.1); the core scales exposed-no-reaction by config.c
_LIKE_VALUE = 0.5      # FAVORITE
_REPOST_VALUE = 1.0    # BOOST


def post_uri(did: str, rkey: str) -> str:
    """The AT-URI that is a post's stable CHORD id."""
    return f"at://{did}/app.bsky.feed.post/{rkey}"


def post_from_uri(uri: str, created_at: float = 0.0) -> Optional[Post]:
    """Synthesize a candidate :class:`Post` from a like/repost *target* URI.

    A post's author DID is encoded in its AT-URI (``at://<did>/app.bsky.feed.post/…``),
    so a post that is being actively liked can become a rankable candidate without our
    having seen its create — which is exactly where the cross-cluster reception signal
    lives (a fresh post has no likes yet). Returns None if ``uri`` is not a feed post."""
    if not isinstance(uri, str) or not uri.startswith("at://"):
        return None
    parts = uri[len("at://"):].split("/")
    if len(parts) != 3 or parts[1] != "app.bsky.feed.post" or not parts[0] or not parts[2]:
        return None
    return Post(id=uri, author_id=parts[0], created_at=created_at)


def parse_created_at(value: Optional[str], fallback_us: Optional[int] = None) -> float:
    """ISO-8601 ``createdAt`` → epoch seconds. Falls back to the Jetstream
    ``time_us`` microsecond stamp, then 0.0. Records carry attacker-controlled
    ``createdAt``, so callers should prefer the firehose stamp for ordering."""
    if value:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    if fallback_us is not None:
        return fallback_us / 1_000_000.0
    return 0.0


def _commit_create(event: dict, collection: str) -> Optional[dict]:
    """Return the commit dict iff this event creates a record in ``collection``."""
    if event.get("kind") != "commit":
        return None
    commit = event.get("commit") or {}
    if commit.get("collection") != collection:
        return None
    if commit.get("operation") not in ("create", "update"):
        return None
    return commit


def event_to_post(event: dict) -> Optional[Post]:
    """A ``app.bsky.feed.post`` create → a candidate :class:`Post` (else None)."""
    commit = _commit_create(event, "app.bsky.feed.post")
    if commit is None:
        return None
    did = event.get("did")
    rkey = commit.get("rkey")
    record = commit.get("record") or {}
    if not did or not rkey:
        return None
    created = parse_created_at(record.get("createdAt"), event.get("time_us"))
    features = {}
    # a cheap, forge-resistant length signal; the depth (§10) merit channel is separate.
    text = record.get("text") or ""
    features["chars"] = float(len(text))
    if record.get("reply"):
        features["is_reply"] = 1.0
    return Post(id=post_uri(did, rkey), author_id=did, created_at=created, features=features)


def event_to_reaction(event: dict) -> Optional[Reaction]:
    """A like/repost create → a signed approval :class:`Reaction` (else None).

    The reacting user is the commit's repo ``did``; the target post is the
    record's ``subject.uri``. Likes are +0.5 (FAVORITE), reposts +1.0 (BOOST)."""
    for collection, value, kind in (
        ("app.bsky.feed.like", _LIKE_VALUE, ReactionKind.FAVORITE),
        ("app.bsky.feed.repost", _REPOST_VALUE, ReactionKind.BOOST),
    ):
        commit = _commit_create(event, collection)
        if commit is None:
            continue
        did = event.get("did")
        record = commit.get("record") or {}
        subject = record.get("subject") or {}
        target = subject.get("uri")
        if not did or not target:
            return None
        ts = parse_created_at(record.get("createdAt"), event.get("time_us"))
        return Reaction(user_id=did, post_id=target, value=value, kind=kind, timestamp=ts)
    return None
