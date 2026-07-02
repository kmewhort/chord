"""Offline: Jetstream events drive ingestion via the pure handle_event path."""
from bluesky.config import BlueskyConfig
from bluesky.jetstream import handle_event, subscribe_url
from bluesky.ranker import ChordFeed


def _post_event(did, rkey, text="hi"):
    return {"did": did, "time_us": 1_725_000_000_000_000, "kind": "commit",
            "commit": {"operation": "create", "collection": "app.bsky.feed.post",
                       "rkey": rkey, "record": {"text": text,
                                                "createdAt": "2024-09-01T00:00:00.000Z"}}}


def _like_event(did, uri):
    return {"did": did, "time_us": 1_725_000_000_500_000, "kind": "commit",
            "commit": {"operation": "create", "collection": "app.bsky.feed.like", "rkey": "l1",
                       "record": {"createdAt": "2024-09-01T00:00:05.000Z",
                                  "subject": {"uri": uri, "cid": "bafy"}}}}


def test_handle_event_ingests_posts_and_reactions():
    feed = ChordFeed(BlueskyConfig(), seed=0)
    uri = "at://did:plc:alice/app.bsky.feed.post/abc"
    assert handle_event(feed, _post_event("did:plc:alice", "abc"), now=1.0)
    assert uri in feed.store.posts
    assert feed.identity.first_seen.get("did:plc:alice") is not None   # first-seen recorded
    assert handle_event(feed, _like_event("did:plc:bob", uri), now=2.0)
    assert any(r.post_id == uri and r.user_id == "did:plc:bob" for r in feed.store._reactions)
    # non-content events (identity/account) are ignored
    assert not handle_event(feed, {"kind": "identity", "did": "did:plc:x"}, now=3.0)


def test_subscribe_url_requests_the_wanted_collections():
    url = subscribe_url(BlueskyConfig())
    assert url.startswith("wss://")
    assert "wantedCollections=app.bsky.feed.post" in url
    assert "app.bsky.feed.like" in url and "app.bsky.feed.repost" in url
