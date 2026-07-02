"""Offline: Jetstream events drive ingestion via the pure handle_event path."""
from chord.types import Post

from bluesky.config import BlueskyConfig
from bluesky.jetstream import handle_event, keep_event, subscribe_url
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


def test_subscribe_url_requests_the_wanted_collections_and_dids():
    url = subscribe_url(BlueskyConfig(wanted_dids=["did:plc:a", "did:plc:b"]))
    assert url.startswith("wss://")
    assert "wantedCollections=app.bsky.feed.post" in url
    assert "app.bsky.feed.like" in url and "app.bsky.feed.repost" in url
    assert "wantedDids=did%3Aplc%3Aa" in url and "wantedDids=did%3Aplc%3Ab" in url


def test_sample_rate_gate_is_deterministic_and_scales():
    ev = lambda did: {"did": did}
    assert keep_event(ev("did:plc:x"), 1.0)                 # full = keep all
    assert not keep_event(ev("did:plc:x"), 0.0)             # zero = drop all
    kept = [keep_event(ev(f"did:plc:{i}"), 0.3) for i in range(2000)]
    assert 0.2 < sum(kept) / len(kept) < 0.4                # ~30% kept
    # deterministic: the same DID always lands the same way (a coherent thinned graph)
    assert keep_event(ev("did:plc:x"), 0.3) == keep_event(ev("did:plc:x"), 0.3)


def test_store_cap_bounds_memory():
    feed = ChordFeed(BlueskyConfig(max_posts=100), seed=0)
    for i in range(500):                                    # a burst well over the cap
        feed.ingest(post=Post(f"at://did:plc:a/app.bsky.feed.post/{i}",
                              "did:plc:a", created_at=float(i)), now=float(i))
    assert len(feed.store.posts) <= 125                     # trimmed toward the cap
    # the newest posts are the ones kept
    assert "at://did:plc:a/app.bsky.feed.post/499" in feed.store.posts
