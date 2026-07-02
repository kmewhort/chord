"""Offline: Jetstream commit events → chord.types. No network."""
from chord.types import ReactionKind

from bluesky.mapping import event_to_post, event_to_reaction, parse_created_at, post_uri


def _post_event(did, rkey, text, created="2024-09-01T00:00:00.000Z"):
    return {"did": did, "time_us": 1_725_000_000_000_000, "kind": "commit",
            "commit": {"operation": "create", "collection": "app.bsky.feed.post",
                       "rkey": rkey, "record": {"$type": "app.bsky.feed.post",
                                                "text": text, "createdAt": created}}}


def _like_event(did, target_uri, coll="app.bsky.feed.like"):
    return {"did": did, "time_us": 1_725_000_000_500_000, "kind": "commit",
            "commit": {"operation": "create", "collection": coll, "rkey": "r1",
                       "record": {"createdAt": "2024-09-01T00:00:05.000Z",
                                  "subject": {"uri": target_uri, "cid": "bafy"}}}}


def test_post_event_maps_to_post():
    p = event_to_post(_post_event("did:plc:alice", "abc", "hello world"))
    assert p is not None
    assert p.id == post_uri("did:plc:alice", "abc")
    assert p.author_id == "did:plc:alice"
    assert p.features["chars"] == 11.0
    assert p.created_at == parse_created_at("2024-09-01T00:00:00.000Z")
    # a like event is not a post
    assert event_to_post(_like_event("did:plc:bob", p.id)) is None


def test_like_and_repost_map_to_signed_reactions():
    uri = post_uri("did:plc:alice", "abc")
    like = event_to_reaction(_like_event("did:plc:bob", uri))
    assert like is not None and like.kind is ReactionKind.FAVORITE
    assert like.user_id == "did:plc:bob" and like.post_id == uri and like.value == 0.5

    repost = event_to_reaction(_like_event("did:plc:carol", uri, "app.bsky.feed.repost"))
    assert repost is not None and repost.kind is ReactionKind.BOOST and repost.value == 1.0
    # a post event is not a reaction
    assert event_to_reaction(_post_event("did:plc:alice", "abc", "x")) is None


def test_delete_and_unknown_ops_ignored():
    ev = _post_event("did:plc:alice", "abc", "x")
    ev["commit"]["operation"] = "delete"
    assert event_to_post(ev) is None
    assert event_to_reaction({"kind": "identity"}) is None
