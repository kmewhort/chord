"""Offline: the ATProto feed-generator HTTP endpoints. Needs the [bluesky] extra."""
import base64
import json

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from chord.types import Post, Reaction, ReactionKind

from bluesky.config import BlueskyConfig
from bluesky.ranker import ChordFeed
from bluesky.server import build_app

U = "at://did:plc:auth_u/app.bsky.feed.post/u"


def _fitted_feed():
    cfg = BlueskyConfig(hostname="feed.test", publisher_did="did:plc:owner",
                        window_seconds=100.0, default_slots=10)
    feed = ChordFeed(cfg, seed=0)
    feed.ingest(post=Post(U, "did:plc:auth_u", created_at=1.0), now=1.0)
    for j in range(12):
        fid = f"at://did:plc:f{j}/app.bsky.feed.post/f{j}"
        feed.ingest(post=Post(fid, f"did:plc:f{j}", created_at=1.0), now=1.0)
        for u in (j % 8, (j + 2) % 8):
            feed.ingest(reaction=Reaction(f"did:plc:u{u}", fid, 0.5, ReactionKind.FAVORITE, 2.0))
    for u in range(8):                                    # U is liked by everyone → bridging
        feed.ingest(reaction=Reaction(f"did:plc:u{u}", U, 0.5, ReactionKind.FAVORITE, 2.0))
    feed.fit(now=200.0)
    return feed, cfg


def _client(feed, cfg):
    return TestClient(build_app(feed, cfg, clock=lambda: 250.0))


def _jwt(iss):
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    return f"{seg({'alg':'none'})}.{seg({'iss': iss})}.sig"


def test_did_document_and_describe():
    feed, cfg = _fitted_feed()
    client = _client(feed, cfg)
    doc = client.get("/.well-known/did.json").json()
    assert doc["id"] == "did:web:feed.test"
    assert doc["service"][0]["type"] == "BskyFeedGenerator"
    desc = client.get("/xrpc/app.bsky.feed.describeFeedGenerator").json()
    assert desc["did"] == "did:web:feed.test"
    assert desc["feeds"][0]["uri"] == cfg.feed_uri


def test_get_feed_skeleton_returns_bridging_feed():
    feed, cfg = _fitted_feed()
    client = _client(feed, cfg)
    resp = client.get("/xrpc/app.bsky.feed.getFeedSkeleton",
                      params={"feed": cfg.feed_uri, "limit": 10},
                      headers={"Authorization": f"Bearer {_jwt('did:plc:viewer')}"})
    assert resp.status_code == 200
    body = resp.json()
    # endpoint contract (ranking *quality* is covered by test_ranker's structured world)
    assert "feed" in body and body["feed"], "skeleton must be non-empty"
    assert all(set(item) == {"post"} for item in body["feed"])   # each entry is {"post": uri}
    assert all(item["post"].startswith("at://") for item in body["feed"])
    # the viewer's DID was logged as the served exposure user (personalization + IPW)
    assert any(e.user_id == "did:plc:viewer" for e in feed.store._served)


def test_unknown_feed_uri_rejected_and_anon_allowed():
    feed, cfg = _fitted_feed()
    client = _client(feed, cfg)
    assert client.get("/xrpc/app.bsky.feed.getFeedSkeleton",
                      params={"feed": "at://did:plc:other/app.bsky.feed.generator/x"}).status_code == 400
    anon = client.get("/xrpc/app.bsky.feed.getFeedSkeleton")   # no auth → global feed
    assert anon.status_code == 200 and anon.json()["feed"]


def test_health():
    feed, cfg = _fitted_feed()
    h = _client(feed, cfg).get("/health").json()
    assert h["status"] == "ok" and h["fitted"] is True and h["windows_fit"] == 1
