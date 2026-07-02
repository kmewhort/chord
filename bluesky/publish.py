"""Publish the feed record so clients can find the CHORD feed (one-time setup).

A custom feed is discoverable via an ``app.bsky.feed.generator`` record in the
publisher's repo that points at the service DID. This writes it with the account's
handle+app-password. Run once (or to update the display name/description). Needs the
``[bluesky]`` extra: httpx.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .config import BlueskyConfig


def publish_feed(config: BlueskyConfig, handle: str, app_password: str,
                 pds_host: str = "https://bsky.social") -> str:
    """Create/update the feed-generator record; returns its at:// URI."""
    import httpx  # deferred so the core/tests don't need the [bluesky] extra

    with httpx.Client(base_url=pds_host, timeout=30.0) as client:
        session = client.post("/xrpc/com.atproto.server.createSession",
                              json={"identifier": handle, "password": app_password})
        session.raise_for_status()
        data = session.json()
        did, jwt = data["did"], data["accessJwt"]

        record = {
            "$type": "app.bsky.feed.generator",
            "did": config.service_did,
            "displayName": config.display_name,
            "description": config.description,
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        resp = client.post("/xrpc/com.atproto.repo.putRecord",
                           headers={"Authorization": f"Bearer {jwt}"},
                           json={"repo": did, "collection": "app.bsky.feed.generator",
                                 "rkey": config.feed_rkey, "record": record})
        resp.raise_for_status()
    # the record lives in the publisher's repo, so the feed URI uses their DID
    return f"at://{did}/app.bsky.feed.generator/{config.feed_rkey}"
