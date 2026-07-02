"""The feed-generator HTTP service (ATProto ``app.bsky.feed`` endpoints).

A Bluesky custom feed is an HTTP service that answers ``getFeedSkeleton`` with a
list of post URIs; the AppView hydrates them. We answer it from ``ChordFeed``:
resolve the viewer DID from the service-auth JWT, run the window/serve loop, return
the skeleton. Also serves ``describeFeedGenerator`` and the ``did:web`` document so
the feed is discoverable and publishable.

Requires the ``[bluesky]`` extra (starlette). Ingestion runs separately (see
``bluesky.jetstream`` / ``bluesky.__main__``) and shares the same ``ChordFeed``.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Callable, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import BlueskyConfig
from .ranker import ChordFeed

ANON_VIEWER = "did:chord:anon"


def _viewer_did(request: Request, require_auth: bool) -> str:
    """Extract the requester DID from the service-auth JWT ``iss`` claim.

    v1 decodes the JWT payload without verifying the signature — enough to
    personalize; a production deployment should verify against the issuer's signing
    key (resolve their DID doc). Anonymous requests get the global bridging feed."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:]
        try:
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)             # pad base64url
            claims = json.loads(base64.urlsafe_b64decode(payload))
            iss = claims.get("iss")
            if iss:
                return iss
        except Exception:
            pass
    return ANON_VIEWER


def build_app(feed: ChordFeed, config: Optional[BlueskyConfig] = None,
              clock: Callable[[], float] = time.time) -> Starlette:
    config = config or feed.config

    async def get_feed_skeleton(request: Request) -> JSONResponse:
        requested = request.query_params.get("feed")
        if requested and requested != config.feed_uri:
            return JSONResponse({"error": "UnknownFeed"}, status_code=400)
        try:
            limit = int(request.query_params.get("limit", config.default_slots))
        except ValueError:
            limit = config.default_slots
        now = clock()
        feed.maybe_fit(now)                                  # advance the learning plane
        viewer = _viewer_did(request, config.require_auth)
        skeleton = feed.serve(viewer, limit=limit, now=now)
        return JSONResponse({"feed": [{"post": uri} for uri in skeleton]})

    async def describe(request: Request) -> JSONResponse:
        return JSONResponse({"did": config.service_did,
                             "feeds": [{"uri": config.feed_uri}]})

    async def did_document(request: Request) -> JSONResponse:
        return JSONResponse({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": config.service_did,
            "service": [{
                "id": "#bsky_fg",
                "type": "BskyFeedGenerator",
                "serviceEndpoint": f"https://{config.hostname}",
            }],
        })

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "windows_fit": feed.windows_fit,
                             "fitted": feed.fitted, "candidates": len(feed.store.posts)})

    return Starlette(routes=[
        Route("/xrpc/app.bsky.feed.getFeedSkeleton", get_feed_skeleton),
        Route("/xrpc/app.bsky.feed.describeFeedGenerator", describe),
        Route("/.well-known/did.json", did_document),
        Route("/health", health),
    ])
