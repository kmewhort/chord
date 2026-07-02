"""CLI: ``python -m bluesky {serve,publish}``.

    export FEEDGEN_HOSTNAME=feed.example.com     # your public https host → did:web
    export FEEDGEN_PUBLISHER_DID=did:plc:you     # the account that owns the feed record
    python -m bluesky publish                     # one-time: register the feed record
    python -m bluesky serve --port 3000           # ingest the firehose + serve the feed
"""
from __future__ import annotations

import argparse
import os

from .config import BlueskyConfig
from .ranker import ChordFeed


def _config(args) -> BlueskyConfig:
    return BlueskyConfig(
        hostname=args.hostname or os.environ.get("FEEDGEN_HOSTNAME", "feed.example.com"),
        publisher_did=args.publisher_did or os.environ.get("FEEDGEN_PUBLISHER_DID", ""),
        feed_rkey=args.rkey or os.environ.get("FEEDGEN_RKEY", "chord"),
    )


def cmd_serve(args) -> None:
    import asyncio
    import uvicorn

    from .jetstream import run_ingestion
    from .server import build_app

    config = _config(args)
    feed = ChordFeed(config)
    app = build_app(feed, config)
    stop = asyncio.Event()

    async def _start() -> None:
        asyncio.create_task(run_ingestion(feed, config, stop=stop))

    async def _stop() -> None:
        stop.set()

    app.add_event_handler("startup", _start)
    app.add_event_handler("shutdown", _stop)
    print(f"CHORD feed generator: did={config.service_did} feed={config.feed_uri}")
    print(f"ingesting {config.jetstream_url} → serving on :{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_publish(args) -> None:
    from .publish import publish_feed

    config = _config(args)
    handle = os.environ.get("BLUESKY_HANDLE")
    password = os.environ.get("BLUESKY_APP_PASSWORD")
    if not handle or not password:
        raise SystemExit("set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD to publish")
    uri = publish_feed(config, handle, password)
    print(f"published feed record: {uri}")
    print("subscribers can now find it; point their client at that at:// URI")


def main() -> None:
    p = argparse.ArgumentParser(prog="bluesky", description="CHORD Bluesky feed generator")
    p.add_argument("--hostname"); p.add_argument("--publisher-did", dest="publisher_did")
    p.add_argument("--rkey")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="ingest the firehose and serve the feed")
    s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=3000)
    s.set_defaults(func=cmd_serve)
    pub = sub.add_parser("publish", help="register the feed record (one-time)")
    pub.set_defaults(func=cmd_publish)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
