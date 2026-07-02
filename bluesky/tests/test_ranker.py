"""Offline end-to-end: ingest a synthetic bipolar world → fit → serve. No network."""
import dataclasses

from chord.types import Post, Reaction, ReactionKind

from bluesky.config import BlueskyConfig
from bluesky.ranker import ChordFeed

AU, AL, AR = "did:plc:auth_u", "did:plc:auth_l", "did:plc:auth_r"
U = "at://did:plc:auth_u/app.bsky.feed.post/u"   # universal (bridging)
L = "at://did:plc:auth_l/app.bsky.feed.post/l"   # left-partisan
R = "at://did:plc:auth_r/app.bsky.feed.post/r"   # right-partisan


def _feed(seed=0):
    cfg = BlueskyConfig(window_seconds=100.0, default_slots=10)
    cfg.default_knobs = dataclasses.replace(cfg.default_knobs, M=1.0)  # pure bridging
    feed = ChordFeed(cfg, seed=seed)
    for pid, author in ((U, AU), (L, AL), (R, AR)):
        feed.ingest(post=Post(id=pid, author_id=author, created_at=1.0), now=1.0)
    # filler posts so the candidate set exceeds the slot count (a real feed always has
    # more candidates than slots), leaving a fresh pool for the ε-exploration slice.
    for j in range(12):
        fid = f"at://did:plc:filler{j}/app.bsky.feed.post/f{j}"
        feed.ingest(post=Post(id=fid, author_id=f"did:plc:filler{j}", created_at=1.0), now=1.0)
        for u in (j % 10, (j + 3) % 10):                       # a little scattered support
            feed.ingest(reaction=Reaction(f"did:plc:user{u}", fid, 0.5,
                                          ReactionKind.FAVORITE, 2.0), now=2.0)
    # 10 users, two poles; everyone likes U, each pole likes only its own partisan post,
    # and is exposed-without-reacting to the other pole's (the §4.1 weak negative).
    for u in range(10):
        left = u < 5
        did = f"did:plc:user{u}"
        feed.ingest(reaction=Reaction(did, U, 0.5, ReactionKind.FAVORITE, 2.0), now=2.0)
        mine, theirs = (L, R) if left else (R, L)
        feed.ingest(reaction=Reaction(did, mine, 0.5, ReactionKind.FAVORITE, 2.0), now=2.0)
        feed.ingest(reaction=Reaction(did, theirs, -0.1,
                                      ReactionKind.EXPOSED_NO_REACTION, 2.0), now=2.0)
    return feed


def test_fit_then_serve_returns_a_valid_skeleton():
    feed = _feed()
    feed.fit(now=200.0)
    assert feed.fitted and feed.windows_fit == 1
    skeleton = feed.serve("did:plc:viewer", limit=10, now=201.0)
    assert skeleton, "feed should be non-empty"
    assert all(uri.startswith("at://") for uri in skeleton)   # valid AT-URIs
    # pure-bridging feed puts the universally-liked post first (above the partisan pair,
    # which is demoted below the 10-slot cutoff)
    assert skeleton[0] == U
    assert L not in skeleton[:1] and R not in skeleton[:1]


def test_cold_start_serves_chronological_before_first_fit():
    feed = _feed()
    skeleton = feed.serve("did:plc:viewer", limit=10, now=5.0)   # never fit
    assert 0 < len(skeleton) <= 10                               # chronological candidates
    assert len(skeleton) == len(set(skeleton))                   # no dupes


def test_serving_logs_exploration_and_derives_exposed_no_reaction():
    feed = _feed()
    feed.fit(now=200.0)
    # serve to several viewers so the ε slice fires and pairs get logged
    for i in range(6):
        feed.serve(f"did:plc:v{i}", limit=10, now=201.0)
    served = feed.store._served
    assert served, "serving must log exposures (the logged policy π)"
    assert any(e.source.name == "EXPLORATION" and e.propensity and e.propensity > 0
               for e in served), "the ε anchor must be logged with known propensity"
    # a served (viewer, post) that no one liked becomes an exposed-no-reaction weak negative
    reactions, exposures, _ = feed.store.build_window()
    assert exposures, "served skeletons are the window's exposures"
    assert any(r.kind is ReactionKind.EXPOSED_NO_REACTION for r in reactions)


def test_budget_binds_to_did_identity_and_forge_cost_grows_with_age():
    feed = _feed()
    feed.identity.now = 100.0
    feed.identity.observe("did:plc:old", 0.0)
    feed.identity.observe("did:plc:new", 99.0)
    assert feed.identity.forge_cost("did:plc:old") > feed.identity.forge_cost("did:plc:new")
    assert feed.identity.forge_cost("did:plc:unseen") == 0.0
