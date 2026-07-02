# CHORD as a Bluesky feed generator

A real [ATProto](https://atproto.com) **feed generator** that ranks with CHORD,
built strictly *on top of* the pure `chord` core (this package never modifies it —
it implements the core's ports §3 and adds the network plumbing the core
deliberately omits). It ingests the firehose, runs CHORD's learning + serving loop,
and answers `app.bsky.feed.getFeedSkeleton` so any Bluesky client can subscribe to a
bridging feed.

```
Jetstream firehose ──▶ mapping ──▶ RollingStore ──▶ Chord.fit_window  (every window)
(posts, likes,          (§4.1)      (candidates,      (learning plane, §9.1)
 reposts)                            reactions,              │
                                     served exposures)       ▼
getFeedSkeleton  ◀──  ChordFeed.serve  ◀────────────  Chord.rank  (per request)
(a list of post URIs)   (+ logs its own served              (serving plane, §9.1)
                         skeletons as exposures)
```

## Why a feed generator is the *right* home for CHORD

CHORD's identifiability (§6.2) needs exposures logged with **known propensity** and
a floored **ε-exploration** anchor. A feed generator is *exactly* a logging policy:
it chooses what each user sees. So `ChordFeed` logs its own served skeletons as
`Exposure`s — the ε slice flagged `EXPLORATION` with propensity `ε`, the rest
`ORGANIC` — and that logged policy is precisely what the IPW layer (§6) corrects
against. The self-logging closes a loop that a pure downstream re-ranker could not.

## The Bluesky → CHORD mapping

| Bluesky | CHORD (`chord.types`) | Notes |
|---|---|---|
| DID | user / author `Id`; Identity port | budgets (§8) & Sybil defenses (§5) bind to the DID; `forge_cost` grows with observed account age |
| post AT-URI | post `Id` | `at://<did>/app.bsky.feed.post/<rkey>` |
| like | `FAVORITE` reaction (+0.5) | signed approval channel (§4.1) |
| repost | `BOOST` reaction (+1.0) | |
| *served but not liked* | `EXPOSED_NO_REACTION` (−c) | the §4.1 weak negative — sourced from **our own** served skeletons (the only place impression data exists) |
| our served skeleton | `Exposure` (known π; ε slice = `EXPLORATION`) | the logged policy §6.2 assumes |
| *(no native signal)* | `VOUCH` / depth (§10) | see below |

Two signals Bluesky does not hand you, and how this handles them:

- **No impression data.** Feed generators don't get "who saw what." Solved by the
  self-logging above — we only need π for what *we* served, which we know exactly.
  Likes on posts we didn't serve still arrive from the firehose and inform the
  factorization, but as *out-of-band* reactions (E5a §13#10 down-weights them for the
  authority signal).
- **No merit / vouch channel.** Bluesky has no "this is substantive, regardless of
  whether I agree" vote. It is a plug — `ranker.VouchSource` — defaulting to none. A
  host wires a real one (a custom `app.chord.vouch` lexicon record, a trusted labeler,
  or a model score). With none, the (default-on) quality-based E9 prior still *lowers*
  an untested firehose's B_LCB but cannot *raise* on unearned merit, and depth stays
  neutral — both the safe side. This is the main integration gap; see the whitepaper
  §10/§13.11.

## Run it

```bash
pip install -e '.[bluesky]'           # starlette, uvicorn, websockets, httpx
python -m pytest bluesky/ -q          # offline tests (no network)
```

To run live you need a **public HTTPS host** (the service identity is
`did:web:<hostname>`, so the host must serve `/.well-known/did.json` over TLS):

```bash
export FEEDGEN_HOSTNAME=feed.example.com      # your public host
export FEEDGEN_PUBLISHER_DID=did:plc:you       # the account that will own the feed
export BLUESKY_HANDLE=you.bsky.social
export BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx # an app password, not your login

python -m bluesky publish                       # one-time: register the feed record
python -m bluesky serve --port 3000             # ingest firehose + serve getFeedSkeleton
```

Put the process behind your TLS terminator on `FEEDGEN_HOSTNAME`, and the feed
appears in-app for anyone who opens its `at://…/app.bsky.feed.generator/chord` URI.
`GET /health` reports windows fit / candidate count.

## How it works in practice

**Data collection and subscription are decoupled.** Ingestion reads the **Jetstream
firehose** — a global, public, no-auth stream of *every* post/like/repost on the whole
network. The instant `serve` connects it is collecting data, whether or not anyone has
subscribed. Subscription only controls *who sees the output*: a user subscribes by opening
the feed's `at://…/app.bsky.feed.generator/<rkey>` URI in the app (share
`bsky.app/profile/<you>/feed/<rkey>`), and *their* refresh is what calls your
`getFeedSkeleton`. So it runs and learns with **zero subscribers**.

**The "does it need mass?" question has two layers:**

1. *The learning machinery runs from day one.* The factorization, opinion clusters, `B_LCB`,
   and λ are computed from firehose likes/reposts — the whole network's reactions — and the
   candidate set fills immediately (a post being **actively liked** becomes a candidate, its
   author DID read from the like's target URI, via `candidates_from_likes`; a fresh post has no
   likes yet, so ranking only created posts would be starved).
2. *But bridging can't discriminate on likes alone — and that plus de-confounding is what the
   feed's own serving fixes.* This is the subtle, load-bearing point, and running it live made
   it concrete: **the firehose is all-positive** (Bluesky has likes/reposts, no "dislike"), so
   every liked post's reception is ≈ +0.5 and `B_LCB` comes out **flat** — in an 18k-event live
   sample, `B_LCB` std ≈ **0.013**, essentially no signal to rank on. Bridging needs *contrast*
   — posts some clusters embrace and others don't — and that negative signal is the
   **exposed-no-reaction** weak negative (§4.1), which exists *only* once the feed serves a post
   and observes non-reaction. The same served exposures also carry the known propensity and the
   ε anchor that de-confound the selection bias (§6.2). So the feed's own serving traffic isn't a
   nicety for de-confounding; **it is what makes bridging possible at all** on a like-only
   network. The code degrades gracefully across this line — no served exposures ⇒ `fit_window`
   still runs, just on flat, positive-only signal (near-uniform ranking) until serving begins.

So the path is: **firehose-only cold start** (candidates fill, but `B_LCB` is flat — the feed is
essentially recency/exploration) → **a handful of subscribers** (its served skeletons start
producing exposed-no-reaction negatives + the ε anchor) → **bridging + de-confounded steady
state**. The corollary: seed *some* serving traffic early (even a few pinned subscribers, or a
scripted ε-serving warm-up) — that, not raw firehose volume, is what turns bridging on.

### Will a dev run drown in the firehose?

**Yes, if you ingest everything** — not from the websocket (that's cheap) but from the *store
size and `fit_window` cost*: a 15-minute slice of the global firehose is hundreds of thousands
of posts and millions of likes, and the numpy/scipy core is sized for a fediverse instance, not
the whole network in one window. **Scope it.** Three levers (`BlueskyConfig`), use any:

- `wanted_dids=[…]` — Jetstream filters *server-side* to ≤10k repos (scope to a community/topic);
- `sample_rate=0.02` — keep a deterministic fraction of the firehose, bucketed by actor DID so a
  kept account's posts and likes stay together (a coherent thinned graph, not orphaned events);
- `max_posts` + a short `window_seconds` — hard-cap the store and keep windows small.

A local dev run needs **no public host and no publishing** — just watch it rank:

```python
import asyncio, time
from bluesky import ChordFeed, BlueskyConfig
from bluesky.jetstream import run_ingestion

cfg = BlueskyConfig(sample_rate=0.02, window_seconds=120, max_posts=5000)
#   ...or scope to a community instead of sampling: BlueskyConfig(wanted_dids=[...])
feed = ChordFeed(cfg)
asyncio.run(run_ingestion(feed, cfg, max_events=50_000))     # ingest a bounded burst
print(feed.serve("did:plc:you", limit=30, now=time.time()))  # inspect the ranked skeleton
```

Start heavily thinned (`sample_rate≈0.01–0.05`), confirm the loop keeps up (`/health`), then dial
up. A production feed almost always uses `wanted_dids`/a candidate filter rather than the raw
global stream anyway.

## Status — what is real vs. a slot

**Real:** the full ingest→learn→serve loop, faithful known-π exposure logging with a
floored ε anchor, the DID identity port, the three ATProto HTTP endpoints, the
firehose consumer, and feed-record publishing. Offline-tested end to end.

**Slots / v1 simplifications (documented, not hidden):**
- `VouchSource` defaults to none (the merit channel — the main gap).
- The service-auth JWT is decoded but **not signature-verified** (enough to
  personalize; production should verify against the issuer's signing key).
- In-memory rolling window (no cross-restart persistence); a DB-backed store is a
  drop-in for `RollingStore`.
- `getFeedSkeleton` returns a single page (no cursor pagination yet).

These are deliberate boundaries, each a clean extension point — the ranking core and
its guarantees are the finished part; this package is the honest, runnable adapter.

## Next steps

### Implement a `VouchSource` (the merit channel — the highest-value gap)

The vouch channel (§10) is what lets the (default-on) quality-based E9 prior *promote*
demonstrated merit and lets depth *gate* bridging-bait — with the default `NoVouchSource`
CHORD only ever demotes on it, never earns. A real `VouchSource` closes that. The contract
is one method:

```python
class VouchSource(Protocol):
    def vouches(self, posts: Sequence[Post]) -> Sequence[Reaction]: ...   # kind=VOUCH, ±1
```

and it wires in at construction: `ChordFeed(config, vouch_source=MyVouchSource())`.

**Recommended: a custom `app.chord.vouch` lexicon record.** Define a small ATProto record
type — a merit vote on a post (`{subject: {uri, cid}, value: +1|-1, createdAt}`) — and let a
companion client (or a button in an existing one) write it to the voter's own repo. Then:

1. add `app.chord.vouch` to `BlueskyConfig.wanted_collections` so Jetstream delivers it;
2. add an `event_to_vouch(event)` mapper (mirror `event_to_reaction`, emitting
   `ReactionKind.VOUCH`) and route it via `ChordFeed.ingest`/`store.add_vouch`;
3. that's it — the vouch flows through the *same* λ-weighting, out-diversity, and loyalty
   machinery as the approval channel, which is exactly what the forged-vouch adversary test
   (`tests/test_simulator_adversary.py::test_forged_vouches_buy_no_reach_under_the_quality_prior`)
   showed already contains a ring that fakes merit votes. The vouch is a *record in the
   voter's repo*, so it inherits the DID's Sybil cost — a fresh puppet's vouch counts ≈0.

The record lives in the voter's own repo (portable, revocable, not owned by the feed), which
is the ATProto-native way to keep merit an *earned, other-attested* quantity (§12 wall) rather
than an author-set feature a baiter could forge.

**Alternatives** the same slot accepts, in decreasing forge-resistance:
- a **trusted labeler** (an `app.bsky.labeler`) whose "substantive" labels map to vouches;
- a **model score** (e.g. an LLM rating of depth/effort) as a soft, lower-trust vouch — cheap
  to run but game-able, so weight it well below human/record vouches.

### Smaller follow-ons
- **Verify the service-auth JWT** against the issuer's signing key (resolve their DID doc)
  before trusting the `iss` DID for per-user personalization.
- **Persist the store** — a DB-backed `RollingStore` (SQLite/Postgres) so windows and the
  E9/budget history survive restarts.
- **Cursor pagination** in `getFeedSkeleton` for deep scroll.
