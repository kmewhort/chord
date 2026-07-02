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
