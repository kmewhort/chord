# `validate/` — retroactive validation on real datasets (Appendix C)

A second, **opt-in** test suite alongside `tests/`. Where `tests/` proves the code
is internally correct on synthetic worlds, `validate/` asks the harder question the
whitepaper's Appendix C poses: **do CHORD's claims hold up on real public data?**

The suite is deliberately built to *find where they don't*. Claims that survive are
plain `assert`s; claims that fail on real data are recorded as documented `xfail`
findings (see [`FINDINGS.md`](FINDINGS.md)) rather than tuned away — a failed claim
is the most useful output here.

## Quick start

```bash
pip install -e '.[validate]'        # adds pandas, pyarrow, requests
python -m validate.fetch all        # download datasets into validate/data (Git LFS)
pytest validate/ -rxX               # run; -rxX lists findings (xfail) and fixes (xpass)
```

The default `pytest -q` (which is pinned to `testpaths = ["tests"]`) does **not**
run this suite, so the core stays offline and fast. Every test here self-skips if
its dataset is absent, so a partial fetch is fine.

## Datasets (Appendix C.1 layer → dataset mapping)

| Dataset | CHORD layer tested | Fetch size | Committed via LFS |
|---|---|---|---|
| **Coat** | §6 propensity / MNAR — real MAR holdout | 0.5 MB | full |
| **Polis** (openData) | §4 clusters + `B_LCB` on real divides | ~5 MB | full (3 conversations) |
| **MovieLens-100K** | §6 semi-synthetic harness (C.3) on real prefs | 5 MB | full |
| **Wikipedia-RfA** (SNAP) | §5 EigenTrust λ on real signed votes | 15 MB | full |
| **Community Notes** | §4/§5 keystone vs the deployed CN model | ~4.5 GB raw → **0.4 MB slice** | slice only |

Community Notes is GB-scale. `fetch` pulls the raw parquet shards from a public
Hugging Face mirror into `data/community_notes/raw/` (git-ignored) and distills a
compact, dense **k-core slice** into `data/community_notes/slice/` (committed). Pull
more ratings shards with `python -m validate.fetch community_notes --cn-shards 4`.

Datasets that need registration (Yahoo!R3), are TB-scale (Reddit dumps), or whose
public endpoints are now login-gated (the live CN `ton.twimg.com` feed) are **not**
wired up; the adapters/fetchers are structured so they can be slotted in later.

## Layout

```
validate/
  fetch.py               # `python -m validate.fetch <name|all>` — downloads data
  _common.py             # data paths, download helper, skip-guard, record_finding()
  _modeling.py           # thin wiring of reactions → chord MF / clusters / B_LCB / λ
  metrics.py             # NDCG@k, AUC, Spearman, Adjusted Rand Index (no sklearn)
  synthetic.py           # impose a synthetic MNAR logging policy on a real matrix (C.3)
  datasets/              # one adapter per dataset → chord.types (Reaction/Post/Exposure)
  experiments/           # prototype fixes for findings (Sybil-ring, keystone) — core untouched
  test_*.py              # one claim-checking test file per dataset (+ *_hardening/_variants)
  data/                  # Git LFS: committed slices; raw/ subdirs are git-ignored
  FINDINGS.md            # the running list of claims that did NOT pan out
```

## How to read a result

- **PASS** — the whitepaper claim held on this dataset at the asserted threshold.
- **XFAIL** — a documented finding: the claim did not hold. Read the reason (it
  carries the measured numbers) and `FINDINGS.md`.
- **XPASS** — a claim that used to fail now passes. Investigate: either a fix
  landed (tighten the assertion back) or the data/params changed.
- **SKIP** — the dataset isn't present. Run `python -m validate.fetch <name>`.

## Not covered here

The §9 feedback loop cannot be validated on any fixed archive — a static dataset
can't respond to the ranker's own allocations (Appendix C.4). That is exercised by
the agent-based simulator in `tests/test_simulator.py`, not here.
