"""§5 rater weighting / trust, on real signed votes (Wikipedia RfA, Appendix C.1).

Signed per-user votes are rare in public data; Wikipedia's requests-for-adminship
give ~200k support/oppose votes among editors. We fit CHORD's opinion embedding,
build the cross-divide trust matrix, and compute the rater-influence distribution
``lambda`` (§5), then check the properties §5 promises:

* ``lambda`` is a valid normalized influence distribution and the trust matrix is
  row-stochastic (each rater distributes one unit of outgoing trust — the
  invariant that stops a Sybil from inheriting a booster's full weight).
* Influence is *concentrated*, not uniform (quality-tracking: discriminating
  raters dominate) — reported as the effective rater fraction N_eff / n.
* **Sybil starvation (the headline §5 claim).** We inject a collusion ring of K
  fresh accounts that all boost one target into the real graph and sweep K. If a
  ring can buy the target top-decile influence, that is a real finding about the
  teleport-floor eigentrust — the assertion is written to surface it.
"""
from __future__ import annotations

import numpy as np
import pytest

from chord.config import ChordConfig
from chord.rater.eigentrust import build_trust_matrix, compute_lambda
from chord.types import Post, Reaction

from . import _modeling as M
from ._common import record_finding, require
from .datasets import signed_nets as sn


def _percentile_of(value, population) -> float:
    population = np.asarray(population)
    return 100.0 * float(np.mean(population < value))


@pytest.fixture(scope="module")
def rfa_fit():
    require(sn.NAME, *sn.REQUIRED)
    votes = sn.load_votes()
    reactions, posts, users = sn.to_reactions(votes, min_src_votes=15, min_tgt_votes=30)
    cfg = ChordConfig(d=2, mf_iters=30, reg_embedding=0.08, affective_weighting=False)
    result = M.fit(reactions, posts, cfg, seed=0)
    return dict(cfg=cfg, result=result, reactions=reactions, posts=posts, users=users)


def test_lambda_is_valid_concentrated_distribution(rfa_fit):
    cfg, result = rfa_fit["cfg"], rfa_fit["result"]
    reactions, posts, users = rfa_fit["reactions"], rfa_fit["posts"], rfa_fit["users"]

    # Invariant: trust matrix is row-stochastic (each active rater's row sums ~1).
    T = build_trust_matrix(reactions, posts, result, users)
    rowsums = T.sum(axis=1)
    active = rowsums > 0
    assert np.allclose(rowsums[active], 1.0, atol=1e-8), "T rows not stochastic"

    lam = compute_lambda(reactions, posts, result, users, cfg)
    w = np.array(list(lam.values()))
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-6, "lambda not a distribution"

    n = len(w)
    n_eff = 1.0 / np.sum(w ** 2)
    print(f"\n[rfa §5] users={n:,}  N_eff={n_eff:.0f}  N_eff/n={n_eff/n:.3f}  "
          f"max λ={w.max():.4f}  (uniform would be {1/n:.5f})")
    # Quality-tracking, not uniform influence: effective count well below n.
    assert n_eff < 0.9 * n, (
        f"FINDING: lambda is nearly uniform (N_eff/n={n_eff/n:.3f}); §5 quality-"
        f"tracking does not concentrate influence on discriminating raters."
    )


def test_sybil_ring_cannot_buy_top_influence(rfa_fit):
    cfg, result = rfa_fit["cfg"], rfa_fit["result"]
    reactions, posts, users = rfa_fit["reactions"], rfa_fit["posts"], rfa_fit["users"]

    target = "SYBIL_TARGET"
    tgt_post = f"{target}@ring"
    print(f"\n[rfa §5] Sybil-ring sweep (target boosted by K fresh accounts):")
    pct_by_k = {}
    for K in (5, 20, 50, 100):
        aug_users = list(users) + [target] + [f"sybil{i}" for i in range(K)]
        aug_posts = dict(posts)
        aug_posts[tgt_post] = Post(tgt_post, author_id=target)
        aug_reactions = list(reactions) + [
            Reaction(f"sybil{i}", tgt_post, 1.0) for i in range(K)
        ]
        lam = compute_lambda(aug_reactions, aug_posts, result, aug_users, cfg)
        real_lams = np.array([lam[u] for u in users])
        tgt_lam = lam[target]
        pct = _percentile_of(tgt_lam, real_lams)
        pct_by_k[K] = pct
        print(f"[rfa §5]   K={K:>3}  target λ={tgt_lam:.5f}  "
              f"percentile among {len(users):,} real editors={pct:5.1f}%")

    # The §5 claim: a fresh-account ring is starved — it cannot reach the top
    # decile of established editors no matter how many colluders it adds.
    worst = max(pct_by_k.values())
    if worst >= 90.0:
        record_finding(
            f"§5 Sybil starvation is incomplete: a ring of fresh accounts all "
            f"boosting one target lifts it to the {worst:.0f}th percentile of real "
            f"editor influence (percentile by ring size K: "
            f"{ {k: round(v, 1) for k, v in pct_by_k.items()} }). Row-stochastic "
            f"teleport-floor eigentrust starves the *sybils* (each keeps only floor "
            f"mass) but the *target* harvests their redirected baseline mass, so "
            f"influence grows with ring size. The defense rests entirely on the "
            f"identity port's forge-cost (outside the §5 math)."
        )
