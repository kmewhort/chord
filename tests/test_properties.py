"""Whole-stack invariants & metamorphic properties over random worlds.

Every other test uses hand-picked inputs. These drive the *composed* learning+serving
pipeline (``Chord.fit_window`` then ``Chord.rank``) over many randomly-generated valid
worlds and assert the properties that must hold for *any* input — the kind of
integration/robustness bug example-based tests miss. Dependency-free (a seeded numpy
generator stands in for Hypothesis); each `seed` is an independent random world.
"""
import numpy as np
import pytest

from chord import ChordConfig, UserKnobs
from chord.loop import Chord
from chord.types import Exposure, ExposureSource, Post, Reaction

WORLD_SEEDS = list(range(40))


def _random_world(seed):
    """A random but non-degenerate bipolar world: (reactions, posts, exposures)."""
    rng = np.random.default_rng(seed)
    n_users = int(rng.integers(8, 25))
    n_posts = int(rng.integers(5, 16))
    n_authors = int(rng.integers(2, max(3, n_users // 2)))
    density = float(rng.uniform(0.4, 0.9))

    opinions = rng.normal(0, 0.4, n_users)
    opinions[: n_users // 2] += 1.5           # two poles
    opinions[n_users // 2:] -= 1.5
    post_pol = rng.normal(0, 1.0, n_posts)
    authors = [f"a{int(rng.integers(n_authors))}" for _ in range(n_posts)]
    posts = {f"p{j}": Post(f"p{j}", authors[j],
                           features={"depth": float(rng.uniform(0, 1))})
             for j in range(n_posts)}

    reactions, exposures = [], []
    for u in range(n_users):
        for j in range(n_posts):
            if rng.random() >= density:
                continue
            explore = rng.random() < 0.1
            src = ExposureSource.EXPLORATION if explore else ExposureSource.ORGANIC
            pi = 0.1 if explore else 0.5
            exposures.append(Exposure(f"u{u}", f"p{j}", source=src, propensity=pi))
            v = float(np.tanh(opinions[u] * post_pol[j] + rng.normal(0, 0.5)))
            reactions.append(Reaction(f"u{u}", f"p{j}", v))
    return reactions, posts, exposures


def _cfg(seed):
    rng = np.random.default_rng(seed + 9999)
    return ChordConfig(d=int(rng.integers(2, 4)), n_clusters=int(rng.integers(2, 4)),
                       mf_iters=12, budget_max=float(rng.uniform(4, 12)))


def _fit(seed):
    reactions, posts, exposures = _random_world(seed)
    if not reactions:
        return None
    cfg = _cfg(seed)
    chord = Chord(cfg, seed=seed, inner_iters=2)
    st = chord.fit_window(reactions, posts, exposures)
    return chord, st, posts, cfg


@pytest.mark.parametrize("seed", WORLD_SEEDS)
def test_pipeline_invariants_on_random_worlds(seed):
    fit = _fit(seed)
    if fit is None:
        pytest.skip("empty world")
    chord, st, posts, cfg = fit

    # lambda_eff is a probability distribution
    lam = np.array(list(st.rater_lambda_eff.values()))
    assert np.all(lam >= -1e-9), "negative rater influence"
    assert abs(lam.sum() - 1.0) < 1e-6, f"lambda_eff sums to {lam.sum()}"

    # every scored post has a finite B_LCB; author budgets stay in [0, budget_max]
    assert all(np.isfinite(v) for v in st.bridging.b_lcb.values())
    for a in {p.author_id for p in posts.values()}:
        b = chord.budget.budget(a)
        assert -1e-9 <= b <= cfg.budget_max + 1e-6, f"budget {b} out of [0,{cfg.budget_max}]"

    # rank returns a valid feed for an arbitrary viewer, no crash / NaN / dupes
    feed = chord.rank("u0", list(posts.values()), UserKnobs(), n_slots=5)
    assert len(feed) == len(set(feed)) <= 5
    assert all(pid in posts for pid in feed)


@pytest.mark.parametrize("seed", WORLD_SEEDS[:12])
def test_determinism(seed):
    fit = _fit(seed)
    if fit is None:
        pytest.skip("empty world")
    reactions, posts, exposures = _random_world(seed)
    cfg = _cfg(seed)
    a = Chord(cfg, seed=seed, inner_iters=2).fit_window(reactions, posts, exposures)
    b = Chord(cfg, seed=seed, inner_iters=2).fit_window(reactions, posts, exposures)
    for pid in a.bridging.b_lcb:
        assert a.bridging.b_lcb[pid] == pytest.approx(b.bridging.b_lcb[pid], abs=1e-9)
    assert a.rater_lambda_eff == pytest.approx(b.rater_lambda_eff, abs=1e-9)


def test_permutation_and_order_invariance():
    """FOUND GAP (left failing on purpose): B_LCB rankings are not reproducible across
    input orderings/relabellings.

    Relabelling user/post ids and shuffling reaction order should leave the bridged-
    support *ranking* of posts unchanged — it is the same data. It does not: mean
    Spearman across random worlds is only ~0.72 (some worlds flip entirely). Root cause:
    the MF initializes X/Y randomly *by index*, so reordering entities changes their
    init and the non-convex bilinear ALS lands in a different local optimum. A
    deterministic SVD-based init was prototyped and fixes it (0.72 -> 0.97 Spearman);
    an order-invariant k-means helps too. Both are reverted for now because they shift
    many *tuned* simulator results — a second finding, the init-fragility of those sim
    claims — so applying them needs a dedicated re-validation pass. Deferred.
    """
    from validate.metrics import spearman
    rhos = []
    for seed in WORLD_SEEDS[:16]:
        reactions, posts, exposures = _random_world(seed)
        if not reactions:
            continue
        cfg = _cfg(seed)
        base = Chord(cfg, seed=seed, inner_iters=2).fit_window(reactions, posts, exposures)
        rng = np.random.default_rng(seed + 7)
        umap = {u: f"U{i}" for i, u in enumerate(rng.permutation(
            list({r.user_id for r in reactions})))}
        pmap = {p: f"P{i}" for i, p in enumerate(rng.permutation(list(posts)))}
        amap = {a: f"A{i}" for i, a in enumerate({p.author_id for p in posts.values()})}
        rp = {pmap[pid]: Post(pmap[pid], amap[po.author_id], features=po.features)
              for pid, po in posts.items()}
        order = rng.permutation(len(reactions))
        rr = [Reaction(umap[reactions[i].user_id], pmap[reactions[i].post_id],
                       reactions[i].value) for i in order]
        re = [Exposure(umap[e.user_id], pmap[e.post_id], source=e.source,
                       propensity=e.propensity) for e in exposures]
        perm = Chord(cfg, seed=seed, inner_iters=2).fit_window(rr, rp, re)
        ids = [p for p in base.bridging.b_lcb if np.isfinite(base.bridging.b_lcb[p])]
        a = np.array([base.bridging.b_lcb[p] for p in ids])
        b = np.array([perm.bridging.b_lcb[pmap[p]] for p in ids])
        rhos.append(spearman(a, b))
    mean_rho = float(np.mean(rhos))
    print(f"\n[properties] B_LCB ranking stability under relabel+reorder: "
          f"mean Spearman={mean_rho:.3f} (target >0.95)")
    assert mean_rho > 0.95, (
        f"FOUND GAP: B_LCB rankings not order-invariant (mean Spearman {mean_rho:.3f}); "
        f"random-by-index MF init + non-convex ALS. SVD init fixes it (deferred). See docstring."
    )


@pytest.mark.parametrize("seed", WORLD_SEEDS[:15])
def test_fresh_rater_gets_minimal_influence(seed):
    """A brand-new rater who reacts to a single post must not gain real influence."""
    reactions, posts, exposures = _random_world(seed)
    if not reactions or not posts:
        pytest.skip("empty world")
    pid = next(iter(posts))
    aug = list(reactions) + [Reaction("FRESH", pid, 1.0)]
    aug_exp = list(exposures) + [Exposure("FRESH", pid, source=ExposureSource.ORGANIC,
                                          propensity=0.5)]
    cfg = _cfg(seed)
    st = Chord(cfg, seed=seed, inner_iters=2).fit_window(aug, posts, aug_exp)
    lam = st.rater_lambda_eff
    if "FRESH" not in lam:
        pytest.skip("fresh rater not a rater node")
    others = [v for u, v in lam.items() if u != "FRESH"]
    assert lam["FRESH"] <= np.median(others) + 1e-9, "fresh rater exceeds median influence"
