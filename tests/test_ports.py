import numpy as np
import pytest

from chord import ChordConfig, Post, UserKnobs
from chord.types import Exposure, ReactionKind
from chord.ports import (
    AccountAgeIdentityAdapter,
    KMeansPartitionAdapter,
    LocalPreferenceAdapter,
    NativeSignalAdapter,
    TimelineCandidateAdapter,
)


def test_identity_maps_aliases_and_forge_cost():
    idp = AccountAgeIdentityAdapter(ages={"old": 100.0, "new": 0.0},
                                    aliases={"puppet": "human"})
    assert idp.identity_of("puppet") == "human"
    assert idp.identity_of("standalone") == "standalone"
    assert idp.forge_cost("old") > idp.forge_cost("new")  # older = costlier to forge


def test_preference_store_roundtrip():
    pref = LocalPreferenceAdapter()
    k = UserKnobs(M=0.3)
    pref.set_knobs("u", k)
    assert pref.knobs("u").M == 0.3
    # default for unknown user
    assert isinstance(pref.knobs("stranger"), UserKnobs)


def test_preference_validates_on_set():
    pref = LocalPreferenceAdapter()
    with pytest.raises(ValueError):
        pref.set_knobs("u", UserKnobs(M=5.0))


def test_native_signal_scales_exposed_no_reaction():
    cfg = ChordConfig(exposed_no_reaction_c=0.1)
    sig = NativeSignalAdapter(cfg)
    sig.record_reaction("u", "p", ReactionKind.BOOST)
    sig.record_reaction("u", "q", ReactionKind.EXPOSED_NO_REACTION)
    sig.record_reaction("u", "r", ReactionKind.MUTE)
    vals = {r.post_id: r.value for r in sig.reactions()}
    assert vals["p"] == 1.0
    assert vals["q"] == -0.1  # weak negative scaled by c
    assert vals["r"] == -1.0


def test_native_signal_exposures():
    sig = NativeSignalAdapter(ChordConfig())
    sig.record_exposure(Exposure("u", "p", propensity=0.5))
    assert len(sig.exposures()) == 1


def test_candidate_timeline_merges_shared():
    cand = TimelineCandidateAdapter(
        timelines={"u": [Post("a", "x")]}, shared=[Post("b", "y")]
    )
    ids = {p.id for p in cand.candidates("u")}
    assert ids == {"a", "b"}


def test_kmeans_partition_separates_poles():
    # Two clearly separated groups -> two clusters.
    emb = {}
    for i in range(5):
        emb[i] = np.array([5.0, 0.0]) + np.random.default_rng(i).normal(0, 0.1, 2)
    for i in range(5, 10):
        emb[i] = np.array([-5.0, 0.0]) + np.random.default_rng(i).normal(0, 0.1, 2)
    part = KMeansPartitionAdapter(n_clusters=2, seed=0)
    assign = part.assign(list(emb.keys()), emb)
    left = {assign[i] for i in range(5)}
    right = {assign[i] for i in range(5, 10)}
    assert len(left) == 1 and len(right) == 1  # each pole is one cluster
    assert left != right                       # and they differ


def test_kmeans_handles_single_cluster_request():
    emb = {i: np.array([float(i), 0.0]) for i in range(4)}
    part = KMeansPartitionAdapter(n_clusters=1, seed=0)
    assign = part.assign(list(emb.keys()), emb)
    assert set(assign.values()) == {0}


def test_kmeans_empty_input():
    part = KMeansPartitionAdapter(n_clusters=2, seed=0)
    assert part.assign([], {}) == {}
