"""Shared fixtures: the canonical bipolar toy world used across tests.

Ten users in two opinion clusters (0-4 left, 5-9 right) and three posts:
A = universal (all boost), B = partisan-left, C = partisan-right. This is the
smallest world that exercises the keystone claim (§4): a universal post should
out-bridge the partisan posts.
"""
from __future__ import annotations

import numpy as np
import pytest

from chord import ChordConfig, Post, Reaction
from chord.model import MatrixFactorization, fit_divisiveness, ClusterModel


@pytest.fixture
def toy_posts():
    return {"A": Post("A", "auth1"), "B": Post("B", "auth2"), "C": Post("C", "auth3")}


@pytest.fixture
def toy_reactions():
    rx = []
    for u in range(10):
        left = u < 5
        rx.append(Reaction(u, "A", 1.0, timestamp=float(u)))
        rx.append(Reaction(u, "B", 1.0 if left else -1.0, timestamp=float(u)))
        rx.append(Reaction(u, "C", -1.0 if left else 1.0, timestamp=float(u)))
    return rx


@pytest.fixture
def toy_config():
    return ChordConfig(d=4, mf_iters=50, n_clusters=2)


@pytest.fixture
def toy_assignments():
    return {u: (0 if u < 5 else 1) for u in range(10)}


@pytest.fixture
def fitted(toy_reactions, toy_posts, toy_config):
    mf = MatrixFactorization(toy_config, seed=1)
    return mf.fit(toy_reactions, toy_posts)


@pytest.fixture
def cluster_model(fitted, toy_assignments):
    return ClusterModel.from_factorization(fitted, toy_assignments)
