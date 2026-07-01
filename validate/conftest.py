"""Pytest fixtures for the retroactive-validation suite.

These tests exercise the real datasets under ``validate/data``. They self-skip
(via :func:`validate._common.require`) when a dataset has not been fetched, so the
suite is safe to run on a machine that only cloned the code.
"""
from __future__ import annotations

import pytest

from chord.config import ChordConfig


@pytest.fixture
def base_config() -> ChordConfig:
    """A modest, deterministic config tuned for the (small) real slices.

    Lower embedding dimension and more ALS sweeps than the library default: the
    public slices are smaller and noisier than a production window, so a tighter
    model with more iterations gives stabler embeddings for the assertions.
    """
    return ChordConfig(
        d=2,
        mf_iters=60,
        reg_embedding=0.08,
        reg_bias_post=0.05,
        n_clusters=2,
        affective_weighting=False,  # no toxicity signal on these datasets → A = I
    )
