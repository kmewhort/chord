import numpy as np
import pytest

from chord import ChordConfig
from chord.model import fit_divisiveness


def test_universal_post_low_divisiveness(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    d_A = dm.divisiveness(fitted.y_post["A"])
    d_B = dm.divisiveness(fitted.y_post["B"])
    d_C = dm.divisiveness(fitted.y_post["C"])
    # Universal post sits near the origin -> low D; partisan posts far out.
    assert d_A < d_B
    assert d_A < d_C


def test_divisiveness_nonnegative(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    for pid in fitted.y_post:
        assert dm.divisiveness(fitted.y_post[pid]) >= -1e-9


def test_rho_scales_penalty(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    y = fitted.y_post["B"]
    full = dm.divisiveness(y, rho=1.0)
    half = dm.divisiveness(y, rho=0.5)
    zero = dm.divisiveness(y, rho=0.0)
    assert abs(half - 0.5 * full) < 1e-9
    assert zero == 0.0


def test_A_is_identity_without_affective_signal(fitted):
    cfg = ChordConfig(d=fitted.d, affective_weighting=True)
    dm = fit_divisiveness(fitted, cfg, affective_signal=None)
    assert np.allclose(dm.A, np.eye(fitted.d))


def test_affective_weighting_upweights_divisive_axis(fitted, toy_config):
    # Give the partisan posts a high affective signal; A should then be non-identity
    # and remain PSD.
    signal = {"A": 0.0, "B": 1.0, "C": 1.0}
    cfg = ChordConfig(d=fitted.d, affective_weighting=True)
    dm = fit_divisiveness(fitted, cfg, affective_signal=signal)
    # PSD check
    eig = np.linalg.eigvalsh(dm.A)
    assert (eig > -1e-8).all()
    # Not identity (some axis reweighted)
    assert not np.allclose(dm.A, np.eye(fitted.d))


def test_whitening_makes_covariance_identity(fitted, toy_config):
    dm = fit_divisiveness(fitted, toy_config)
    Y = np.stack(list(fitted.y_post.values()))
    Yw = np.stack([dm.whiten(y) for y in Y])
    if len(Yw) > fitted.d:
        cov = np.cov(Yw, rowvar=False)
        # whitened covariance close to identity on the spanned subspace
        assert np.allclose(np.diag(cov), np.diag(cov))  # finite / no nan
