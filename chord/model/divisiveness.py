"""Divisiveness and the divide-weighting matrix A (§4.1).

Divisiveness is the population spread of the alignment term:

    D(p) = y_p^T A y_p

With whitened embeddings and A = I this is just ||y_p||^2 — universal posts sit
near the origin of opinion space, partisan posts sit far out. But A = I is "too
glib": it penalizes a fishing-vs-libraries split as much as a political fault
line. So we take A >= 0 weighted toward the axes that correlate with an
*affective-polarization signal* (regress the signal on embedding dimensions).
This is where the instance's "which divides do we care to bridge" decision (the
rho knob, §7) physically lives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from ..config import ChordConfig
from ..types import Id
from .factorization import FactorizationResult


@dataclass
class DivisivenessModel:
    """Whitening transform + the divide-weighting matrix A (§4.1)."""

    mean: np.ndarray          # opinion-space centroid removed before whitening
    whitener: np.ndarray      # (d,d) maps raw embeddings -> whitened coords
    A: np.ndarray             # (d,d) PSD divide-weighting in whitened coords
    d: int

    def whiten(self, y: np.ndarray) -> np.ndarray:
        return self.whitener @ (np.asarray(y, dtype=float) - self.mean)

    def divisiveness(self, y_post: np.ndarray, rho: float = 1.0) -> float:
        """D(p) = y^T A y in whitened coordinates, scaled by rho.

        rho in [0,1] scales A (§12): rho=0 means "no divide is worth penalizing".
        """
        yw = self.whiten(y_post)
        return float(rho * (yw @ self.A @ yw))

    def alignment(self, x_user: np.ndarray, y_post: np.ndarray) -> float:
        """The partisan appeal term <x_u, y_p> in whitened coordinates.

        Whitening is applied symmetrically so the inner product is preserved up
        to the shared linear map; we use raw embeddings for personalization in
        the value model and reserve whitening for the divisiveness quadratic.
        """
        return float(np.dot(x_user, y_post))


def fit_divisiveness(
    result: FactorizationResult,
    config: ChordConfig,
    affective_signal: Optional[Dict[Id, float]] = None,
) -> DivisivenessModel:
    """Whiten the post loadings and build A (§4.1).

    Parameters
    ----------
    result : the fitted factorization.
    affective_signal : optional per-post scalar correlating with affective
        polarization (e.g. a toxicity / hostility measure). When provided and
        ``config.affective_weighting`` is set, A is built by regressing this
        signal onto the whitened embedding dimensions, so axes that carry
        genuine affective divides are weighted up and benign splits are not.
        When absent, A = I (the glib but serviceable default).
    """
    ids = list(result.y_post.keys())
    if not ids:
        eye = np.eye(result.d)
        return DivisivenessModel(np.zeros(result.d), eye, eye, result.d)

    Y = np.stack([result.y_post[i] for i in ids])  # (n, d)
    mean = Y.mean(axis=0)
    Yc = Y - mean

    # --- whitening: make cov(Yw) = I so ||y||^2 is comparable across axes ---
    cov = np.cov(Yc, rowvar=False)
    cov = np.atleast_2d(cov)
    # symmetric inverse square root via eigendecomposition (PSD, regularized)
    eigvals, eigvecs = np.linalg.eigh(cov + 1e-8 * np.eye(result.d))
    eigvals = np.clip(eigvals, 1e-8, None)
    whitener = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    Yw = Yc @ whitener.T

    # --- A ---
    if not config.affective_weighting or affective_signal is None:
        A = np.eye(result.d)
    else:
        # Regress the affective signal on whitened dims; weight axes by squared
        # standardized coefficient so A is PSD and emphasizes divisive axes.
        s = np.array([affective_signal.get(i, np.nan) for i in ids], dtype=float)
        mask = ~np.isnan(s)
        if mask.sum() < 2 or np.allclose(s[mask], s[mask][0]):
            A = np.eye(result.d)
        else:
            Ym = Yw[mask]
            sm = s[mask] - s[mask].mean()
            # ridge regression coefficients
            G = Ym.T @ Ym + 1e-6 * np.eye(result.d)
            coef = np.linalg.solve(G, Ym.T @ sm)
            # A = I + coef coef^T scaled so the divisive axis dominates but no
            # axis is zeroed (keeps A strictly PSD and full-rank).
            cc = coef / (np.linalg.norm(coef) + 1e-12)
            weight = float(np.clip(np.linalg.norm(coef), 0.0, 10.0))
            A = np.eye(result.d) + weight * np.outer(cc, cc)

    return DivisivenessModel(mean=mean, whitener=whitener, A=A, d=result.d)
