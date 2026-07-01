"""Weighted, biased matrix factorization — the relation model (§4.1).

Fits

    r_hat_up = mu + b_u + b_a(p) + b_p + <x_u, y_p>

to the signed reaction matrix by weighted alternating least squares (ALS), the
block-convex, closed-form-per-block fit of §9.1. Each observation carries a
weight ``omega_up`` (§6.2) combining rater influence ``lambda_u``, the clipped
inverse propensity, and the silent-disagreement discount.

Design notes tied to the paper:

* The **author term** ``b_a(p)`` is essential and easy to omit (§4.1): without
  it a famous account's blanket elevation leaks into ``b_p``. We fit it with
  partial pooling (a ridge prior, ``tau_a^-2 = reg_bias_author``) so ``b_p``
  measures a post's *marginal* breadth above its author's baseline.
* Partial pooling on ``b_p`` (``tau_p^-2 = reg_bias_post``) likewise keeps the
  post intercept honest.
* The fit is agnostic to *where* the weights come from; §6's propensity layer and
  §5's rater weighting produce them upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import ChordConfig
from ..types import Id, Post, Reaction


@dataclass
class FactorizationResult:
    """Fitted parameters of the relation model, keyed by original ids."""

    mu: float
    b_user: Dict[Id, float]
    b_author: Dict[Id, float]
    b_post: Dict[Id, float]
    x_user: Dict[Id, np.ndarray]  # opinion embedding x_u in R^d
    y_post: Dict[Id, np.ndarray]  # opinion loading y_p in R^d
    d: int
    # Diagnostics
    weighted_rmse: float = float("nan")
    n_iter: int = 0

    def predict(self, user_id: Id, post_id: Id, author_id: Id) -> float:
        """Reconstruct r_hat_up for a (user, post) pair."""
        pred = self.mu
        pred += self.b_user.get(user_id, 0.0)
        pred += self.b_author.get(author_id, 0.0)
        pred += self.b_post.get(post_id, 0.0)
        x = self.x_user.get(user_id)
        y = self.y_post.get(post_id)
        if x is not None and y is not None:
            pred += float(np.dot(x, y))
        return pred


class MatrixFactorization:
    """Weighted biased MF fitted by ALS (§4.1, §9.1)."""

    def __init__(self, config: ChordConfig, seed: int = 0):
        self.config = config
        self.d = config.d
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        reactions: Sequence[Reaction],
        posts: Dict[Id, Post],
        weights: Sequence[float] | None = None,
    ) -> FactorizationResult:
        """Fit the model.

        Parameters
        ----------
        reactions : signed reaction events (§4.1). Duplicate (u,p) pairs are
            allowed and simply add observations.
        posts : mapping post id -> Post, used for the author term b_a(p).
        weights : per-observation omega_up (§6.2). If None, all weights are 1.
        """
        n = len(reactions)
        if n == 0:
            raise ValueError("cannot fit on an empty reaction set")
        if weights is None:
            weights = np.ones(n, dtype=float)
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must align 1:1 with reactions")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

        # --- index the entities into contiguous arrays ---
        users = _Index()
        items = _Index()
        authors = _Index()
        u_idx = np.empty(n, dtype=np.int64)
        p_idx = np.empty(n, dtype=np.int64)
        a_idx = np.empty(n, dtype=np.int64)
        r = np.empty(n, dtype=float)
        post_author_of_item: Dict[int, int] = {}
        for i, rx in enumerate(reactions):
            post = posts.get(rx.post_id)
            if post is None:
                raise KeyError(f"reaction references unknown post {rx.post_id!r}")
            ui = users.add(rx.user_id)
            pi = items.add(rx.post_id)
            ai = authors.add(post.author_id)
            u_idx[i], p_idx[i], a_idx[i] = ui, pi, ai
            r[i] = rx.value
            post_author_of_item[pi] = ai

        nu, npost, na = len(users), len(items), len(authors)

        # group observation indices by user / post for the embedding solves
        by_user = _group(u_idx, nu)
        by_post = _group(p_idx, npost)
        by_author = _group(a_idx, na)

        # --- parameters ---
        cfg = self.config
        mu = float(np.average(r, weights=w) if w.sum() > 0 else r.mean())
        b_u = np.zeros(nu)
        b_a = np.zeros(na)
        b_p = np.zeros(npost)
        scale = 0.1 / np.sqrt(self.d)
        X = self._rng.normal(0.0, scale, size=(nu, self.d))
        Y = self._rng.normal(0.0, scale, size=(npost, self.d))

        prev_rmse = np.inf
        n_iter = 0
        for sweep in range(cfg.mf_iters):
            n_iter = sweep + 1
            pred = _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx)

            # -- mu (free global bias) --
            resid_ex = r - (pred - mu)
            wsum = w.sum()
            mu = float(np.dot(w, resid_ex) / wsum) if wsum > 0 else mu
            pred = _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx)

            # -- user biases b_u (ridge) --
            resid_ex = r - (pred - b_u[u_idx])
            b_u = _weighted_group_mean(resid_ex, w, by_user, nu, cfg.reg_bias_user)
            pred = _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx)

            # -- author baselines b_a (partial pooling tau_a) --
            resid_ex = r - (pred - b_a[a_idx])
            b_a = _weighted_group_mean(resid_ex, w, by_author, na, cfg.reg_bias_author)
            pred = _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx)

            # -- post intercepts b_p (partial pooling tau_p) --
            resid_ex = r - (pred - b_p[p_idx])
            b_p = _weighted_group_mean(resid_ex, w, by_post, npost, cfg.reg_bias_post)
            pred = _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx)

            # -- user embeddings x_u (ridge regression on Y) --
            base = mu + b_u[u_idx] + b_a[a_idx] + b_p[p_idx]
            target = r - base
            X = _solve_embeddings(target, w, u_idx, p_idx, Y, by_user, nu,
                                  self.d, cfg.reg_embedding)

            # -- post loadings y_p (ridge regression on X) --
            # ``target`` (= r - base) is unchanged by the x_u solve because base
            # excludes the dot-product term, so we reuse it.
            Y = _solve_embeddings(target, w, p_idx, u_idx, X, by_post, npost,
                                  self.d, cfg.reg_embedding)

            pred = _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx)
            rmse = _weighted_rmse(r, pred, w)
            if abs(prev_rmse - rmse) < cfg.mf_tol:
                break
            prev_rmse = rmse

        # --- pack results back into id-keyed dicts ---
        result = FactorizationResult(
            mu=mu,
            b_user={users.key(i): float(b_u[i]) for i in range(nu)},
            b_author={authors.key(i): float(b_a[i]) for i in range(na)},
            b_post={items.key(i): float(b_p[i]) for i in range(npost)},
            x_user={users.key(i): X[i].copy() for i in range(nu)},
            y_post={items.key(i): Y[i].copy() for i in range(npost)},
            d=self.d,
            weighted_rmse=_weighted_rmse(
                r, _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx), w
            ),
            n_iter=n_iter,
        )
        return result


# ---------------------------------------------------------------- helpers
class _Index:
    """Bijection between opaque ids and contiguous integer indices."""

    def __init__(self) -> None:
        self._to_idx: Dict[Id, int] = {}
        self._keys: List[Id] = []

    def add(self, key: Id) -> int:
        idx = self._to_idx.get(key)
        if idx is None:
            idx = len(self._keys)
            self._to_idx[key] = idx
            self._keys.append(key)
        return idx

    def key(self, idx: int) -> Id:
        return self._keys[idx]

    def __len__(self) -> int:
        return len(self._keys)


def _group(idx: np.ndarray, n_groups: int) -> List[np.ndarray]:
    """List of observation-index arrays, one per group id."""
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    for obs_i, g in enumerate(idx):
        groups[g].append(obs_i)
    return [np.asarray(g, dtype=np.int64) for g in groups]


def _predict_all(mu, b_u, b_a, b_p, X, Y, u_idx, p_idx, a_idx) -> np.ndarray:
    dots = np.einsum("ij,ij->i", X[u_idx], Y[p_idx])
    return mu + b_u[u_idx] + b_a[a_idx] + b_p[p_idx] + dots


def _weighted_group_mean(resid_ex, w, groups, n_groups, reg) -> np.ndarray:
    """Closed-form ridge-regularized weighted mean per group.

    b_k = sum_i w_i resid_ex_i / (sum_i w_i + reg)
    """
    out = np.zeros(n_groups)
    for k, obs in enumerate(groups):
        if obs.size == 0:
            continue
        wk = w[obs]
        denom = wk.sum() + reg
        if denom <= 0:
            continue
        out[k] = float(np.dot(wk, resid_ex[obs]) / denom)
    return out


def _solve_embeddings(target, w, primary_idx, other_idx, OTHER, groups,
                      n_groups, d, reg) -> np.ndarray:
    """Ridge-regress each primary entity's embedding on the other factor.

    For entity k:  v_k = (sum w_i o_i o_i^T + reg I)^-1 (sum w_i o_i t_i)
    where o_i are the *other* factor rows and t_i the residual targets.
    """
    out = np.zeros((n_groups, d))
    reg_eye = reg * np.eye(d)
    for k, obs in enumerate(groups):
        if obs.size == 0:
            continue
        O = OTHER[other_idx[obs]]          # (m, d)
        wk = w[obs]                        # (m,)
        tk = target[obs]                   # (m,)
        A = O.T @ (O * wk[:, None]) + reg_eye
        b = O.T @ (wk * tk)
        try:
            out[k] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            out[k] = np.linalg.lstsq(A, b, rcond=None)[0]
    return out


def _weighted_rmse(r, pred, w) -> float:
    wsum = w.sum()
    if wsum <= 0:
        return float("nan")
    return float(np.sqrt(np.dot(w, (r - pred) ** 2) / wsum))
