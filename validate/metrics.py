"""Evaluation metrics used across the validation tests.

Self-contained numpy implementations (no sklearn dependency) of the standard
retroactive-evaluation metrics named in Appendix C.3: unbiased NDCG@k / AUC
against a MAR holdout, plus rank correlation and cluster-agreement measures.
"""
from __future__ import annotations

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean rank), like scipy."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # resolve ties to average rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann–Whitney U statistic. ``labels`` in {0,1}.

    Returns 0.5 (chance) when one class is empty.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = rankdata(scores)
    rank_pos = ranks[labels == 1].sum()
    u = rank_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def dcg(rels: np.ndarray, k: int) -> float:
    rels = np.asarray(rels, dtype=float)[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(pred_scores: np.ndarray, true_rels: np.ndarray, k: int) -> float:
    """NDCG@k: rank items by ``pred_scores``, score gains from ``true_rels``.

    This is the "unbiased NDCG" of C.3 when ``true_rels`` come from the MAR
    holdout — the ranking is judged against relevances the ranker never saw.
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    true_rels = np.asarray(true_rels, dtype=float)
    order = np.argsort(-pred_scores, kind="mergesort")
    ranked = true_rels[order]
    ideal = np.sort(true_rels)[::-1]
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(ranked, k) / idcg


def adjusted_rand_index(labels_a, labels_b) -> float:
    """Adjusted Rand Index between two clusterings (chance-corrected).

    1.0 = identical partitions; ~0.0 = random agreement. Used to check that
    CHORD's recovered opinion clusters (§4.2) match a dataset's own validated
    groups (e.g. Polis group-id).
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    contingency = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for x, y in zip(a, b):
        contingency[ua[x], ub[y]] += 1
    sum_comb_c = np.sum([_comb2(n) for n in contingency.flatten()])
    sum_comb_a = np.sum([_comb2(n) for n in contingency.sum(axis=1)])
    sum_comb_b = np.sum([_comb2(n) for n in contingency.sum(axis=0)])
    n = len(a)
    total = _comb2(n)
    if total == 0:
        return 1.0
    expected = sum_comb_a * sum_comb_b / total
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        return 1.0
    return float((sum_comb_c - expected) / (max_index - expected))


def _comb2(n: float) -> float:
    return n * (n - 1) / 2.0
