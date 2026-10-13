"""Shared, deterministic linkage metrics for D7.

All feature calculations use float64 and truth ties are placed last.  These
definitions intentionally match the corrected A3/A4 protocol (AF-028/033).
"""

from __future__ import annotations

from typing import Any

import numpy as np


SEED = 1911
BOOTSTRAP = 10_000
SCORERS = ("abs", "abs_unit", "sorted_abs", "sorted_abs_unit", "norm")


def feature(matrix: np.ndarray, scorer: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).reshape(len(matrix), -1)
    if scorer == "abs":
        return np.abs(values)
    if scorer == "sorted_abs":
        return np.sort(np.abs(values), axis=1)
    if scorer == "norm":
        return np.linalg.norm(values, axis=1, keepdims=True)
    if scorer == "abs_unit":
        base = feature(values, "abs")
        return base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
    if scorer == "sorted_abs_unit":
        base = feature(values, "sorted_abs")
        return base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
    raise ValueError(f"unknown scorer: {scorer}")


def ranks_l2(query: np.ndarray, gallery: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pessimistic ranks: the true match is last among exact ties."""
    if len(query) != len(truth):
        raise ValueError("query/truth length mismatch")
    ranks = np.empty(len(query), dtype=np.int64)
    for index, target in enumerate(truth):
        distances = np.linalg.norm(gallery - query[index], axis=1)
        true_distance = distances[int(target)]
        ranks[index] = int(np.sum(distances < true_distance) + np.sum(distances == true_distance))
    return ranks


def ranks_cosine(query: np.ndarray, gallery: np.ndarray, truth: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64).reshape(len(query), -1)
    g = np.asarray(gallery, dtype=np.float64).reshape(len(gallery), -1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
    ranks = np.empty(len(q), dtype=np.int64)
    for index, target in enumerate(truth):
        similarities = g @ q[index]
        true_similarity = similarities[int(target)]
        ranks[index] = int(
            np.sum(similarities > true_similarity) + np.sum(similarities == true_similarity)
        )
    return ranks


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1:
        raise ValueError("total must be positive")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def rank_metrics(ranks: np.ndarray, gallery_size: int, *, seed: int = SEED) -> dict[str, Any]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if len(ranks) < 1 or np.any(ranks < 1) or np.any(ranks > gallery_size):
        raise ValueError("invalid ranks")
    top1_count = int(np.sum(ranks <= 1))
    top5_count = int(np.sum(ranks <= min(5, gallery_size)))
    reciprocal = 1.0 / ranks
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ranks), size=(BOOTSTRAP, len(ranks)))
    boot = reciprocal[draws].mean(axis=1)
    return {
        "queries": len(ranks), "gallery_size": gallery_size,
        "random_top1": 1.0 / gallery_size,
        "top1": top1_count / len(ranks), "top1_ci95_wilson": list(wilson(top1_count, len(ranks))),
        "top5": top5_count / len(ranks), "top5_ci95_wilson": list(wilson(top5_count, len(ranks))),
        "mAP": float(reciprocal.mean()),
        "mAP_ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "median_rank": float(np.median(ranks)), "max_rank": int(ranks.max()),
    }
