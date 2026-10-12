"""A4.2 gallery linkage: scorer correctness, controls, Wilson CI, determinism."""

from __future__ import annotations

import numpy as np

from scripts.a4_t2wb_gallery import feature, metrics, ranks_of_truth, wilson_ci


def _gallery(n: int = 40, side: int = 32):
    return np.random.default_rng(0).standard_normal((n, 1, side, side)).astype(np.float32)


def test_exact_query_links_at_top1_for_every_scorer() -> None:
    gallery = _gallery()
    truth = np.arange(len(gallery))
    for scorer in ("abs", "sorted_abs", "norm", "lowfreq_abs"):
        ranks = ranks_of_truth(feature(gallery, scorer), feature(gallery, scorer), truth)
        assert metrics(ranks, len(gallery), np.random.default_rng(1911))["top1"] == 1.0


def test_sign_flip_preserves_abs_but_not_the_latent() -> None:
    gallery = _gallery()
    rng = np.random.default_rng(7)
    signs = rng.choice([-1.0, 1.0], size=gallery.shape).astype(np.float32)
    query = gallery * signs  # S0-like: |query| == |gallery| exactly
    truth = np.arange(len(gallery))
    assert metrics(ranks_of_truth(feature(query, "abs"), feature(gallery, "abs"), truth),
                   len(gallery), np.random.default_rng(1911))["top1"] == 1.0


def test_random_target_is_near_chance() -> None:
    gallery = _gallery()
    neg_truth = np.random.default_rng(1911).permutation(len(gallery))
    ranks = ranks_of_truth(feature(gallery, "abs"), feature(gallery, "abs"), neg_truth)
    top1 = metrics(ranks, len(gallery), np.random.default_rng(1911))["top1"]
    assert top1 <= 3.0 / len(gallery)


def test_metrics_are_deterministic() -> None:
    gallery = _gallery()
    truth = np.arange(len(gallery))
    ranks = ranks_of_truth(feature(gallery, "sorted_abs"), feature(gallery, "sorted_abs"), truth)
    assert metrics(ranks, len(gallery), np.random.default_rng(1911)) == metrics(
        ranks, len(gallery), np.random.default_rng(1911))


def test_wilson_ci_boundary() -> None:
    low, high = wilson_ci(200, 200)
    assert 0.97 < low < 1.0 and high == 1.0
    assert wilson_ci(0, 0) == [0.0, 0.0]
