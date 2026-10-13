"""Focused regression tests for D7 security evidence helpers."""

import numpy as np
import torch

from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for, tamper_rows
from scripts.d7_metrics import feature, rank_metrics, ranks_l2
from src.crypto.sot import transform_vectors


def test_d7_features_and_pessimistic_ties() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    assert np.allclose(feature(values, "norm"), [[5.0], [5.0]])
    assert np.allclose(np.linalg.norm(feature(values, "abs_unit"), axis=1), 1.0)
    truth = np.asarray([0, 1])
    ranks = ranks_l2(feature(values, "norm"), feature(values, "norm"), truth)
    assert ranks.tolist() == [2, 2]
    assert rank_metrics(ranks, 2)["top1"] == 0.0


def test_cross_nonce_learned_transform_positive_control() -> None:
    dimension = 64
    known = torch.eye(dimension, dtype=torch.float64)
    heldout = torch.randn(4, dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(1911))
    nonce_a, nonce_b = nonce_for("unit/A"), nonce_for("unit/B")
    cipher_known = transform_vectors(
        known, FIXTURE_KEY, nonce_a, rounds=1, use_permutation=False,
    )
    learned_q_t = known.T @ cipher_known
    same = transform_vectors(
        heldout, FIXTURE_KEY, nonce_a, rounds=1, use_permutation=False,
    ) @ learned_q_t.T
    cross = transform_vectors(
        heldout, FIXTURE_KEY, nonce_b, rounds=1, use_permutation=False,
    ) @ learned_q_t.T
    assert torch.max(torch.abs(same - heldout)).item() <= 1e-12
    assert torch.sqrt(torch.mean((cross - heldout).square())).item() > 0.5


def test_d7_tamper_matrix_rejects_s2_but_not_s2a() -> None:
    rows = tamper_rows()
    assert all(row["rejected"] for row in rows if row["method"] == "S2")
    s2a = [row for row in rows if row["method"] == "S2a"]
    assert len(s2a) == 1 and not s2a[0]["rejected"]
    assert s2a[0]["output_max_abs_change"] > 0
