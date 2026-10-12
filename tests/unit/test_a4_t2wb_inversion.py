"""A4.1 inversion runner: deterministic summaries and cache hashing."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.a4_t2wb_inversion import (
    BOOTSTRAP_RESAMPLES,
    SCHEMES,
    STORAGES,
    SUMMARY_METRICS,
    _sha256_array,
    summarize,
)


def _rows(n: int = 4):
    rows = []
    for scheme_i, scheme in enumerate(SCHEMES):
        for storage_i, storage in enumerate(STORAGES):
            for image_i in range(n):
                value = float(10 * scheme_i + 2 * storage_i + image_i + 1)
                rows.append({
                    "scheme": scheme,
                    "storage": storage,
                    **{metric: value for metric in SUMMARY_METRICS},
                })
    return rows


def test_summary_is_deterministic_and_complete() -> None:
    first = summarize(_rows(), 1911)
    second = summarize(_rows(), 1911)
    assert first == second
    assert len(first) == len(SCHEMES) * len(STORAGES) * len(SUMMARY_METRICS)
    assert all(row["n"] == 4 for row in first)
    assert all(row["ci95_low"] <= row["mean"] <= row["ci95_high"] for row in first)
    assert BOOTSTRAP_RESAMPLES == 10_000


def test_summary_rejects_missing_cell() -> None:
    rows = [
        row for row in _rows()
        if not (row["scheme"] == "S1" and row["storage"] == "legacy_png")
    ]
    with pytest.raises(ValueError, match="missing rows"):
        summarize(rows, 1911)


def test_array_hash_depends_on_float32_bytes() -> None:
    first = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    second = first.copy()
    assert _sha256_array(first) == _sha256_array(second)
    second[0, 0, 0] += np.float32(1)
    assert _sha256_array(first) != _sha256_array(second)
