"""E2.5 runner: latent-cache integrity checks and deterministic bootstrap summary."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.e2_roundtrip_runner import SCHEMES, SUMMARY_METRICS, bootstrap_summary, load_latent_cache


def make_cache(run_dir: Path, sample_ids: list[str], split_hash: str) -> np.ndarray:
    run_dir.mkdir(parents=True)
    latents = np.random.default_rng(1911).standard_normal((len(sample_ids), 1, 256, 256)).astype(np.float32)
    np.save(run_dir / "latents.npy", latents)
    (run_dir / "manifest.json").write_text(
        json.dumps({"task_id": "E2.3", "run_id": "E2.3_test", "dataset_split_hash": split_hash}),
        encoding="utf-8",
    )
    with (run_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cache_index", "latent_sha256"])
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            digest = hashlib.sha256(np.ascontiguousarray(latents[index]).tobytes()).hexdigest()
            writer.writerow({"sample_id": sample_id, "cache_index": index, "latent_sha256": digest})
    return latents


def split_rows(sample_ids: list[str], split_hash: str = "h" * 64) -> list[dict[str, str]]:
    return [{"sample_id": sample_id, "split_sha256": split_hash} for sample_id in sample_ids]


def test_valid_cache_loads(tmp_path: Path) -> None:
    ids = ["a", "b"]
    latents = make_cache(tmp_path / "run", ids, "h" * 64)
    assert np.array_equal(load_latent_cache(tmp_path / "run", split_rows(ids)), latents)


def test_cache_rejects_split_order_and_tampering(tmp_path: Path) -> None:
    ids = ["a", "b"]
    make_cache(tmp_path / "run", ids, "h" * 64)
    with pytest.raises(ValueError, match="split hash"):
        load_latent_cache(tmp_path / "run", split_rows(ids, "x" * 64))
    with pytest.raises(ValueError, match="order"):
        load_latent_cache(tmp_path / "run", split_rows(["b", "a"]))
    latents = np.load(tmp_path / "run" / "latents.npy")
    latents[1, 0, 0, 0] += 1.0
    np.save(tmp_path / "run" / "latents.npy", latents)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_latent_cache(tmp_path / "run", split_rows(ids))


def test_bootstrap_summary_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    rows = [
        {"scheme": scheme, **{metric: float(rng.normal()) for metric in SUMMARY_METRICS}}
        for scheme in SCHEMES for _ in range(5)
    ]
    first, second = bootstrap_summary(rows, 1911), bootstrap_summary(rows, 1911)
    assert first == second
    assert len(first) == len(SCHEMES) * len(SUMMARY_METRICS)
    assert all(entry["ci95_low"] <= entry["mean"] <= entry["ci95_high"] for entry in first)
