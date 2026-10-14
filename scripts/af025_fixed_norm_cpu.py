#!/usr/bin/env python3
"""AF-025 formal CPU evidence for fixed-norm S2 v2 (N=200)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.d7_cpu_security import FIXTURE_KEY, ROUNDS, USE_PERMUTATION, load_cache, nonce_for
from scripts.d7_metrics import feature, rank_metrics, ranks_l2
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import sha256_file
from src.crypto.s2_fixed_norm import FIXED_NORM, FIXED_NORM_FLOAT64_BITS, normalize_to_fixed_norm
from src.crypto.sot import transform


SEED = 1911


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def protect_fixed(latents: np.ndarray, sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    protected, source_norms = [], []
    for latent, sample_id in zip(latents, sample_ids):
        normalized, original_norm = normalize_to_fixed_norm(torch.from_numpy(latent))
        encrypted = transform(
            normalized, FIXTURE_KEY, nonce_for(f"formal/{sample_id}"),
            rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        )
        protected.append(encrypted.numpy()); source_norms.append(original_norm)
    return np.stack(protected).astype(np.float32, copy=False), np.asarray(source_norms)


def norm_row(latents: np.ndarray, protected: np.ndarray) -> dict[str, Any]:
    truth = np.arange(len(latents))
    query = feature(protected, "norm")
    gallery = feature(latents, "norm")
    attack = rank_metrics(ranks_l2(query, gallery, truth), len(latents))
    negative = rank_metrics(
        ranks_l2(query, gallery, np.random.default_rng(SEED).permutation(len(latents))), len(latents),
    )
    protected_norm = np.linalg.norm(protected.astype(np.float64).reshape(len(protected), -1), axis=1)
    return {
        "task": "AF-025/D7.3", "threat_model": "T1", "method": "S2-fixed-norm/v2",
        "storage": "exact_latent", "scorer": "norm", "n": len(latents),
        "gallery_size": len(latents), "top1": attack["top1"], "top5": attack["top5"],
        "mAP": attack["mAP"], "median_rank": attack["median_rank"],
        "top1_ci_low": attack["top1_ci95_wilson"][0],
        "top1_ci_high": attack["top1_ci95_wilson"][1],
        "negative_top1": negative["top1"], "random_top1": 1 / len(latents),
        "protected_norm_mean": float(protected_norm.mean()),
        "protected_norm_max_abs_from_C": float(np.max(np.abs(protected_norm - float(FIXED_NORM)))),
        "fixed_norm": float(FIXED_NORM), "tie_break": "pessimistic",
    }


def probe_rows() -> list[dict[str, Any]]:
    dimension = 4096
    probes = {
        "basis_0": torch.nn.functional.one_hot(torch.tensor(0), dimension).double(),
        "basis_123": torch.nn.functional.one_hot(torch.tensor(123), dimension).double(),
        "constant": torch.ones(dimension, dtype=torch.float64),
        "sparse_16": torch.cat((torch.ones(16, dtype=torch.float64), torch.zeros(dimension - 16, dtype=torch.float64))),
    }
    rows = []
    for name, probe in probes.items():
        normalized, original_norm = normalize_to_fixed_norm(probe)
        protected = transform(
            normalized, FIXTURE_KEY, nonce_for(f"fixed-probe/{name}"),
            rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        )
        rows.append({
            "probe": name, "dimension": dimension, "original_norm": original_norm,
            "normalized_norm": float(torch.linalg.vector_norm(normalized)),
            "protected_norm": float(torch.linalg.vector_norm(protected)),
            "protected_norm_minus_C": float(torch.linalg.vector_norm(protected) - float(FIXED_NORM)),
            "output_mean": float(protected.mean()),
            "output_variance": float(protected.var(unbiased=False)),
            "output_max_abs": float(protected.abs().max()),
            "interpretation": "fixed norm removes norm distinction; structural diagnostics are not a security proof",
        })
    return rows


def repeat_rows(latent: np.ndarray) -> list[dict[str, Any]]:
    normalized, _ = normalize_to_fixed_norm(torch.from_numpy(latent))
    outputs, rows = [], []
    for index in range(100):
        nonce = nonce_for(f"fixed-repeat/{index}")
        protected = transform(
            normalized, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        ).numpy()
        outputs.append(protected.reshape(-1))
        rows.append({
            "repeat": index,
            "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
            "payload_sha256": hashlib.sha256(np.ascontiguousarray(protected).tobytes()).hexdigest(),
            "protected_norm": float(np.linalg.norm(protected.astype(np.float64))),
        })
    matrix = np.stack(outputs).astype(np.float64)
    normalized_outputs = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine = normalized_outputs @ normalized_outputs.T
    offdiag = cosine[~np.eye(len(cosine), dtype=bool)]
    unique_nonces = len({row["nonce_sha256"] for row in rows})
    unique_payloads = len({row["payload_sha256"] for row in rows})
    for row in rows:
        row.update({
            "unique_nonces_total": unique_nonces, "unique_payloads_total": unique_payloads,
            "offdiag_cosine_mean": float(offdiag.mean()),
            "offdiag_cosine_max": float(offdiag.max()),
            "all_unique": unique_nonces == unique_payloads == 100,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-meta", type=Path, default=Path("results/E2.3_DONE.json"))
    parser.add_argument("--norm-table", type=Path, default=Path("paper_assets/tables/table_AF025_fixed_norm_T1.csv"))
    parser.add_argument("--probe-table", type=Path, default=Path("paper_assets/tables/table_AF025_fixed_norm_probe.csv"))
    parser.add_argument("--repeat-table", type=Path, default=Path("paper_assets/tables/table_AF025_fixed_norm_repeat.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/AF025_fixed_norm_cpu.json"))
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.norm_table, args.probe_table, args.repeat_table, args.json_output], args.force)
    latents, sample_ids, cache_meta = load_cache(args.cache_meta)
    protected, source_norms = protect_fixed(latents, sample_ids)
    norm = [norm_row(latents, protected)]
    probes, repeats = probe_rows(), repeat_rows(latents[0])
    write_csv(args.norm_table, norm); write_csv(args.probe_table, probes); write_csv(args.repeat_table, repeats)
    payload = {
        "schema_version": 1, "task_id": "AF025.FIXED_NORM.CPU",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)), "cache_run_id": cache_meta["run_id"],
        "cache_sha256": cache_meta["cache_file_sha256"], "split_sha256": cache_meta["split_sha256"],
        "sample_count": len(latents), "fixed_norm": float(FIXED_NORM),
        "fixed_norm_float64_bits": FIXED_NORM_FLOAT64_BITS,
        "source_norm_min": float(source_norms.min()), "source_norm_max": float(source_norms.max()),
        "rounds": ROUNDS, "permutation": USE_PERMUTATION,
        "tables": {"D7.3_T1": str(args.norm_table), "D7.6": str(args.probe_table), "D7.7": str(args.repeat_table)},
        "disclaimer": "scientific outcomes are measurements, not engineering pass gates",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote AF-025 fixed-norm CPU evidence for {len(latents)} formal samples")


if __name__ == "__main__":
    main()
