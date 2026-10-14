#!/usr/bin/env python3
"""Analyze AF-025 fixed-norm T1 and T2-WB norm linkage."""

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

from scripts.d7_cpu_security import load_cache
from scripts.d7_metrics import feature, rank_metrics, ranks_l2
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import sha256_file
from src.crypto.s2_fixed_norm import FIXED_NORM


SEED = 1911


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def load_run(path: Path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    queries = np.load(path / "query_latents.npy").astype(np.float32, copy=False)
    index = json.loads((path / "query_index.json").read_text(encoding="utf-8"))
    with (path / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(queries) != len(index) or len(rows) != len(index):
        raise RuntimeError("query run artifact lengths differ")
    for entry in index:
        digest = hashlib.sha256(np.ascontiguousarray(queries[entry["query_index"]]).tobytes()).hexdigest()
        if digest != entry["query_sha256"]:
            raise RuntimeError(f"query hash mismatch: {entry['sample_id']}")
    return queries, index, rows, manifest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows: fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-meta", type=Path, default=Path("results/E2.3_DONE.json"))
    parser.add_argument("--t1-table", type=Path, default=Path("paper_assets/tables/table_AF025_fixed_norm_T1.csv"))
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--dev-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper_assets/tables/table_AF025_fixed_norm_norm_leakage.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/AF025_fixed_norm_T1_T2WB.json"))
    add_force_flag(parser); args = parser.parse_args()
    refuse_overwrite([args.output, args.json_output], args.force)
    latents, sample_ids, cache_meta = load_cache(args.cache_meta)
    formal_q, formal_index, formal_rows, formal_manifest = load_run(args.formal_run)
    _, _, dev_rows, dev_manifest = load_run(args.dev_run)
    if (formal_manifest.get("formal") is not True or formal_manifest.get("evidence_role") != "formal"
            or dev_manifest.get("formal") is not False or dev_manifest.get("evidence_role") != "dev_calibration"):
        raise RuntimeError("formal/dev evidence roles are not explicit")
    if formal_manifest["dataset_split_hash"] != cache_meta["split_sha256"]:
        raise RuntimeError("formal split differs from frozen cache")
    image_pos = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    formal_index.sort(key=lambda entry: image_pos[entry["sample_id"]])
    positions = np.asarray([entry["query_index"] for entry in formal_index])
    truth = np.asarray([image_pos[entry["sample_id"]] for entry in formal_index])
    if not np.array_equal(truth, np.arange(len(latents))):
        raise RuntimeError("formal query order/coverage mismatch")
    query = formal_q[positions]
    gallery_feature = feature(latents, "norm"); query_feature = feature(query, "norm")
    dev_bias = float(np.median([
        float(row["attacker_latent_norm_ratio_to_C"]) for row in dev_rows
    ]))
    shuffled = np.random.default_rng(SEED).permutation(len(latents))
    negative = rank_metrics(ranks_l2(query_feature, gallery_feature, shuffled), len(latents))
    with args.t1_table.open(newline="", encoding="utf-8") as handle:
        output_rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    detail = {}
    for calibration, values in (
        ("raw", query_feature), ("dev_median_to_C", query_feature / dev_bias),
    ):
        measured = rank_metrics(ranks_l2(values, gallery_feature, truth), len(latents))
        detail[calibration] = measured
        output_rows.append({
            "task": "AF-025/D7.3", "threat_model": "T2-WB",
            "method": "S2-fixed-norm/v2", "storage": "private_range_png", "scorer": "norm",
            "calibration": calibration, "dev_norm_bias": dev_bias,
            "n": len(query), "gallery_size": len(latents), "top1": measured["top1"],
            "top5": measured["top5"], "mAP": measured["mAP"],
            "median_rank": measured["median_rank"],
            "top1_ci_low": measured["top1_ci95_wilson"][0],
            "top1_ci_high": measured["top1_ci95_wilson"][1],
            "negative_top1": negative["top1"], "random_top1": 1 / len(latents),
            "fixed_norm": float(FIXED_NORM), "tie_break": "pessimistic",
        })
    write_csv(args.output, output_rows)
    legitimate_cosines = np.asarray([float(row["legitimate_latent_cosine"]) for row in formal_rows])
    payload = {
        "schema_version": 1, "task_id": "AF025.FIXED_NORM.D7.3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)), "formal_run_id": formal_manifest["run_id"],
        "dev_run_id": dev_manifest["run_id"], "cache_run_id": cache_meta["run_id"],
        "sample_count": len(latents), "fixed_norm": float(FIXED_NORM), "dev_norm_bias": dev_bias,
        "cells": detail, "negative_control": negative,
        "legitimate_latent_cosine_mean": float(legitimate_cosines.mean()),
        "legitimate_latent_cosine_min": float(legitimate_cosines.min()),
        "calibration_rule": "dev median attacker norm divided by public fixed C",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote AF-025 fixed-norm linkage for n={len(latents)}")


if __name__ == "__main__":
    main()
