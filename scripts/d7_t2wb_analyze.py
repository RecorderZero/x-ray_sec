#!/usr/bin/env python3
"""D7.2/D7.3 T1 and T2-WB linkage analysis for S2a/S2."""

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

from scripts.d7_cpu_security import (
    FIXTURE_KEY, ROUNDS, USE_PERMUTATION, linkage_rows, load_cache, nonce_for, protect_formal,
)
from scripts.d7_metrics import SCORERS, feature, rank_metrics, ranks_cosine, ranks_l2
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import sha256_file
from src.crypto.sot import inverse_transform


STORAGES = ("legacy_png", "range_preserving_png")
SEED = 1911


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_run(path: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
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
    return queries, index, manifest, rows


def calibration_bias(dev_rows: list[dict[str, str]]) -> dict[str, float]:
    output = {}
    for storage in STORAGES:
        values = [float(row["attacker_latent_norm_ratio"]) for row in dev_rows if row["storage"] == storage]
        if not values:
            raise RuntimeError(f"dev run missing storage: {storage}")
        output[storage] = float(np.median(values))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-meta", type=Path, default=Path("results/E2.3_DONE.json"))
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--dev-run", type=Path, required=True)
    parser.add_argument("--attack-table", type=Path, default=Path("paper_assets/tables/table_attack_original.csv"))
    parser.add_argument("--norm-table", type=Path, default=Path("paper_assets/tables/table_norm_leakage.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/D7_T1_T2WB_linkage.json"))
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.attack_table, args.norm_table, args.json_output], args.force)
    latents, sample_ids, cache_meta = load_cache(args.cache_meta)
    protected = protect_formal(latents, sample_ids)
    t1 = linkage_rows(latents, protected)
    formal_q, formal_index, formal_manifest, _ = load_run(args.formal_run)
    _, _, dev_manifest, dev_rows = load_run(args.dev_run)
    if (formal_manifest.get("formal") is not True
            or formal_manifest.get("evidence_role") != "formal"
            or dev_manifest.get("formal") is not False
            or dev_manifest.get("evidence_role") != "dev_calibration"):
        raise RuntimeError("formal/dev run roles are not explicit")
    if formal_manifest["dataset_split_hash"] != cache_meta["split_sha256"]:
        raise RuntimeError("formal split differs from latent cache")
    bias = calibration_bias(dev_rows)
    image_pos = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    truth_all = np.arange(len(latents))
    shuffled = np.random.default_rng(SEED).permutation(len(latents))
    attack_rows = [row for row in t1 if row["task"] == "D7.2"]
    norm_rows = [row for row in t1 if row["task"] == "D7.3"]
    detail: dict[str, Any] = {}
    for storage in STORAGES:
        entries = [entry for entry in formal_index if entry["storage"] == storage]
        entries.sort(key=lambda entry: image_pos[entry["image_id"]])
        positions = np.asarray([entry["query_index"] for entry in entries])
        truth = np.asarray([image_pos[entry["image_id"]] for entry in entries])
        if not np.array_equal(truth, truth_all):
            raise RuntimeError(f"formal query order/coverage mismatch: {storage}")
        query = formal_q[positions]
        decrypted = []
        for entry, value in zip(entries, query):
            restored = inverse_transform(
                torch.from_numpy(value), FIXTURE_KEY, nonce_for(f"formal/{entry['image_id']}"),
                rounds=ROUNDS, use_permutation=USE_PERMUTATION,
            )
            decrypted.append(restored.numpy())
        positive = rank_metrics(ranks_cosine(np.stack(decrypted), latents, truth), len(latents))
        storage_detail: dict[str, Any] = {"positive_control": positive}
        for scorer in SCORERS:
            gallery_feature = feature(latents, scorer)
            query_feature = feature(query, scorer)
            attack = rank_metrics(ranks_l2(query_feature, gallery_feature, truth), len(latents))
            negative = rank_metrics(ranks_l2(query_feature, gallery_feature, shuffled), len(latents))
            variants = [("raw", attack)]
            if scorer == "norm":
                calibrated = rank_metrics(
                    ranks_l2(query_feature / bias[storage], gallery_feature, truth), len(latents)
                )
                variants.append(("dev_median", calibrated))
            storage_detail[scorer] = {"raw": attack, "negative": negative}
            if scorer == "norm":
                storage_detail[scorer]["dev_median"] = variants[-1][1]
            for calibration, measured in variants:
                row = {
                    "task": "D7.3" if scorer == "norm" else "D7.2", "threat_model": "T2-WB",
                    "method": "S2a/S2", "storage": storage, "scorer": scorer,
                    "calibration": calibration, "dev_norm_bias": bias[storage] if scorer == "norm" else "",
                    "n": len(query), "gallery_size": len(latents), "top1": measured["top1"],
                    "top1_ci_low": measured["top1_ci95_wilson"][0],
                    "top1_ci_high": measured["top1_ci95_wilson"][1], "top5": measured["top5"],
                    "mAP": measured["mAP"], "median_rank": measured["median_rank"],
                    "negative_top1": negative["top1"], "positive_top1": positive["top1"],
                    "random_top1": 1 / len(latents), "tie_break": "pessimistic",
                    "rounds": ROUNDS, "permutation": USE_PERMUTATION,
                }
                (norm_rows if scorer == "norm" else attack_rows).append(row)
        detail[storage] = storage_detail
    write_csv(args.attack_table, attack_rows); write_csv(args.norm_table, norm_rows)
    payload = {
        "schema_version": 1, "task_id": "D7.2-D7.3", "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)), "formal_run_id": formal_manifest["run_id"],
        "dev_run_id": dev_manifest["run_id"], "cache_run_id": cache_meta["run_id"],
        "sample_count": len(latents), "dev_norm_bias": bias, "cells": detail,
        "positive_control_rule": "correct-key inverse then cosine gallery",
        "negative_control_rule": "seed-1911 shuffled truth labels",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote D7 T1/T2-WB linkage tables for n={len(latents)}")


if __name__ == "__main__":
    main()
