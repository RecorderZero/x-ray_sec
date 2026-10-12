#!/usr/bin/env python3
"""A4.2: T2-WB magnitude / sorted-magnitude / norm gallery linkage.

The attacker has re-inverted anonymous-image latents ``zhat_ano`` (A4.1), one per
(image, scheme, storage).  The magnitude invariants survive inversion with error,
so the scorers need no key: ``|zhat_ano|`` for S0, ``sort(|zhat_ano|)`` for S1,
``||zhat_ano||`` for norm.  Each (scheme, storage) cell is ranked against the
gallery of original security_v1 latents (E2.3 cache) and compared with:

  positive control  decrypt ``zhat_ano`` with the correct key -> ``zhat``; link
                    by latent cosine to the gallery (scheme-specific, AUD-20261010-02)
  negative control  gallery labels shuffled (seed 1911)

The norm attacker norm is biased by inversion; AF-025 requires a dev-only
multiplicative calibration (``--dev-calibration-run``) and both raw and
calibrated norm-only Top-k.  The public range-preserving ``lo/hi`` would be a
side channel only with a separate labelled enrollment set; with one record per
patient it is marked ``not_evaluable`` (reports/t2wb_protocol.md §3.1).

  scripts/run_cfg_ddim.sh python -m scripts.a4_t2wb_gallery \
    --gallery-cache artifacts/runs/E2.3/<security run> \
    --query-run artifacts/runs/A4.1/<formal run> \
    --dev-calibration-run artifacts/runs/A4.1_DEV_CAL/<dev run>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "past/SourceCode"))

from scripts.e2_legacy_wrapper import apply_legacy_key  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

SCHEMES = ("S0", "S1")
STORAGES = ("legacy_png", "range_preserving_png")
SEED = 1911
BOOTSTRAP = 10_000
LOWFREQ_POOL = 8
# scorer -> the scheme whose invariant it is the designated attack for
DESIGNATED = {"S0": "abs", "S1": "sorted_abs"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else "unavailable"


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def feature(matrix: np.ndarray, scorer: str) -> np.ndarray:
    """Magnitude features in float64 (AF-028)."""
    m64 = matrix.astype(np.float64).reshape(matrix.shape[0], -1)
    if scorer == "abs":
        return np.abs(m64)
    if scorer == "sorted_abs":
        return np.sort(np.abs(m64), axis=1)
    if scorer == "norm":
        return np.linalg.norm(m64, axis=1, keepdims=True)
    if scorer == "lowfreq_abs":
        side = int(round(m64.shape[1] ** 0.5))
        images = np.abs(m64).reshape(-1, side, side)
        pooled = images.reshape(-1, side // LOWFREQ_POOL, LOWFREQ_POOL,
                                side // LOWFREQ_POOL, LOWFREQ_POOL).mean(axis=(2, 4))
        return pooled.reshape(m64.shape[0], -1)
    raise ValueError(scorer)


def ranks_of_truth(query_feat: np.ndarray, gallery_feat: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pessimistic rank (1=best) of each query's true gallery image under L2 distance."""
    ranks = np.empty(len(query_feat), dtype=np.int64)
    for i in range(len(query_feat)):
        distances = np.linalg.norm(gallery_feat - query_feat[i], axis=1)
        d_true = distances[truth[i]]
        ranks[i] = int(np.sum(distances < d_true)) + int(np.sum(distances == d_true))
    return ranks


def cosine_ranks(query: np.ndarray, gallery: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rank of the true gallery image by latent cosine (for the positive control)."""
    q = query.astype(np.float64).reshape(len(query), -1)
    g = gallery.astype(np.float64).reshape(len(gallery), -1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    ranks = np.empty(len(q), dtype=np.int64)
    for i in range(len(q)):
        sims = g @ q[i]
        s_true = sims[truth[i]]
        ranks[i] = int(np.sum(sims > s_true)) + int(np.sum(sims == s_true))
    return ranks


def metrics(ranks: np.ndarray, gallery_size: int, rng: np.random.Generator) -> dict[str, Any]:
    n = len(ranks)
    top1 = int(np.sum(ranks <= 1))
    top5 = int(np.sum(ranks <= 5))
    ap = 1.0 / ranks
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    ap_means = ap[draws].mean(axis=1)
    return {
        "queries": n, "gallery_size": gallery_size, "random_top1": 1.0 / gallery_size,
        "top1": top1 / n, "top1_ci95_wilson": wilson_ci(top1, n),
        "top5": top5 / n, "top5_ci95_wilson": wilson_ci(top5, n),
        "mAP": float(ap.mean()),
        "mAP_ci95_bootstrap": [float(np.percentile(ap_means, 2.5)), float(np.percentile(ap_means, 97.5))],
        "median_rank": float(np.median(ranks)), "max_rank": int(ranks.max()),
        "cmc": [float(np.mean(ranks <= k)) for k in range(1, min(gallery_size, 20) + 1)],
    }


def load_query_run(run_dir: Path) -> tuple[np.ndarray, list[dict[str, str]], dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((run_dir / "query_index.json").read_text(encoding="utf-8"))
    latents = np.load(run_dir / "query_latents.npy")
    for entry in index:
        digest = hashlib.sha256(np.ascontiguousarray(latents[entry["query_index"]]).tobytes()).hexdigest()
        if digest != entry["query_sha256"]:
            raise ValueError(f"query latent hash mismatch: {entry['sample_id']}")
    return latents.astype(np.float32), index, manifest


def load_gallery(cache_dir: Path, split_rows: list[dict[str, str]]) -> np.ndarray:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_id") != "E2.3":
        raise ValueError(f"{cache_dir} is not an E2.3 cache")
    with (cache_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        order = [row["sample_id"] for row in csv.DictReader(handle)]
    if order != [row["sample_id"] for row in split_rows]:
        raise ValueError("gallery cache order differs from split")
    return np.load(cache_dir / "latents.npy").astype(np.float32)


def dev_norm_bias(dev_run: Path) -> dict[tuple[str, str], float]:
    """median(||zhat_ano|| / ||z_ano||) per (scheme, storage) from the dev run (AF-025)."""
    with (dev_run / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    bias: dict[tuple[str, str], float] = {}
    for scheme in SCHEMES:
        for storage in STORAGES:
            ratios = [float(r["attacker_latent_norm_ratio"]) for r in rows
                      if r["scheme"] == scheme and r["storage"] == storage]
            if not ratios:
                raise ValueError(f"dev calibration missing {scheme}/{storage}")
            bias[(scheme, storage)] = float(np.median(ratios))
    return bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--gallery-cache", type=Path, required=True)
    parser.add_argument("--query-run", type=Path, required=True)
    parser.add_argument("--dev-calibration-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--table-output", type=Path,
                        default=Path("paper_assets/tables/table_T2_WB_magnitude.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/A4.2_T2WB_gallery.json"))
    parser.add_argument("--cmc-output", type=Path, default=Path("image/A4.2_T2WB_cmc.png"))
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.table_output, args.json_output, args.cmc_output], args.force)

    from scripts.e1_ddim_runner import load_split
    import torch

    split_rows = load_split(args.split)
    gallery = load_gallery(args.gallery_cache, split_rows)  # (200,1,256,256)
    gallery_size = gallery.shape[0]
    queries, query_index, query_manifest = load_query_run(args.query_run)
    bias = dev_norm_bias(args.dev_calibration_run)
    rng_shuffle = np.random.default_rng(SEED)
    shuffle = rng_shuffle.permutation(gallery_size)

    # map (image_id, scheme, storage) -> query row
    by_cell: dict[tuple[str, str], list[int]] = {}
    image_order = [row["sample_id"] for row in split_rows]
    image_pos = {name: i for i, name in enumerate(image_order)}
    truth_by_cell: dict[tuple[str, str], np.ndarray] = {}
    for entry in query_index:
        image_id, scheme, storage = entry["sample_id"].split("|")
        by_cell.setdefault((scheme, storage), []).append(entry["query_index"])
    for cell, positions in by_cell.items():
        # queries were written in image order within each cell by A4.1
        truth_by_cell[cell] = np.array([image_pos[query_index[p]["sample_id"].split("|")[0]]
                                        for p in positions])

    gallery_feat = {s: feature(gallery, s) for s in ("abs", "sorted_abs", "norm", "lowfreq_abs")}
    cells: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    cmc_curves: dict[str, list[float]] = {}

    for scheme in SCHEMES:
        for storage in STORAGES:
            positions = np.array(by_cell[(scheme, storage)])
            truth = truth_by_cell[(scheme, storage)]
            q = queries[positions]  # (200,1,256,256) re-inverted ciphertext latents
            # Negative control: each query is assigned a random gallery target
            # (seed 1911), so Top-1 ~ 1/N regardless of the scorer.
            neg_truth = shuffle

            cell_result: dict[str, Any] = {}
            scorers = ["abs", "sorted_abs", "norm", "lowfreq_abs"]
            for scorer in scorers:
                qf = feature(q, scorer)
                gf = gallery_feat[scorer]
                attack = metrics(ranks_of_truth(qf, gf, truth), gallery_size, np.random.default_rng(SEED))
                negative = metrics(ranks_of_truth(qf, gf, neg_truth), gallery_size, np.random.default_rng(SEED))
                entry = {"attack": attack, "negative_control": negative,
                         "designated_for": [s for s, sc in DESIGNATED.items() if sc == scorer]}
                if scorer == "norm":
                    # AF-025: dev-calibrated norm-only
                    qf_cal = qf / bias[(scheme, storage)]
                    cal = metrics(ranks_of_truth(qf_cal, gf, truth), gallery_size, np.random.default_rng(SEED))
                    entry["calibrated_attack"] = cal
                    entry["dev_norm_bias"] = bias[(scheme, storage)]
                cell_result[scorer] = entry
                cmc_curves[f"{scheme}/{storage}/{scorer}"] = attack["cmc"]

            # scheme-specific positive control: decrypt zhat_ano with correct key
            z_hat = apply_legacy_key(torch.from_numpy(q), args.source_root, scheme, "deanonymize").numpy()
            pos = metrics(cosine_ranks(z_hat, gallery, truth), gallery_size, np.random.default_rng(SEED))
            cell_result["positive_control_decrypted_cosine"] = pos

            # Inconclusive判定 per scorer (AUD-20261010-02 §4)
            neg_top1_hi = cell_result["abs"]["negative_control"]["top1_ci95_wilson"][1]  # same gallery size
            pos_lo = pos["top1_ci95_wilson"][0]
            pos_vs_neg_separable = pos_lo > neg_top1_hi
            def classify(attack_m: dict, neg_m: dict) -> str:
                if attack_m["top1_ci95_wilson"][0] > neg_m["top1_ci95_wilson"][1]:
                    return "attack_success"
                if not pos_vs_neg_separable:
                    return "Inconclusive"  # positive control too weak to interpret a miss
                return "linkage_not_significant_positive_control_passed"

            neg = {sc: cell_result[sc]["negative_control"] for sc in scorers}
            for scorer in scorers:
                cell_result[scorer]["verdict"] = classify(cell_result[scorer]["attack"], neg[scorer])
            # AF-025: the norm-only threat is judged on the dev-calibrated attack.
            cell_result["norm"]["calibrated_verdict"] = classify(
                cell_result["norm"]["calibrated_attack"], neg["norm"])

            # lo/hi side channel: not evaluable with one record per patient (AF-025)
            cell_result["lohi_sidechannel"] = {
                "status": "not_evaluable",
                "reason": ("one anonymous record per patient; lo/hi linkage needs a separate labelled "
                           "enrollment set from different acquisitions (reports/t2wb_protocol.md §3.1)"),
                "metadata_constant": storage == "legacy_png",
            }

            cells[f"{scheme}/{storage}"] = cell_result
            for scorer in scorers:
                e = cell_result[scorer]
                row = {
                    "scheme": scheme, "storage": storage, "scorer": scorer,
                    "designated": scorer == DESIGNATED[scheme],
                    "queries": e["attack"]["queries"], "gallery_size": gallery_size,
                    "attack_top1": e["attack"]["top1"],
                    "attack_top1_ci_low": e["attack"]["top1_ci95_wilson"][0],
                    "attack_top1_ci_high": e["attack"]["top1_ci95_wilson"][1],
                    "attack_top5": e["attack"]["top5"], "attack_mAP": e["attack"]["mAP"],
                    "negctrl_top1": e["negative_control"]["top1"],
                    "negctrl_top1_ci_high": e["negative_control"]["top1_ci95_wilson"][1],
                    "calibrated_top1": e.get("calibrated_attack", {}).get("top1", ""),
                    "calibrated_top1_ci_low": e.get("calibrated_attack", {}).get("top1_ci95_wilson", ["", ""])[0],
                    "calibrated_verdict": e.get("calibrated_verdict", ""),
                    "poscontrol_top1": pos["top1"], "poscontrol_top1_ci_low": pos["top1_ci95_wilson"][0],
                    "verdict": e["verdict"],
                }
                table_rows.append(row)

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with args.table_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)

    payload = {
        "schema_version": 1, "task_id": "A4.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "script_sha256": sha256_file(Path(__file__)),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "threat_model": "T2-WB (re-inverted anonymous-image latents, with inversion error)",
        "seed": SEED, "bootstrap_resamples": BOOTSTRAP, "numpy": np.__version__,
        "gallery_cache_run_id": json.loads((args.gallery_cache / "manifest.json").read_text())["run_id"],
        "gallery_cache_sha256": sha256_file(args.gallery_cache / "latents.npy"),
        "query_run_id": query_manifest["run_id"],
        "query_cache_sha256": sha256_file(args.query_run / "query_latents.npy"),
        "dev_calibration_run_id": json.loads((args.dev_calibration_run / "manifest.json").read_text())["run_id"],
        "dev_norm_bias": {f"{s}/{st}": bias[(s, st)] for s in SCHEMES for st in STORAGES},
        "gallery_size": gallery_size,
        "tie_break": "pessimistic for attacker",
        "inconclusive_rule": "AUD-20261010-02 §4; scheme-specific decrypted-cosine positive control",
        "cells": cells,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_cmc(cmc_curves, args.cmc_output)
    print(f"wrote {args.table_output}, {args.json_output}, {args.cmc_output}")
    for row in table_rows:
        if row["designated"]:
            print(f"{row['scheme']}/{row['storage']} {row['scorer']}: top1={row['attack_top1']:.3f} "
                  f"neg={row['negctrl_top1']:.3f} verdict={row['verdict']}")


def plot_cmc(curves: dict[str, list[float]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(STORAGES), figsize=(11, 4), sharey=True)
    colour = {"S0": "#2a78d6", "S1": "#eb6834"}
    style = {"abs": "-", "sorted_abs": "--", "norm": ":", "lowfreq_abs": "-."}
    for ax, storage in zip(axes, STORAGES):
        for scheme in SCHEMES:
            for scorer in ("abs", "sorted_abs", "norm"):
                curve = curves[f"{scheme}/{storage}/{scorer}"]
                ax.plot(range(1, len(curve) + 1), curve, colour[scheme], linestyle=style[scorer],
                        linewidth=1.3, label=f"{scheme} {scorer}")
        ax.set_title(storage, fontsize=10)
        ax.set_xlabel("rank k"); ax.set_xlim(1, 20); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CMC: P(true match within top k)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("A4.2 T2-WB magnitude gallery linkage (security_v1, N=200)", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
