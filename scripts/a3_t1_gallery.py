#!/usr/bin/env python3
"""A3.4: T1 magnitude / sorted-magnitude / norm gallery linkage on exact latents.

Threat model T1 (PROPOSAL §3): the attacker holds the exact ciphertext latent
``z_ano`` (no inversion error).  For each scheme we build a gallery of the
original inversion latents ``z_i`` (the frozen E2.3 cache) and, for every query
image, rank the gallery by a magnitude-based scorer, recording the rank of the
true image.  This is the upper-bound (error-free) version of the magnitude
attack; the inversion-error version is A4 (T2-WB).

Scorers (features invariant under the respective legacy transform):
  abs           elementwise |z|        — exact for S0 (z_ano = k⊙z, |z_ano|=|z|)
  sorted_abs    sort(|z|)              — exact for S1 (z_ano = P(k⊙z))
  norm          ||z||                  — exact for every orthogonal transform

Controls (reports/t2wb_protocol.md):
  P        identity key (z_ano = z): positive control, every scorer must hit Top-1
  shuffle  gallery labels permuted (seed 1911): negative control ≈ random
Metrics: Top-1, Top-5, mAP, full CMC, with bootstrap 95% CI over queries.

CPU only; reads the frozen cache, no GPU, no checkpoint.

  scripts/run_cfg_ddim.sh python -m scripts.a3_t1_gallery \\
      --latent-cache artifacts/runs/E2.3/<security_v1 cache run>
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
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "past/SourceCode"))

from scripts.e2_legacy_wrapper import build_legacy_components  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite

SCHEMES = ("P", "S0", "S1")
SCORERS = ("abs", "sorted_abs", "norm")
SEED = 1911
BOOTSTRAP = 10_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else "unavailable"


def load_cache(run_dir: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_id") != "E2.3":
        raise ValueError(f"{run_dir} is not an E2.3 cache run")
    latents = np.load(run_dir / "latents.npy").astype(np.float32).reshape(len(manifest["sample_ids"]), -1)
    with (run_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    for row in index:
        digest = hashlib.sha256(
            np.ascontiguousarray(latents[int(row["cache_index"])].reshape(1, 256, 256)).tobytes()
        ).hexdigest()
        if digest != row["latent_sha256"]:
            raise ValueError(f"latent hash mismatch for {row['sample_id']}")
    return latents, manifest


def encrypt_all(latents: np.ndarray, scheme: str) -> np.ndarray:
    """Exact T1 ciphertext latents z_ano for every gallery image (CPU, full mask)."""
    import torch

    from guided_diffusion.anonymization import AnonymizationMask, SignedPermutationKey

    anonymizer, key = build_legacy_components(ROOT / "past/SourceCode", scheme, torch.device("cpu"))
    mask = AnonymizationMask(shape=(1, 256, 256), mask_type=anonymizer.mask_type, margin=anonymizer.margin).mask
    tensor = torch.from_numpy(latents.reshape(-1, 1, 256, 256))
    expanded = mask.unsqueeze(0).expand_as(tensor)
    if isinstance(key, SignedPermutationKey):
        out = key.apply(tensor, expanded)
    else:
        k = key.key.unsqueeze(0).expand_as(tensor)
        out = expanded * (k * tensor) + (1 - expanded) * tensor
    return out.numpy().reshape(latents.shape[0], -1)


def feature(matrix: np.ndarray, scorer: str) -> np.ndarray:
    if scorer == "abs":
        return np.abs(matrix)
    if scorer == "sorted_abs":
        return np.sort(np.abs(matrix), axis=1)
    if scorer == "norm":
        return np.linalg.norm(matrix, axis=1, keepdims=True)
    raise ValueError(scorer)


def ranks_of_truth(query_feat: np.ndarray, gallery_feat: np.ndarray) -> np.ndarray:
    """Rank (1=best) of the diagonal true match under Euclidean distance.

    Ties are broken pessimistically for the attacker: the true image's rank is
    1 + (number of gallery items strictly closer) + (ties ahead of it), i.e. it
    is placed last among equal-distance items.
    """
    n = query_feat.shape[0]
    results = np.empty(n, dtype=np.int64)
    for i in range(n):
        distances = np.linalg.norm(gallery_feat - query_feat[i], axis=1)
        truth = distances[i]
        strictly_closer = int(np.sum(distances < truth))
        ties = int(np.sum(distances == truth)) - 1
        results[i] = strictly_closer + ties + 1
    return results


def summarize(ranks: np.ndarray, gallery_size: int, rng: np.random.Generator) -> dict:
    top1 = ranks <= 1
    top5 = ranks <= 5
    average_precision = 1.0 / ranks  # one true match per query
    draws = rng.integers(0, len(ranks), size=(BOOTSTRAP, len(ranks)))

    def ci(values: np.ndarray) -> list[float]:
        means = values[draws].mean(axis=1)
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    cmc = [float(np.mean(ranks <= k)) for k in range(1, gallery_size + 1)]
    return {
        "queries": int(len(ranks)),
        "gallery_size": gallery_size,
        "random_top1": 1.0 / gallery_size,
        "top1": float(top1.mean()), "top1_ci95": ci(top1.astype(float)),
        "top5": float(top5.mean()), "top5_ci95": ci(top5.astype(float)),
        "mAP": float(average_precision.mean()), "mAP_ci95": ci(average_precision),
        "median_rank": float(np.median(ranks)), "max_rank": int(ranks.max()),
        "cmc": cmc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, default=ROOT / "paper_assets/tables/table_T1_magnitude.csv")
    parser.add_argument("--cmc-output", type=Path, default=ROOT / "image/A3.4_T1_cmc.png")
    parser.add_argument("--json-output", type=Path, default=ROOT / "results/A3.4_T1_gallery.json")
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.table_output, args.cmc_output, args.json_output], args.force)

    latents, cache_manifest = load_cache(args.latent_cache)
    gallery_size = latents.shape[0]
    rng = np.random.default_rng(SEED)
    shuffle = rng.permutation(gallery_size)

    rows: list[dict] = []
    detail: dict = {}
    gallery_cache: dict[tuple[str, str], np.ndarray] = {}
    for scheme in SCHEMES:
        z_ano = encrypt_all(latents, scheme)
        for scorer in SCORERS:
            gallery_feat = feature(latents, scorer)
            query_feat = feature(z_ano, scorer)
            ranks = ranks_of_truth(query_feat, gallery_feat)
            positive = summarize(ranks, gallery_size, np.random.default_rng(SEED))
            # Negative control: same queries, gallery labels shuffled.
            shuffled_ranks = ranks_of_truth(query_feat, gallery_feat[shuffle])
            negative = summarize(shuffled_ranks, gallery_size, np.random.default_rng(SEED))
            detail[f"{scheme}/{scorer}"] = {"attack": positive, "negative_control": negative}
            rows.append({
                "scheme": scheme, "scorer": scorer,
                "queries": positive["queries"], "gallery_size": gallery_size,
                "random_top1": positive["random_top1"],
                "top1": positive["top1"], "top1_ci95_low": positive["top1_ci95"][0], "top1_ci95_high": positive["top1_ci95"][1],
                "top5": positive["top5"], "mAP": positive["mAP"],
                "median_rank": positive["median_rank"], "max_rank": positive["max_rank"],
                "negctrl_top1": negative["top1"], "negctrl_top1_ci95_high": negative["top1_ci95"][1],
            })
            gallery_cache[(scheme, scorer)] = np.array(positive["cmc"])
            print(f"{scheme}/{scorer}: top1={positive['top1']:.3f} top5={positive['top5']:.3f} "
                  f"mAP={positive['mAP']:.3f} negctrl_top1={negative['top1']:.3f}")

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with args.table_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": 1, "task_id": "A3.4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "script_sha256": sha256_file(Path(__file__)),
        "git_commit": git_commit(),
        "threat_model": "T1 (exact ciphertext latent, no inversion error)",
        "seed": SEED, "bootstrap_resamples": BOOTSTRAP,
        "numpy": np.__version__,
        "latent_cache_run_id": cache_manifest["run_id"],
        "split_sha256": cache_manifest["dataset_split_hash"],
        "gallery_size": gallery_size,
        "tie_break": "pessimistic for attacker (true match placed last among equal-distance items)",
        "results": detail,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_cmc(gallery_cache, args.cmc_output)
    print(f"wrote {args.table_output}, {args.json_output}, {args.cmc_output}")


def plot_cmc(cmc: dict[tuple[str, str], np.ndarray], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SCORERS), figsize=(12, 3.6), sharey=True)
    for ax, scorer in zip(axes, SCORERS):
        for scheme in SCHEMES:
            curve = cmc[(scheme, scorer)]
            ax.plot(range(1, len(curve) + 1), curve, label=scheme, linewidth=1.5)
        ax.set_title(f"scorer: {scorer}")
        ax.set_xlabel("rank k")
        ax.set_xlim(1, 20)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CMC: P(true match within top k)")
    axes[0].legend(title="scheme")
    fig.suptitle("A3.4 T1 magnitude gallery linkage (exact latent, security_v1)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
