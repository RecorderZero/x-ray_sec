#!/usr/bin/env python3
"""P3.1: attack-overview figure for two fixed dev cases.

For each fixed case shows, in one row: the original image, the inversion latent
z (heatmap), the sorted-|z| profile overlaying z / S0 z_ano / S1 z_ano (the
magnitude invariants that A3.4 exploits), and the S0 and S1 anonymous images.
It makes the A3.4 numbers legible: S0 preserves |z| per coordinate, S1 preserves
only the sorted multiset, both leave the anonymous image visually scrambled.

Latents come from the frozen E2.3 dev cache; the two anonymous images per case
are generated on the GPU with the legacy pipeline (4 generations total).

  scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id P3.1 \\
      --validate-artifacts -- python -m scripts.p3_attack_overview \\
      --checkpoint <ckpt> --latent-cache artifacts/runs/E2.3/<dev cache run>
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import Tee, create_runtime, load_split, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.e2_legacy_wrapper import apply_legacy_key, legacy_generate
from scripts.run_artifacts import finalize_run, open_run, sha256_file

FIXED_CASES = ("dev_v1.1_000", "dev_v1.1_010")  # one healthy, one effusion


def _u8(array: np.ndarray) -> np.ndarray:
    lo, hi = float(array.min()), float(array.max())
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint((array - lo) / (hi - lo) * 255).astype(np.uint8)


def build_figure(panels: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(panels), 5, figsize=(16, 3.4 * len(panels)))
    if len(panels) == 1:
        axes = axes[None, :]
    titles = ("original x0", "latent z", "sorted |·| profile", "S0 anonymous", "S1 anonymous")
    for row, panel in enumerate(panels):
        axes[row][0].imshow(_u8(panel["x0"]), cmap="gray")
        axes[row][1].imshow(panel["z"], cmap="gray")
        ax = axes[row][2]
        ax.plot(panel["sorted_z"], label="z", linewidth=1.0)
        ax.plot(panel["sorted_s0"], "--", label="S0 z_ano", linewidth=1.0)
        ax.plot(panel["sorted_s1"], ":", label="S1 z_ano", linewidth=1.2)
        ax.set_xlabel("coordinate rank"); ax.set_ylabel("|value|")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        axes[row][3].imshow(_u8(panel["s0_img"]), cmap="gray")
        axes[row][4].imshow(_u8(panel["s1_img"]), cmap="gray")
        for column in (0, 1, 3, 4):
            axes[row][column].set_xticks([]); axes[row][column].set_yticks([])
        axes[row][0].set_ylabel(panel["sample_id"], fontsize=10)
        if row == 0:
            for column, title in enumerate(titles):
                axes[row][column].set_title(title, fontsize=11)
    fig.suptitle("P3.1 attack overview: magnitude invariants behind A3.4", fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--figure-output", type=Path, default=Path("image/P3.1_attack_overview.png"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="P3.1")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = {row["sample_id"]: row for row in load_split(args.split)}
        latents = np.load(args.latent_cache / "latents.npy").astype(np.float32)
        order = [row["sample_id"] for row in load_split(args.split)]
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        panels, records = [], []
        for sample_id in FIXED_CASES:
            row = rows[sample_id]
            x0, x0_array = preprocess(Path(row["local_path"]))
            z = torch.from_numpy(latents[order.index(sample_id)]).unsqueeze(0).to(device)
            z_s0 = apply_legacy_key(z, args.source_root, "S0", "anonymize")
            z_s1 = apply_legacy_key(z, args.source_root, "S1", "anonymize")
            s0_img = legacy_generate(diffusion, model_fn, z_s0, args.noise_level)
            s1_img = legacy_generate(diffusion, model_fn, z_s1, args.noise_level)
            z_np, s0_np, s1_np = (t.detach().float().cpu().numpy().ravel() for t in (z, z_s0, z_s1))
            panels.append({
                "sample_id": sample_id, "x0": x0_array,
                "z": z.detach().float().cpu().numpy().reshape(256, 256),
                "sorted_z": np.sort(np.abs(z_np)), "sorted_s0": np.sort(np.abs(s0_np)),
                "sorted_s1": np.sort(np.abs(s1_np)),
                "s0_img": s0_img[0, 0].detach().float().cpu().numpy(),
                "s1_img": s1_img[0, 0].detach().float().cpu().numpy(),
            })
            records.append({
                "schema_version": 1, "sample_id": sample_id, "label": row["label"],
                "abs_equal_s0": float(np.max(np.abs(np.abs(z_np) - np.abs(s0_np)))),
                "sorted_abs_equal_s1": float(np.max(np.abs(np.sort(np.abs(z_np)) - np.sort(np.abs(s1_np))))),
                **{f"s0_anon_vs_x0_{k.split('_', 2)[-1]}": v for k, v in
                   image_pair_metrics("s0_anon_vs_x0", x0.unsqueeze(0).to(device), s0_img).items()},
            })
            print(f"{sample_id}: abs_equal_s0={records[-1]['abs_equal_s0']:.2e} "
                  f"sorted_abs_equal_s1={records[-1]['sorted_abs_equal_s1']:.2e}")
        build_figure(panels, args.figure_output)
        numeric = [k for k, v in records[0].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "schema_version"]
        cache_manifest = __import__("json").loads((args.latent_cache / "manifest.json").read_text())
        manifest = {
            "dataset_split_hash": next(iter(rows.values()))["split_sha256"],
            "checkpoint_hash": sha256_file(args.checkpoint), "seed_list": [0] * len(records),
            "steps": args.noise_level, "scheme": "S0/S1 fixed cases", "rounds": 0,
            "nonce_mode": "none", "container_version": "none",
            "pipeline": "legacy_anonymization:ddim_sample_loop_anonymization",
            "latent_cache_run_id": cache_manifest["run_id"], "figure": str(args.figure_output),
        }
        finalize_run(context, Path(__file__), records, numeric, manifest,
                     stdout_capture.getvalue(), stderr_capture.getvalue())


if __name__ == "__main__":
    main()
