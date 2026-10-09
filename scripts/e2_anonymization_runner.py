#!/usr/bin/env python3
"""Runs on the predecessor's anonymization (encryption) pipeline for P/S0/S1.

AF-017: the unencrypted control P goes through the same legacy sampler as
S0/S1 (``ddim_sample_loop_anonymization``) with an identity Rademacher key, so
P, S0 and S1 differ only in the key.  The inference/lesion-localization flow
used by E1.4 (``ddim_sample_loop_known_progressive``) is not used here.

Modes:
  benchmark  per-step cost of the legacy inversion pass (null=True) and the
             generation pass at guidance -1 (and 0 for comparison) at batch
             1/4/8; basis of reports/compute_budget.md (AF-021).
  p-smoke    four dev images through P: anonymous output, deterministic
             repeat of the inversion, re-inversion of the anonymous image
             (the T2-WB positive-control path) and M1 end-to-end recovery,
             with float and legacy-PNG hand-off (AF-017, AF-019).

The anonymization pipeline has no t=0 q_sample noise, so re-inversion here is
deterministic; the shared/unknown-noise distinction of AF-019 does not apply.

Examples:
  scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id AF017_P_SMOKE \\
      --validate-artifacts -- python -m scripts.e2_anonymization_runner p-smoke \\
      --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from scripts.e1_ddim_runner import Tee, create_runtime, load_split, preprocess
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING,
    LEGACY_GUIDANCE_SCALE,
    LEGACY_PIPELINE,
    legacy_anonymize,
    legacy_deanonymize,
    legacy_invert,
    legacy_model_kwargs,
)
from scripts.run_artifacts import finalize_run, open_run, sha256_file


SMOKE_ROWS = (0, 1, 10, 11)  # same dev images as E1.4
CACHE_RECOMPUTE_TOLERANCE = 1e-5  # WORKFLOW E2.3: sampled recomputation MaxAbs
CACHE_SHAPE = (1, 256, 256)
LOWFREQ_POOL = 8  # 256x256 latent -> 32x32 block means (AUD-20261008-03 diagnostic)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latent_pair_metrics(prefix: str, reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """Latent agreement plus low/high-frequency and |z| diagnostics (AF-019)."""
    x = reference.detach().float().cpu()
    y = other.detach().float().cpu()
    flat_x, flat_y = x.numpy().ravel(), y.numpy().ravel()
    error = flat_y - flat_x
    mse = float(np.mean(error * error))

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    low_x = F.avg_pool2d(x.reshape(1, 1, 256, 256), LOWFREQ_POOL)
    low_y = F.avg_pool2d(y.reshape(1, 1, 256, 256), LOWFREQ_POOL)
    high_x = x.reshape(1, 1, 256, 256) - F.interpolate(low_x, scale_factor=LOWFREQ_POOL, mode="nearest")
    high_y = y.reshape(1, 1, 256, 256) - F.interpolate(low_y, scale_factor=LOWFREQ_POOL, mode="nearest")
    return {
        f"{prefix}_cosine": cosine(flat_x, flat_y),
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": math.sqrt(mse),
        f"{prefix}_max_abs": float(np.max(np.abs(error))),
        f"{prefix}_float32_bit_exact_rate": float(np.mean(flat_x.view(np.uint32) == flat_y.view(np.uint32))),
        f"{prefix}_norm_ratio": float(np.linalg.norm(flat_y) / np.linalg.norm(flat_x)),
        f"{prefix}_lowfreq32_cosine": cosine(low_x.numpy().ravel(), low_y.numpy().ravel()),
        f"{prefix}_highfreq_cosine": cosine(high_x.numpy().ravel(), high_y.numpy().ravel()),
        f"{prefix}_abs_pearson": float(np.corrcoef(np.abs(flat_x), np.abs(flat_y))[0, 1]),
    }


def image_pair_metrics(prefix: str, reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """Image-space agreement on the [0,1] scale (WORKFLOW 3.3)."""
    x = reference.detach().float().cpu().numpy().reshape(256, 256)
    y = other.detach().float().cpu().numpy().reshape(256, 256)
    error = (y - x).ravel()
    mse = float(np.mean(error * error))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    x8 = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
    y8 = np.rint(np.clip(y, 0, 1) * 255).astype(np.uint8)
    return {
        f"{prefix}_psnr_db": float(-10.0 * math.log10(max(mse, np.finfo(float).tiny))),
        f"{prefix}_ssim": float(structural_similarity(x, y, data_range=1.0)),
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": math.sqrt(mse),
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_linf": float(np.max(np.abs(error))),
        f"{prefix}_cosine": float(np.sum(x * y) / denom) if denom else 0.0,
        f"{prefix}_uint8_pixel_equality_rate": float(np.mean(x8 == y8)),
        f"{prefix}_float32_bit_exact_rate": float(np.mean(x.view(np.uint32) == y.view(np.uint32))),
    }


def legacy_png_handoff(image: torch.Tensor) -> torch.Tensor:
    """Emulate the predecessor's anonymous-image storage and reload.

    ``cfg_image_sample_anonymization.py`` saves ``(visualize(sample)*255).astype(uint8)``
    (per-image min-max, truncation) as PNG; the loader min-max normalizes on
    reload, which is the identity on an image already spanning 0..255.
    """
    low, high = image.amin(dim=(1, 2, 3), keepdim=True), image.amax(dim=(1, 2, 3), keepdim=True)
    stored = ((image - low) / (high - low) * 255).to(torch.uint8)
    reloaded = stored.float() / 255.0
    r_low, r_high = reloaded.amin(dim=(1, 2, 3), keepdim=True), reloaded.amax(dim=(1, 2, 3), keepdim=True)
    return (reloaded - r_low) / (r_high - r_low)


def _to_u8(array: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def save_p_contact_sheet(items, output: Path) -> None:
    columns = ("original", "P output", "M1 float", "M1 PNG", "|orig-M1 PNG| 0..0.1")
    width, height = 256 * len(columns), len(items) * 286
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (sample_id, panels) in enumerate(items):
        y0 = row_index * 286
        original, anonymous, rec_float, rec_png = panels
        diff = np.rint(np.clip(np.abs(rec_png - original) / 0.1, 0, 1) * 255).astype(np.uint8)
        for column, array in enumerate((_to_u8(original), _to_u8(anonymous), _to_u8(rec_float), _to_u8(rec_png), diff)):
            canvas.paste(Image.fromarray(array, mode="L").convert("RGB"), (column * 256, y0 + 30))
        draw.text((4, y0 + 5), f"{sample_id}: " + " | ".join(columns), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def common_manifest(args, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "dataset_split_hash": rows[0]["split_sha256"],
        "checkpoint_hash": sha256_file(args.checkpoint),
        "source_sampler_sha256": sha256_file(
            args.source_root / "guided_diffusion" / "gaussian_diffusion.py"
        ),
        "steps": args.noise_level,
        "rounds": 0,
        "nonce_mode": "none",
        "container_version": "none",
        "pipeline": LEGACY_PIPELINE,
        "guidance_scale": LEGACY_GUIDANCE_SCALE,
        "generation_conditioning": GENERATION_CONDITIONING,
        "inversion_conditioning": "null=True (unconditional)",
        "raw_key_material_recorded": False,
    }


@torch.inference_mode()
def run_p_smoke(args, rows, diffusion, model_fn, device, context) -> tuple[list[dict[str, Any]], list[str]]:
    output_rows: list[dict[str, Any]] = []
    contact_items = []
    for row_index in SMOKE_ROWS:
        row = rows[row_index]
        x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings: dict[str, float] = {}

        _sync(device); start = time.perf_counter()
        anonymous, latent, _ = legacy_anonymize(diffusion, model_fn, x0, args.source_root, "P", args.noise_level)
        _sync(device); timings["runtime_seconds_anonymize"] = time.perf_counter() - start

        start = time.perf_counter()
        latent_repeat, _ = diffusion.ddim_anonymization_forward(
            model_fn, x0, args.noise_level, clip_denoised=True,
            model_kwargs=legacy_model_kwargs(1, device),
        )
        _sync(device); timings["runtime_seconds_repeat_inversion"] = time.perf_counter() - start

        start = time.perf_counter()
        recovered_float, reinverted_float, _ = legacy_deanonymize(
            diffusion, model_fn, anonymous, args.source_root, "P", args.noise_level
        )
        _sync(device); timings["runtime_seconds_m1_float"] = time.perf_counter() - start

        anonymous_png = legacy_png_handoff(anonymous)
        start = time.perf_counter()
        recovered_png, reinverted_png, _ = legacy_deanonymize(
            diffusion, model_fn, anonymous_png, args.source_root, "P", args.noise_level
        )
        _sync(device); timings["runtime_seconds_m1_png"] = time.perf_counter() - start

        tensors = (anonymous, latent, latent_repeat, recovered_float, reinverted_float,
                   anonymous_png, recovered_png, reinverted_png)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"NaN/Inf detected for {row['sample_id']}")
        record: dict[str, Any] = {
            "schema_version": 1,
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "scheme": "P",
            "seed": args.seed,
            "noise_level": args.noise_level,
            **timings,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "finite": True,
        }
        # P output: x0 -> z -> identity key -> generation (reconstruction).
        record.update(image_pair_metrics("image_p_output_vs_original", x0, anonymous))
        # E2.3 resampled recomputation analogue: same input, same pipeline.
        record.update(latent_pair_metrics("latent_repeat", latent, latent_repeat))
        # Re-inversion of the P output (T2-WB positive-control path), float hand-off.
        record.update(latent_pair_metrics("latent_reinversion_float", latent, reinverted_float))
        record.update(image_pair_metrics("image_m1_float_vs_original", x0, recovered_float))
        # Same with the predecessor's per-image min-max uint8 PNG storage.
        record.update(image_pair_metrics("image_png_handoff_vs_p_output", anonymous, anonymous_png))
        record.update(latent_pair_metrics("latent_reinversion_png", latent, reinverted_png))
        record.update(image_pair_metrics("image_m1_png_vs_original", x0, recovered_png))
        output_rows.append(record)
        contact_items.append((row["sample_id"], [
            value[0, 0].detach().float().cpu().numpy()
            for value in (x0, anonymous, recovered_float, recovered_png)
        ]))
        print(
            f"completed {row['sample_id']}: p_output_psnr={record['image_p_output_vs_original_psnr_db']:.3f} "
            f"reinv_float_cos={record['latent_reinversion_float_cosine']:.4f} "
            f"m1_png_psnr={record['image_m1_png_vs_original_psnr_db']:.3f}"
        )
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    with args.smoke_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    save_p_contact_sheet(contact_items, args.contact_sheet)
    numeric = [
        key for key, value in output_rows[0].items()
        if key.startswith(("image_", "latent_", "runtime_seconds_")) or key == "peak_vram_bytes"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return output_rows, numeric


@torch.inference_mode()
def run_benchmark(args, rows, diffusion, model_fn, device, context) -> tuple[list[dict[str, Any]], list[str]]:
    images = [preprocess(Path(row["local_path"]))[0] for row in rows[:8]]
    timestep = args.benchmark_timestep
    results = []
    for batch_size in (1, 4, 8):
        batch = torch.stack(images[:batch_size]).to(device)
        kwargs = legacy_model_kwargs(batch_size, device)
        reverse_kwargs = dict(kwargs, null=True)
        t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

        def forward_step():
            diffusion.ddim_reverse_sample(model_fn, batch, t, clip_denoised=True, model_kwargs=reverse_kwargs)

        def generation_step(guidance):
            diffusion.ddim_sample(
                model_fn, batch, t, clip_denoised=True, model_kwargs=dict(kwargs), guidance_scale=guidance
            )

        def timed(step) -> float:
            step()  # warm-up
            _sync(device)
            samples = []
            for _ in range(args.benchmark_repeats):
                start = time.perf_counter()
                for _ in range(args.benchmark_steps):
                    step()
                _sync(device)
                samples.append((time.perf_counter() - start) / args.benchmark_steps / batch_size)
            return float(np.mean(samples))

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        forward = timed(forward_step)
        generation = timed(lambda: generation_step(LEGACY_GUIDANCE_SCALE))
        generation_g0 = timed(lambda: generation_step(0.0))
        steps = args.noise_level - 1
        results.append({
            "sample_id": f"batch_{batch_size}",
            "batch_size": batch_size,
            "forward_seconds_per_step_per_image": forward,
            "generation_seconds_per_step_per_image": generation,
            "generation_guidance0_seconds_per_step_per_image": generation_g0,
            "estimated_forward_half_cycle_seconds_per_image": forward * steps,
            "estimated_generation_half_cycle_seconds_per_image": generation * steps,
            "estimated_generation_guidance0_half_cycle_seconds_per_image": generation_g0 * steps,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        })
        print(f"batch {batch_size}: forward={forward:.5f}s/step/img generation(-1)={generation:.5f} generation(0)={generation_g0:.5f}")
    payload = {
        "schema_version": 1,
        "task_id": context.task_id,
        "run_id": context.run_id,
        "pipeline": LEGACY_PIPELINE,
        "benchmark_timestep": timestep,
        "benchmark_steps_per_repeat": args.benchmark_steps,
        "benchmark_repeats": args.benchmark_repeats,
        "steps_per_half_cycle": args.noise_level - 1,
        "results": results,
        "interpretation": "half-cycle estimate = per-step time x (noise_level-1); model loading, I/O and validation excluded",
    }
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    numeric = [key for key in results[0] if key not in {"sample_id"}]
    return results, numeric


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@torch.inference_mode()
def run_cache(args, rows, diffusion, model_fn, device, context) -> tuple[list[dict[str, Any]], list[str]]:
    """E2.3: legacy x_T for every split image plus a seeded batch-1 recomputation."""
    selected = rows[: args.limit] if args.limit else rows
    count = len(selected)
    images = [preprocess(Path(row["local_path"]))[0] for row in selected]
    latents = np.empty((count, *CACHE_SHAPE), dtype=np.float32)
    batch_runtime: dict[int, float] = {}
    for start in range(0, count, args.batch_size):
        batch = torch.stack(images[start:start + args.batch_size]).to(device)
        _sync(device)
        began = time.perf_counter()
        x_T = legacy_invert(diffusion, model_fn, batch, args.noise_level)
        _sync(device)
        batch_runtime[start] = (time.perf_counter() - began) / batch.shape[0]
        latents[start:start + batch.shape[0]] = x_T.detach().float().cpu().numpy()
        print(f"cached {start + batch.shape[0]}/{count}")

    recompute_ids = sorted(random.Random(args.seed).sample(range(count), min(args.recompute_count, count)))
    recompute: dict[int, float] = {}
    for index in recompute_ids:
        again = legacy_invert(diffusion, model_fn, images[index].unsqueeze(0).to(device), args.noise_level)
        recompute[index] = float(np.max(np.abs(again.detach().float().cpu().numpy()[0] - latents[index])))

    latent_path = context.run_dir / "latents.npy"
    np.save(latent_path, latents)
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        z = latents[index]
        output_rows.append({
            "schema_version": 1,
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "cache_index": index,
            "x0_sha256": _array_sha256(images[index].numpy()),
            "latent_sha256": _array_sha256(z),
            "latent_finite": int(bool(np.isfinite(z).all())),
            "latent_l2": float(np.linalg.norm(z)),
            "latent_mean": float(z.mean()),
            "latent_std": float(z.std()),
            "latent_abs_max": float(np.abs(z).max()),
            "runtime_seconds_per_image": batch_runtime[index - index % args.batch_size],
            "recomputed": int(index in recompute),
            # Blank when not sampled; deliberately not a declared numeric field.
            "recompute_max_abs": recompute.get(index, ""),
        })

    shape_ok = latents.shape == (count, *CACHE_SHAPE)
    finite_ok = bool(np.isfinite(latents).all())
    recompute_max = max(recompute.values()) if recompute else math.nan
    recompute_ok = bool(recompute) and recompute_max <= CACHE_RECOMPUTE_TOLERANCE
    done = {
        "schema_version": 1,
        "task_id": context.task_id,
        "run_id": context.run_id,
        "status": "passed" if shape_ok and finite_ok and recompute_ok else "failed",
        "split": str(args.split),
        "split_sha256": rows[0]["split_sha256"],
        "sample_count": count,
        "sample_ids": [row["sample_id"] for row in selected],
        "pipeline": LEGACY_PIPELINE,
        "pass": "legacy_invert (ddim_anonymization_forward, null=True, clip_denoised=True)",
        "noise_level": args.noise_level,
        "batch_size": args.batch_size,
        "dtype": "float32",
        "latent_shape": list(latents.shape),
        "cache_file": str(latent_path),
        "cache_file_sha256": sha256_file(latent_path),
        "checks": {
            "shape_ok": shape_ok,
            "finite_ok": finite_ok,
            "recompute_batch_size": 1,
            "recompute_seed": args.seed,
            "recompute_sample_ids": [selected[i]["sample_id"] for i in recompute_ids],
            "recompute_max_abs": recompute_max,
            "recompute_tolerance": CACHE_RECOMPUTE_TOLERANCE,
            "recompute_ok": recompute_ok,
            "recompute_bit_exact_count": sum(value == 0.0 for value in recompute.values()),
        },
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "raw_key_material_recorded": False,
        "key_independent": "x_T does not depend on the key; P/S0/S1/S2 reuse this cache",
    }
    args.done_output.parent.mkdir(parents=True, exist_ok=True)
    args.done_output.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(done["checks"], indent=2))
    if done["status"] != "passed":
        raise RuntimeError(f"E2.3 cache checks failed: {done['checks']}")
    numeric = [
        "cache_index", "latent_finite", "latent_l2", "latent_mean", "latent_std",
        "latent_abs_max", "runtime_seconds_per_image", "recomputed",
    ]
    return output_rows, numeric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "p-smoke", "cache"))
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1911)
    parser.add_argument("--benchmark-timestep", type=int, default=250)
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--benchmark-output", type=Path, default=Path("results/AF021_anonymization_benchmark.json"))
    parser.add_argument("--smoke-output", type=Path, default=Path("results/AF017_P_anonymization_smoke.csv"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("image/AF017_P_anonymization_smoke.png"))
    parser.add_argument("--batch-size", type=int, default=1, help="cache mode batch size")
    parser.add_argument("--limit", type=int, default=0, help="cache mode: only the first N rows (smoke)")
    parser.add_argument("--recompute-count", type=int, default=20, help="cache mode: seeded batch-1 recomputations")
    parser.add_argument("--done-output", type=Path, default=Path("results/E2.3_DONE.json"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default=None, help="defaults to AF021_ANON_BENCH / AF017_P_SMOKE / E2.3")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    task_id = args.task_id or {
        "benchmark": "AF021_ANON_BENCH", "p-smoke": "AF017_P_SMOKE", "cache": "E2.3",
    }[args.mode]
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = load_split(args.split)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        config.update(task_id=task_id, pipeline=LEGACY_PIPELINE, guidance_scale=LEGACY_GUIDANCE_SCALE)
        context = open_run(task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        runner = {"benchmark": run_benchmark, "p-smoke": run_p_smoke, "cache": run_cache}[args.mode]
        per_sample, numeric = runner(args, rows, diffusion, model_fn, device, context)
        manifest = common_manifest(args, rows)
        manifest.update(
            scheme={"p-smoke": "P", "benchmark": "P/S0/S1 shared passes",
                    "cache": "key-independent x_T shared by P/S0/S1/S2a/S2"}[args.mode],
            seed_list=[args.seed] * len(per_sample),
        )
        finalize_run(
            context, Path(__file__), per_sample, numeric, manifest,
            stdout_capture.getvalue(), stderr_capture.getvalue(),
        )


if __name__ == "__main__":
    main()
