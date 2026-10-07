#!/usr/bin/env python3
"""Reproducible E1 CFG-DDIM benchmark and four-image smoke test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
import os
import platform
import subprocess
from datetime import datetime, timezone
import types
from pathlib import Path

import numpy as np
import torch
import cv2
import PIL
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


MODEL_CONFIG = {
    "image_size": 256,
    "in_channels": 1,
    "num_channels": 128,
    "num_classes": 2,
    "class_cond": True,
    "num_res_blocks": 2,
    "num_heads": 1,
    "learn_sigma": True,
    "use_scale_shift_norm": False,
    "attention_resolutions": "16",
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "rescale_learned_sigmas": False,
    "rescale_timesteps": False,
    "timestep_respacing": "ddim1000",
}


class NullVisdom:
    """No-op Visdom replacement for headless experiments."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def install_headless_visdom() -> None:
    module = types.ModuleType("visdom")
    module.Visdom = NullVisdom
    sys.modules["visdom"] = module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["split_sha256"] for row in rows}) != 1:
        raise ValueError("split is empty or contains inconsistent split hashes")
    return rows


def preprocess(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"cannot read image: {path}")
    equalized = cv2.equalizeHist(raw)
    resized = cv2.resize(equalized, (256, 256), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    if not ok:
        raise ValueError(f"cannot encode preprocessed image: {path}")
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    if high <= low:
        raise ValueError(f"constant image: {path}")
    array = (array - low) / (high - low)
    return torch.from_numpy(array[None, ...]), array


def create_runtime(source_root: Path, checkpoint: Path, device: torch.device):
    install_headless_visdom()
    sys.path.insert(0, str(source_root.resolve()))
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    config = model_and_diffusion_defaults()
    config.update(MODEL_CONFIG)
    model, diffusion = create_model_and_diffusion(**config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    def model_fn(x, t, y=None, p_uncond=-1, null=False, clf_free=True):
        if y is None:
            raise ValueError("class label y is required")
        return model(x, t, y, p_uncond, null, clf_free)

    return model, diffusion, model_fn


@torch.inference_mode()
def one_cycle(diffusion, model_fn, x: torch.Tensor, timestep: int, guidance: float):
    kwargs = {
        "y": torch.zeros(x.shape[0], device=x.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    reverse_kwargs = dict(kwargs)
    reverse_kwargs["null"] = True
    t = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
    latent = diffusion.ddim_reverse_sample(
        model_fn, x, t, clip_denoised=True, model_kwargs=reverse_kwargs
    )["sample"]
    reconstruction = diffusion.ddim_sample(
        model_fn,
        latent,
        t,
        clip_denoised=True,
        model_kwargs=kwargs,
        guidance_scale=guidance,
    )["sample"]
    return latent, reconstruction


def git_value(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_run_artifacts(
    task_id: str,
    args,
    split_rows: list[dict[str, str]],
    per_sample_rows: list[dict[str, object]],
    numeric_fields: list[str],
    started_at: str,
) -> Path:
    """Write and immediately validate a complete WORKFLOW 3.2 run directory."""
    from artifact_schema import validate_run

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["model_config"] = MODEL_CONFIG
    config_bytes = (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{task_id}_{timestamp}_{config_hash[:8]}"
    run_dir = args.artifacts_root / task_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_bytes(config_bytes)
    with (run_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_sample_rows)
    numeric_means = {
        field: float(np.mean([float(row[field]) for row in per_sample_rows]))
        for field in numeric_fields
    }
    summary = {
        "schema_version": 2,
        "task_id": task_id,
        "run_id": run_id,
        "sample_count": len(per_sample_rows),
        "numeric_means": numeric_means,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 2,
        "task_id": task_id,
        "run_id": run_id,
        "config_hash": config_hash,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain=v1")),
        "dataset_split_hash": split_rows[0]["split_sha256"],
        "checkpoint_hash": sha256_file(args.checkpoint),
        "expected_samples": len(per_sample_rows),
        "sample_ids": [str(row["sample_id"]) for row in per_sample_rows],
        "command": " ".join(sys.argv),
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": 0,
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "dtype": "float32",
        "seed_list": [int(row.get("seed", args.seed)) for row in per_sample_rows],
        "steps": args.noise_level,
        "packages": {
            "numpy": np.__version__, "opencv": cv2.__version__,
            "pillow": PIL.__version__,
        },
        "scheme": "legacy_ddim_P",
        "rounds": 0,
        "nonce_mode": "none",
        "container_version": "none",
        "numeric_fields": numeric_fields,
        "failure_reason": "",
        "skip_reason": "",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "stdout.log").write_text(
        f"{task_id} completed {len(per_sample_rows)} rows\n", encoding="utf-8"
    )
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    validation = validate_run(run_dir)
    if not validation["valid"]:
        raise RuntimeError(f"invalid run artifacts: {validation['errors']}")
    print(f"validated run artifacts: {run_dir}")
    return run_dir


def run_benchmark(args, rows, diffusion, model_fn, device) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    tensors = [preprocess(Path(row["local_path"]))[0] for row in rows[:8]]
    results = []
    for batch_size in (1, 4, 8):
        batch = torch.stack(tensors[:batch_size]).to(device)
        entry = {"batch_size": batch_size, "feasible": False}
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            one_cycle(diffusion, model_fn, batch, args.benchmark_timestep, args.guidance_scale)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings = []
            for _ in range(args.benchmark_repeats):
                start = time.perf_counter()
                one_cycle(diffusion, model_fn, batch, args.benchmark_timestep, args.guidance_scale)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - start)
            entry.update(
                {
                    "feasible": True,
                    "seconds_per_reverse_reconstruct_cycle_mean": float(np.mean(timings)),
                    "seconds_per_reverse_reconstruct_cycle_std": float(np.std(timings)),
                    "estimated_seconds_noise_level_500": float(np.mean(timings) * 499),
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                }
            )
        except torch.OutOfMemoryError as error:
            entry["error"] = str(error)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(entry)

    feasible = [entry for entry in results if entry["feasible"]]
    recommended = max(feasible, key=lambda entry: entry["batch_size"])["batch_size"]
    payload = {
        "schema_version": 1,
        "task_id": "E1.2",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split_sha256": rows[0]["split_sha256"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "none",
        "benchmark_timestep": args.benchmark_timestep,
        "benchmark_repeats": args.benchmark_repeats,
        "guidance_scale": args.guidance_scale,
        "preprocessing": "legacy CheXpert: grayscale; histogram equalization; OpenCV INTER_AREA 256x256; JPEG quality 100 round-trip; per-image min-max [0,1]",
        "results": results,
        "recommended_batch_size": recommended,
        "formal_n_decision": "start with N=200; increase only after full-pipeline pilot confirms estimate",
        "interpretation": "The estimate multiplies one reverse+guided-reconstruction cycle by 499; full-run I/O and initialization are excluded.",
    }
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    artifact_rows = [dict(entry, sample_id=f"batch_{entry['batch_size']}") for entry in results]
    numeric = [
        "seconds_per_reverse_reconstruct_cycle_mean",
        "seconds_per_reverse_reconstruct_cycle_std",
        "estimated_seconds_noise_level_500",
        "peak_vram_bytes",
    ]
    write_run_artifacts("E1.2", args, rows, artifact_rows, numeric, started_at)


def pair_metrics(prefix: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    x = a.detach().float().cpu().numpy().ravel()
    y = b.detach().float().cpu().numpy().ravel()
    error = y - x
    mse = float(np.mean(error * error))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return {
        f"{prefix}_cosine": float(np.dot(x, y) / denom) if denom else 0.0,
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": math.sqrt(mse),
        f"{prefix}_max_abs": float(np.max(np.abs(error))),
        f"{prefix}_float32_bit_exact_rate": float(
            np.mean(x.view(np.uint32) == y.view(np.uint32))
        ),
    }


def metrics(original: torch.Tensor, reconstruction: torch.Tensor, latent: torch.Tensor) -> dict[str, float]:
    x = original.detach().float().cpu().numpy().ravel()
    y = reconstruction.detach().float().cpu().numpy().ravel()
    z = latent.detach().float().cpu().numpy().ravel()
    error = y - x
    mse = float(np.mean(error * error))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    x8 = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
    y8 = np.rint(np.clip(y, 0, 1) * 255).astype(np.uint8)
    return {
        "image_mse": mse,
        "image_rmse": math.sqrt(mse),
        "image_mae": float(np.mean(np.abs(error))),
        "image_max_abs": float(np.max(np.abs(error))),
        "image_psnr_db": float(-10.0 * math.log10(max(mse, np.finfo(float).tiny))),
        "image_ssim": float(structural_similarity(x.reshape(256, 256), y.reshape(256, 256), data_range=1.0)),
        "image_cosine": float(np.dot(x, y) / denom) if denom else 0.0,
        "image_uint8_pixel_equality_rate": float(np.mean(x8 == y8)),
        "image_float32_bit_exact_rate": float(np.mean(x.view(np.uint32) == y.view(np.uint32))),
        "latent_l2": float(np.linalg.norm(z)),
        "latent_mean": float(np.mean(z)),
        "latent_std": float(np.std(z)),
    }


@torch.inference_mode()
def invert_reconstruct(diffusion, model_fn, x0, noise_level, guidance, seed):
    """Call the legacy progressive sampler without reimplementing its timestep schedule."""
    torch.manual_seed(seed)
    if x0.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    kwargs = {
        "y": torch.zeros(x0.shape[0], device=x0.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    t0 = torch.zeros(x0.shape[0], device=x0.device, dtype=torch.long)
    x_noisy = diffusion.q_sample(x_start=x0, t=t0, noise=torch.randn_like(x0))
    latent = None
    reconstruction = None
    inversion_outputs = noise_level - 1
    for index, out in enumerate(diffusion.ddim_sample_loop_known_progressive(
        model_fn, tuple(x0.shape), time=noise_level, noise=x_noisy,
        clip_denoised=True, model_kwargs=kwargs, device=x0.device,
        guidance_scale=guidance,
    )):
        reconstruction = out["sample"]
        if index == inversion_outputs - 1:
            latent = reconstruction.clone()
    if latent is None or reconstruction is None:
        raise RuntimeError("legacy DDIM sampler returned no complete cycle")
    return latent, reconstruction


def normalize_u8(array: np.ndarray) -> np.ndarray:
    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint((array - low) / (high - low) * 255).astype(np.uint8)


def save_contact_sheet(items, output: Path) -> None:
    width, height = 256 * 3, len(items) * 286
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (sample_id, original, reconstruction) in enumerate(items):
        y0 = row_index * 286
        orig = normalize_u8(original)
        rec = normalize_u8(reconstruction)
        # Fixed 0..0.1 scale makes difference intensity comparable across rows.
        diff = np.rint(np.clip(np.abs(reconstruction - original) / 0.1, 0, 1) * 255).astype(np.uint8)
        for column, array in enumerate((orig, rec, diff)):
            panel = Image.fromarray(array, mode="L").convert("RGB")
            canvas.paste(panel, (column * 256, y0 + 30))
        draw.text((4, y0 + 5), f"{sample_id}: original | reconstruction | abs diff (0..0.1)", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_smoke(args, rows, diffusion, model_fn, device) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    chosen = [rows[0], rows[1], rows[10], rows[11]]
    output_rows = []
    contact_items = []
    for index, row in enumerate(chosen):
        tensor, original_array = preprocess(Path(row["local_path"]))
        x0 = tensor.unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        seed = args.seed + index
        latent, reconstruction = invert_reconstruct(
            diffusion, model_fn, x0, args.noise_level, args.guidance_scale, seed
        )
        latent_repeat, _ = invert_reconstruct(
            diffusion, model_fn, x0, args.noise_level, args.guidance_scale, seed
        )
        reinverted_latent, _ = invert_reconstruct(
            diffusion, model_fn, reconstruction, args.noise_level, args.guidance_scale, seed
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        tensors = (latent, reconstruction, latent_repeat, reinverted_latent)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"NaN/Inf detected for {row['sample_id']}")
        record = {
            "schema_version": 1,
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "split_sha256": row["split_sha256"],
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "seed": seed,
            "noise_level": args.noise_level,
            "guidance_scale": args.guidance_scale,
            "runtime_seconds": elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "finite": True,
        }
        record.update(metrics(x0, reconstruction, latent))
        record.update(pair_metrics("latent_repeat", latent, latent_repeat))
        record.update(pair_metrics("latent_reinversion", latent, reinverted_latent))
        if not all(math.isfinite(float(value)) for key, value in record.items() if key.startswith(("image_", "latent_"))):
            raise RuntimeError(f"non-finite metric for {row['sample_id']}")
        output_rows.append(record)
        rec_array = reconstruction[0, 0].detach().float().cpu().numpy()
        contact_items.append((row["sample_id"], original_array, rec_array))
        print(f"completed {row['sample_id']} in {elapsed:.3f}s")
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    with args.smoke_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    save_contact_sheet(contact_items, args.contact_sheet)
    numeric = [
        key for key, value in output_rows[0].items()
        if key.startswith(("image_", "latent_")) or key in ("runtime_seconds", "peak_vram_bytes")
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    write_run_artifacts("E1.4", args, rows, output_rows, numeric, started_at)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "smoke"))
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1911)
    parser.add_argument("--benchmark-timestep", type=int, default=250)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--benchmark-output", type=Path, default=Path("results/E1.2_benchmark.json"))
    parser.add_argument("--smoke-output", type=Path, default=Path("results/E1.4_ddim_smoke.csv"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("image/E1.4_ddim_smoke_contact_sheet.png"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for E1.2/E1.4")
    rows = load_split(args.split)
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    if args.mode == "benchmark":
        run_benchmark(args, rows, diffusion, model_fn, device)
    else:
        run_smoke(args, rows, diffusion, model_fn, device)


if __name__ == "__main__":
    main()
