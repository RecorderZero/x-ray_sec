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
import types
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw


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


def run_benchmark(args, rows, diffusion, model_fn, device) -> None:
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


def metrics(original: torch.Tensor, reconstruction: torch.Tensor, latent: torch.Tensor) -> dict[str, float]:
    x = original.detach().float().cpu().numpy().ravel()
    y = reconstruction.detach().float().cpu().numpy().ravel()
    z = latent.detach().float().cpu().numpy().ravel()
    error = y - x
    mse = float(np.mean(error * error))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return {
        "image_mse": mse,
        "image_rmse": math.sqrt(mse),
        "image_mae": float(np.mean(np.abs(error))),
        "image_max_abs": float(np.max(np.abs(error))),
        "image_psnr_db": float("inf") if mse == 0 else -10.0 * math.log10(mse),
        "image_cosine": float(np.dot(x, y) / denom) if denom else float("nan"),
        "latent_l2": float(np.linalg.norm(z)),
        "latent_mean": float(np.mean(z)),
        "latent_std": float(np.std(z)),
    }


@torch.inference_mode()
def invert_reconstruct(diffusion, model_fn, x0, noise_level, guidance, seed):
    torch.manual_seed(seed)
    if x0.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    kwargs = {
        "y": torch.zeros(x0.shape[0], device=x0.device, dtype=torch.int64),
        "p_uncond": -1,
        "clf_free": True,
    }
    reverse_kwargs = dict(kwargs)
    reverse_kwargs["null"] = True
    t0 = torch.zeros(x0.shape[0], device=x0.device, dtype=torch.long)
    latent = diffusion.q_sample(x_start=x0, t=t0, noise=torch.randn_like(x0))
    for timestep in range(noise_level - 1):
        t = torch.full((x0.shape[0],), timestep, device=x0.device, dtype=torch.long)
        latent = diffusion.ddim_reverse_sample(
            model_fn, latent, t, clip_denoised=True, model_kwargs=reverse_kwargs
        )["sample"]
    reconstruction = latent
    for timestep in reversed(range(noise_level - 1)):
        t = torch.full((x0.shape[0],), timestep, device=x0.device, dtype=torch.long)
        reconstruction = diffusion.ddim_sample(
            model_fn,
            reconstruction,
            t,
            clip_denoised=True,
            model_kwargs=kwargs,
            guidance_scale=guidance,
        )["sample"]
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
        diff = normalize_u8(np.abs(reconstruction - original))
        for column, array in enumerate((orig, rec, diff)):
            panel = Image.fromarray(array, mode="L").convert("RGB")
            canvas.paste(panel, (column * 256, y0 + 30))
        draw.text((4, y0 + 5), f"{sample_id}: original | reconstruction | abs diff", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_smoke(args, rows, diffusion, model_fn, device) -> None:
    chosen = [rows[0], rows[1], rows[10], rows[11]]
    output_rows = []
    contact_items = []
    for index, row in enumerate(chosen):
        tensor, original_array = preprocess(Path(row["local_path"]))
        x0 = tensor.unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        latent, reconstruction = invert_reconstruct(
            diffusion,
            model_fn,
            x0,
            args.noise_level,
            args.guidance_scale,
            args.seed + index,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if not torch.isfinite(latent).all() or not torch.isfinite(reconstruction).all():
            raise RuntimeError(f"NaN/Inf detected for {row['sample_id']}")
        record = {
            "schema_version": 1,
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "split_sha256": row["split_sha256"],
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "seed": args.seed + index,
            "noise_level": args.noise_level,
            "guidance_scale": args.guidance_scale,
            "runtime_seconds": elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "finite": True,
        }
        record.update(metrics(x0, reconstruction, latent))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "smoke"))
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1911)
    parser.add_argument("--benchmark-timestep", type=int, default=250)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--benchmark-output", type=Path, default=Path("results/E1.2_benchmark.json"))
    parser.add_argument("--smoke-output", type=Path, default=Path("results/E1.4_ddim_smoke.csv"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("image/E1.4_ddim_smoke_contact_sheet.png"))
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
