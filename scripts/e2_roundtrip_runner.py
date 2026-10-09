#!/usr/bin/env python3
"""E2.5: complete P/S0/S1 dev round-trip on the legacy anonymization pipeline.

Starts from the E2.3 latent cache (key-independent x_T) and, per image and
scheme, reports three paths separately (PROPOSAL 7.2, reports/t2wb_protocol.md):

  transform-only  z -> Enc -> Dec                       (latent MaxAbs / bit-exact)
  T1              exact z_ano -> Dec -> generation      (exact latent payload)
  T2-WB / M1      anonymous image -> storage hand-off -> target-model inversion
                  -> Dec -> generation.  The PNG hand-off reproduces the
                  predecessor's per-image min-max uint8 PNG and is the primary
                  path; the float hand-off is a diagnostic upper bound.

Image-space metrics are primary; latent metrics are diagnostics (AF-019).  For
each scheme the decrypted-latent fidelity is the scheme-specific positive-control
quantity of the T2-WB Inconclusive rule (AUD-20261010-02).  Image metrics are
reported against x0 and against the P output (the base-flow output, i.e. the
predecessor's comparison style; AF-022).

Example:
  scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id E2.5 \\
      --validate-artifacts -- python -m scripts.e2_roundtrip_runner \\
      --checkpoint <ckpt> --latent-cache artifacts/runs/E2.3/<dev cache run>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.e1_ddim_runner import Tee, create_runtime, load_split, preprocess
from scripts.e2_anonymization_runner import (
    _sync,
    _to_u8,
    image_pair_metrics,
    latent_pair_metrics,
    legacy_png_handoff,
)
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING,
    LEGACY_GUIDANCE_SCALE,
    LEGACY_PIPELINE,
    apply_legacy_key,
    legacy_generate,
    legacy_invert,
)
from scripts.run_artifacts import finalize_run, open_run, sha256_file


SCHEMES = ("P", "S0", "S1")  # P first: its output is the reference for S0/S1
GRID_ROWS = (0, 10)  # pre-specified: first healthy and first effusion dev image
BOOTSTRAP_RESAMPLES = 10_000
SUMMARY_METRICS = (
    "anon_vs_x0_psnr_db", "anon_vs_x0_ssim",
    "t1_vs_x0_psnr_db", "t1_vs_x0_ssim", "t1_vs_x0_linf",
    "m1png_vs_x0_psnr_db", "m1png_vs_x0_ssim", "m1png_vs_x0_linf",
    "m1png_vs_x0_uint8_pixel_equality_rate", "m1png_vs_p_output_psnr_db",
    "m1png_vs_p_output_cosine",
    "m1float_vs_x0_psnr_db", "m1float_vs_x0_ssim",
    "transform_latent_max_abs", "transform_latent_float32_bit_exact_rate",
    "m1png_decrypted_latent_cosine", "m1png_decrypted_latent_rmse",
    "m1png_decrypted_latent_lowfreq32_cosine", "m1png_decrypted_latent_abs_pearson",
    "m1png_attacker_latent_cosine", "m1png_attacker_latent_abs_pearson",
    "m1float_decrypted_latent_cosine", "m1float_decrypted_latent_rmse",
)


def load_latent_cache(cache_dir: Path, rows: list[dict[str, str]]) -> np.ndarray:
    """Load an E2.3 cache and verify split, order and per-latent hashes."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_id") != "E2.3":
        raise ValueError(f"{cache_dir} is not an E2.3 cache run")
    if manifest.get("dataset_split_hash") != rows[0]["split_sha256"]:
        raise ValueError("cache split hash differs from the requested split")
    with (cache_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    if [entry["sample_id"] for entry in index] != [row["sample_id"] for row in rows]:
        raise ValueError("cache sample order differs from the split")
    latents = np.load(cache_dir / "latents.npy")
    for entry in index:
        digest = hashlib.sha256(
            np.ascontiguousarray(latents[int(entry["cache_index"])]).tobytes()
        ).hexdigest()
        if digest != entry["latent_sha256"]:
            raise ValueError(f"latent hash mismatch for {entry['sample_id']}")
    return latents


def _timed(device: torch.device, function, *arguments):
    _sync(device)
    start = time.perf_counter()
    value = function(*arguments)
    _sync(device)
    return value, time.perf_counter() - start


def handoff_path(prefix, diffusion, model_fn, args, device, scheme, x0, z, z_ano, anonymous, p_output, handoff):
    """Anonymous image -> hand-off -> inversion -> inverse key -> generation."""
    stored = legacy_png_handoff(anonymous) if handoff == "png" else anonymous
    z_hat_ano, t_inv = _timed(device, legacy_invert, diffusion, model_fn, stored, args.noise_level)
    z_hat = apply_legacy_key(z_hat_ano, args.source_root, scheme, "deanonymize")
    recovered, t_gen = _timed(device, legacy_generate, diffusion, model_fn, z_hat, args.noise_level)
    record: dict[str, Any] = {
        f"runtime_seconds_{prefix}_inversion": t_inv,
        f"runtime_seconds_{prefix}_generation": t_gen,
    }
    # Attacker view: re-inverted ciphertext vs the exact ciphertext.
    record.update(latent_pair_metrics(f"{prefix}_attacker_latent", z_ano, z_hat_ano))
    # Legitimate decryption: decrypted latent vs original latent (positive-control quantity).
    record.update(latent_pair_metrics(f"{prefix}_decrypted_latent", z, z_hat))
    record.update(image_pair_metrics(f"{prefix}_vs_x0", x0, recovered))
    record.update(image_pair_metrics(f"{prefix}_vs_p_output", p_output, recovered))
    tensors = (stored, z_hat_ano, z_hat, recovered)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError(f"NaN/Inf in {prefix} path for scheme {scheme}")
    return record, recovered


@torch.inference_mode()
def run_roundtrip(args, rows, latents, diffusion, model_fn, device):
    output_rows: list[dict[str, Any]] = []
    grid: dict[tuple[int, str], list[np.ndarray]] = {}
    for image_index, row in enumerate(rows):
        x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
        z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
        p_output = None
        for scheme in SCHEMES:
            torch.manual_seed(args.seed)
            z_ano = apply_legacy_key(z, args.source_root, scheme, "anonymize")
            z_roundtrip = apply_legacy_key(z_ano, args.source_root, scheme, "deanonymize")
            anonymous, t_anon = _timed(device, legacy_generate, diffusion, model_fn, z_ano, args.noise_level)
            if scheme == "P":
                p_output = anonymous
            transform_exact = bool(torch.equal(z_roundtrip, z))
            # T1: exact decrypted latent.  When the transform round-trips
            # bit-exactly the generation input equals P's, so P's output is reused.
            if transform_exact:
                t1_output, t_t1 = p_output, 0.0
            else:
                t1_output, t_t1 = _timed(device, legacy_generate, diffusion, model_fn, z_roundtrip, args.noise_level)
            record: dict[str, Any] = {
                "schema_version": 1,
                "sample_id": f"{row['sample_id']}|{scheme}",
                "image_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "label": row["label"],
                "scheme": scheme,
                "seed": args.seed,
                "noise_level": args.noise_level,
                "transform_bit_exact": int(transform_exact),
                "t1_reused_p_output": int(transform_exact),
                "runtime_seconds_anonymous_generation": t_anon,
                "runtime_seconds_t1_generation": t_t1,
            }
            transform = latent_pair_metrics("transform_latent", z, z_roundtrip)
            record["transform_latent_max_abs"] = transform["transform_latent_max_abs"]
            record["transform_latent_float32_bit_exact_rate"] = transform["transform_latent_float32_bit_exact_rate"]
            record.update(image_pair_metrics("anon_vs_x0", x0, anonymous))
            record.update(image_pair_metrics("t1_vs_x0", x0, t1_output))
            png_record, m1_png = handoff_path(
                "m1png", diffusion, model_fn, args, device, scheme, x0, z, z_ano, anonymous, p_output, "png"
            )
            float_record, m1_float = handoff_path(
                "m1float", diffusion, model_fn, args, device, scheme, x0, z, z_ano, anonymous, p_output, "float"
            )
            record.update(png_record)
            record.update(float_record)
            if not torch.isfinite(anonymous).all():
                raise RuntimeError(f"NaN/Inf anonymous output for {row['sample_id']} {scheme}")
            output_rows.append(record)
            if image_index in GRID_ROWS:
                grid[(image_index, scheme)] = [
                    value[0, 0].detach().float().cpu().numpy()
                    for value in (x0, anonymous, t1_output, m1_png)
                ]
            print(
                f"{row['sample_id']} {scheme}: anon_psnr={record['anon_vs_x0_psnr_db']:.2f} "
                f"t1_psnr={record['t1_vs_x0_psnr_db']:.2f} m1png_psnr={record['m1png_vs_x0_psnr_db']:.2f} "
                f"dec_cos={record['m1png_decrypted_latent_cosine']:.4f}"
            )
    return output_rows, grid


def bootstrap_summary(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    summary = []
    for scheme in SCHEMES:
        selected = [row for row in rows if row["scheme"] == scheme]
        draws = rng.integers(0, len(selected), size=(BOOTSTRAP_RESAMPLES, len(selected)))
        for metric in SUMMARY_METRICS:
            values = np.array([float(row[metric]) for row in selected])
            means = values[draws].mean(axis=1)
            summary.append({
                "scheme": scheme,
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "ci95_low": float(np.percentile(means, 2.5)),
                "ci95_high": float(np.percentile(means, 97.5)),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return summary


def save_grid(grid, rows, output: Path) -> None:
    columns = ("original", "anonymous", "T1 recovered", "M1 PNG recovered", "|x0-M1 PNG| 0..0.1")
    keys = [(index, scheme) for index in GRID_ROWS for scheme in SCHEMES if (index, scheme) in grid]
    canvas = Image.new("RGB", (256 * len(columns), 286 * len(keys)), "white")
    draw = ImageDraw.Draw(canvas)
    for position, key in enumerate(keys):
        original, anonymous, t1_output, m1_png = grid[key]
        diff = np.rint(np.clip(np.abs(m1_png - original) / 0.1, 0, 1) * 255).astype(np.uint8)
        y0 = position * 286
        for column, array in enumerate((_to_u8(original), _to_u8(anonymous), _to_u8(t1_output), _to_u8(m1_png), diff)):
            canvas.paste(Image.fromarray(array, mode="L").convert("RGB"), (column * 256, y0 + 30))
        draw.text((4, y0 + 5), f"{rows[key[0]]['sample_id']} {key[1]}: " + " | ".join(columns), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--latent-cache", type=Path, required=True, help="E2.3 cache run directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1911)
    parser.add_argument("--per-sample-output", type=Path, default=Path("results/E2.5_roundtrip_per_sample.csv"))
    parser.add_argument("--table-output", type=Path, default=Path("paper_assets/tables/table_baseline_correctness.csv"))
    parser.add_argument("--grid-output", type=Path, default=Path("image/E2.5_roundtrip_grid.png"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="E2.5")
    parser.add_argument("--limit", type=int, default=0, help="only the first N images (smoke)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = load_split(args.split)
        latents = load_latent_cache(args.latent_cache, rows)
        if args.limit:
            rows, latents = rows[: args.limit], latents[: args.limit]
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(pipeline=LEGACY_PIPELINE, guidance_scale=LEGACY_GUIDANCE_SCALE, schemes=list(SCHEMES))
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        per_sample, grid = run_roundtrip(args, rows, latents, diffusion, model_fn, device)

        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(per_sample)
        summary = bootstrap_summary(per_sample, args.seed)
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        with args.table_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
        save_grid(grid, rows, args.grid_output)

        numeric = [
            key for key, value in per_sample[0].items()
            if key not in {"schema_version", "seed", "noise_level"}
            and isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        cache_manifest = json.loads((args.latent_cache / "manifest.json").read_text(encoding="utf-8"))
        manifest = {
            "dataset_split_hash": rows[0]["split_sha256"],
            "checkpoint_hash": sha256_file(args.checkpoint),
            "source_sampler_sha256": sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py"),
            "seed_list": [args.seed] * len(per_sample),
            "steps": args.noise_level,
            "scheme": "P/S0/S1",
            "rounds": 0,
            "nonce_mode": "none",
            "container_version": "none",
            "pipeline": LEGACY_PIPELINE,
            "guidance_scale": LEGACY_GUIDANCE_SCALE,
            "generation_conditioning": GENERATION_CONDITIONING,
            "inversion_conditioning": "null=True (unconditional)",
            "latent_cache_run_id": cache_manifest["run_id"],
            "latent_cache_sha256": sha256_file(args.latent_cache / "latents.npy"),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "raw_key_material_recorded": False,
        }
        finalize_run(
            context, Path(__file__), per_sample, numeric, manifest,
            stdout_capture.getvalue(), stderr_capture.getvalue(),
        )


if __name__ == "__main__":
    main()
