#!/usr/bin/env python3
"""GPU dev evaluation for D6.4 generation quality and D6.7 S2 integration.

``quality`` generates P once and S2a/S2's shared anonymous image for every
R/permutation cell on the 20-image dev split.  ``integration`` exercises the
authenticated M2 and range-preserving M1 open paths on four fixed dev images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.e1_ddim_runner import Tee, create_runtime, load_split, preprocess
from scripts.e2_anonymization_runner import _to_u8, image_pair_metrics, latent_pair_metrics
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING,
    LEGACY_GUIDANCE_SCALE,
    LEGACY_PIPELINE,
    legacy_generate,
    legacy_invert,
)
from scripts.e2_roundtrip_runner import _timed, load_latent_cache
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import finalize_run, open_run, sha256_file
from src.crypto.container import VERSION
from src.crypto.s2 import open_anonymous_image, open_latent, seal_anonymous_image, seal_latent
from src.crypto.sot import transform


SEED = 1911
ROUNDS = (1, 2, 3, 4)
PERMUTATIONS = (False, True)
INTEGRATION_ROWS = (0, 1, 10, 11)
BOOTSTRAP_RESAMPLES = 10_000
FIXTURE_KEY = hashlib.sha256(b"cxr-s2/D6/public-reproducibility-fixture/v1").digest()


def nonce_for(label: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/D6/{label}".encode()).digest()[:16]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def numeric_fields(row: dict[str, Any]) -> list[str]:
    excluded = {"schema_version", "seed", "noise_level", "rounds", "permutation"}
    return [
        key for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def bootstrap_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "anon_vs_x0_psnr_db", "anon_vs_x0_ssim", "anon_vs_x0_cosine",
        "anon_vs_p_psnr_db", "anon_vs_p_ssim", "anon_vs_p_cosine",
        "runtime_seconds_transform", "runtime_seconds_generation",
    )
    output = []
    for use_permutation in PERMUTATIONS:
        for rounds in ROUNDS:
            selected = [
                row for row in rows
                if row["rounds"] == rounds and row["permutation"] == int(use_permutation)
            ]
            rng = np.random.default_rng(SEED + rounds + 100 * int(use_permutation))
            draws = rng.integers(0, len(selected), (BOOTSTRAP_RESAMPLES, len(selected)))
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                means = values[draws].mean(axis=1)
                output.append({
                    "rounds": rounds, "permutation": use_permutation,
                    "metric": metric, "n": len(values), "mean": float(values.mean()),
                    "ci95_low": float(np.percentile(means, 2.5)),
                    "ci95_high": float(np.percentile(means, 97.5)),
                    "median": float(np.median(values)), "min": float(values.min()),
                    "max": float(values.max()),
                })
    return output


@torch.inference_mode()
def quality_rows(args, rows, latents, diffusion, model_fn, device) -> list[dict[str, Any]]:
    output = []
    for image_index, row in enumerate(rows):
        x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
        z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
        p_output, p_seconds = _timed(device, legacy_generate, diffusion, model_fn, z, args.noise_level)
        for use_permutation in PERMUTATIONS:
            for rounds in ROUNDS:
                nonce = nonce_for(f"D6.true/{row['sample_id']}/R{rounds}/P{int(use_permutation)}")
                qz, transform_seconds = _timed(
                    device,
                    lambda: transform(
                        z, FIXTURE_KEY, nonce, rounds=rounds,
                        use_permutation=use_permutation,
                    ),
                )
                anonymous, generation_seconds = _timed(
                    device, legacy_generate, diffusion, model_fn, qz, args.noise_level,
                )
                record: dict[str, Any] = {
                    "schema_version": 1,
                    "sample_id": f"{row['sample_id']}|R{rounds}|P{int(use_permutation)}",
                    "image_id": row["sample_id"], "patient_id": row["patient_id"],
                    "label": row["label"], "rounds": rounds,
                    "permutation": int(use_permutation), "seed": args.seed,
                    "noise_level": args.noise_level,
                    "runtime_seconds_p_generation": p_seconds,
                    "runtime_seconds_transform": transform_seconds,
                    "runtime_seconds_generation": generation_seconds,
                    "output_finite": int(bool(torch.isfinite(anonymous).all())),
                }
                record.update(image_pair_metrics("anon_vs_x0", x0, anonymous))
                record.update(image_pair_metrics("anon_vs_p", p_output, anonymous))
                output.append(record)
                print(
                    f"{record['sample_id']}: PSNR(x0)={record['anon_vs_x0_psnr_db']:.3f} "
                    f"SSIM(x0)={record['anon_vs_x0_ssim']:.4f}"
                )
    return output


def save_integration_grid(items: list[tuple[str, list[np.ndarray]]], output: Path) -> None:
    columns = ("original", "anonymous", "M2/T1 recovered", "M1 recovered", "|x0-M1| 0..0.1")
    canvas = Image.new("RGB", (256 * len(columns), 286 * len(items)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (sample_id, panels) in enumerate(items):
        original, anonymous, t1, m1 = panels
        difference = np.rint(np.clip(np.abs(m1 - original) / 0.1, 0, 1) * 255).astype(np.uint8)
        for column, array in enumerate((_to_u8(original), _to_u8(anonymous), _to_u8(t1), _to_u8(m1), difference)):
            canvas.paste(Image.fromarray(array, mode="L").convert("RGB"), (256 * column, 286 * row_index + 30))
        draw.text((4, 286 * row_index + 5), f"{sample_id}: " + " | ".join(columns), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output)


@torch.inference_mode()
def integration_rows(args, rows, latents, diffusion, model_fn, device):
    output, grid = [], []
    model_hash = sha256_file(args.checkpoint)
    for image_index in INTEGRATION_ROWS:
        row = rows[image_index]
        x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
        z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
        m1_nonce = nonce_for(f"D6.7/M1/{row['sample_id']}")
        m2_nonce = nonce_for(f"D6.7/M2/{row['sample_id']}")
        qz = transform(
            z, FIXTURE_KEY, m1_nonce, rounds=args.rounds,
            use_permutation=args.use_permutation,
        )
        anonymous, anonymous_seconds = _timed(
            device, legacy_generate, diffusion, model_fn, qz, args.noise_level,
        )
        aad = {
            "study_instance_uid": f"research:{row['patient_id']}",
            "sop_instance_uid": f"research:{row['sample_id']}",
            "model_sha256": model_hash,
            "sampler_version": LEGACY_PIPELINE,
        }
        m1 = seal_anonymous_image(
            anonymous, FIXTURE_KEY, m1_nonce, rounds=args.rounds,
            use_permutation=args.use_permutation, latent_shape=tuple(z.shape),
            latent_dtype=str(z.detach().cpu().numpy().dtype), aad=aad,
        )
        inversion_seconds = [0.0]

        def inverter(image):
            inverted, inversion_seconds[0] = _timed(
                device, legacy_invert, diffusion, model_fn, image, args.noise_level,
            )
            return inverted

        decrypted_m1 = open_anonymous_image(m1, FIXTURE_KEY, inverter, device=device)
        recovered_m1, recovery_seconds = _timed(
            device, legacy_generate, diffusion, model_fn, decrypted_m1, args.noise_level,
        )
        m2 = seal_latent(
            z, FIXTURE_KEY, m2_nonce, rounds=args.rounds,
            use_permutation=args.use_permutation, aad=aad,
        )
        decrypted_m2 = open_latent(m2, FIXTURE_KEY, device=device)
        recovered_m2, t1_seconds = _timed(
            device, legacy_generate, diffusion, model_fn, decrypted_m2, args.noise_level,
        )
        record: dict[str, Any] = {
            "schema_version": 1, "sample_id": row["sample_id"],
            "patient_id": row["patient_id"], "label": row["label"],
            "rounds": args.rounds, "permutation": int(args.use_permutation),
            "seed": args.seed, "noise_level": args.noise_level,
            "m1_container_bytes": len(m1), "m2_container_bytes": len(m2),
            "m1_container_sha256": hashlib.sha256(m1).hexdigest(),
            "m2_container_sha256": hashlib.sha256(m2).hexdigest(),
            "runtime_seconds_anonymous_generation": anonymous_seconds,
            "runtime_seconds_m1_inversion": inversion_seconds[0],
            "runtime_seconds_m1_recovery_generation": recovery_seconds,
            "runtime_seconds_m2_t1_generation": t1_seconds,
            "m1_authenticated_before_inversion": 1,
            "s2a_s2_anonymous_pixels_identical_by_construction": 1,
        }
        record.update(latent_pair_metrics("m2_decrypted_latent", z, decrypted_m2))
        record.update(latent_pair_metrics("m1_decrypted_latent", z, decrypted_m1))
        record.update(image_pair_metrics("anonymous_vs_x0", x0, anonymous))
        record.update(image_pair_metrics("m2_t1_vs_x0", x0, recovered_m2))
        record.update(image_pair_metrics("m1_vs_x0", x0, recovered_m1))
        if not all(torch.isfinite(value).all() for value in (qz, anonymous, decrypted_m1, recovered_m1, decrypted_m2, recovered_m2)):
            raise RuntimeError(f"non-finite D6.7 output for {row['sample_id']}")
        output.append(record)
        grid.append((row["sample_id"], [
            x0[0, 0].cpu().numpy(), anonymous[0, 0].cpu().numpy(),
            recovered_m2[0, 0].cpu().numpy(), recovered_m1[0, 0].cpu().numpy(),
        ]))
        print(
            f"{row['sample_id']}: M2 maxabs={record['m2_decrypted_latent_max_abs']:.3g} "
            f"M1 PSNR={record['m1_vs_x0_psnr_db']:.3f}"
        )
    return output, grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("quality", "integration"))
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--use-permutation", action="store_true")
    parser.add_argument("--quality-output", type=Path, default=Path("results/D6.4_generation_per_sample.csv"))
    parser.add_argument("--quality-table", type=Path, default=Path("paper_assets/tables/table_D6_generation_quality.csv"))
    parser.add_argument("--integration-output", type=Path, default=Path("results/D6.7_dev_integration.csv"))
    parser.add_argument("--grid-output", type=Path, default=Path("image/D6.7_s2_dev.png"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="")
    add_force_flag(parser)
    args = parser.parse_args()
    target_outputs = (
        [args.quality_output, args.quality_table] if args.mode == "quality"
        else [args.integration_output, args.grid_output]
    )
    refuse_overwrite(target_outputs, args.force)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = load_split(args.split)
        latents = load_latent_cache(args.latent_cache, rows)
        task_id = args.task_id or ("D6.4_QUALITY" if args.mode == "quality" else "D6.7_INTEGRATION")
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(task_id=task_id, pipeline=LEGACY_PIPELINE, guidance_scale=LEGACY_GUIDANCE_SCALE)
        context = open_run(task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        if args.mode == "quality":
            per_sample = quality_rows(args, rows, latents, diffusion, model_fn, device)
            write_csv(args.quality_output, per_sample)
            write_csv(args.quality_table, bootstrap_quality(per_sample))
            scheme, rounds = "S2a/S2", "1/2/3/4"
        else:
            per_sample, grid = integration_rows(args, rows, latents, diffusion, model_fn, device)
            write_csv(args.integration_output, per_sample)
            save_integration_grid(grid, args.grid_output)
            scheme, rounds = "S2", args.rounds
        cache_manifest = json.loads((args.latent_cache / "manifest.json").read_text(encoding="utf-8"))
        manifest = {
            "dataset_split_hash": rows[0]["split_sha256"],
            "checkpoint_hash": sha256_file(args.checkpoint),
            "source_sampler_sha256": sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py"),
            "seed_list": [args.seed] * len(per_sample), "steps": args.noise_level,
            "scheme": scheme, "rounds": rounds,
            "nonce_mode": "unique deterministic public D6 reproducibility fixture",
            "container_version": VERSION if args.mode == "integration" else "S2a transform-only",
            "pipeline": LEGACY_PIPELINE, "guidance_scale": LEGACY_GUIDANCE_SCALE,
            "generation_conditioning": GENERATION_CONDITIONING,
            "inversion_conditioning": "null=True (unconditional)",
            "latent_cache_run_id": cache_manifest["run_id"],
            "latent_cache_sha256": sha256_file(args.latent_cache / "latents.npy"),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES if args.mode == "quality" else 0,
            "raw_key_material_recorded": False,
            "public_fixture_key_sha256": hashlib.sha256(FIXTURE_KEY).hexdigest(),
        }
        finalize_run(
            context, Path(__file__), per_sample, numeric_fields(per_sample[0]), manifest,
            stdout_capture.getvalue(), stderr_capture.getvalue(),
        )


if __name__ == "__main__":
    main()
