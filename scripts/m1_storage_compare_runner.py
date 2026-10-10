#!/usr/bin/env python3
"""AF-024: P/S0/S1 dev round-trip under the range-preserving M1 storage protocol.

Re-runs the T2-WB / M1 path of E2.5 with ``range_preserving_png/v1`` instead of
the predecessor's min-max PNG, starting from the same E2.3 dev cache, and joins
the E2.5 legacy-PNG and float hand-off results (read-only) so the table shows
the three storage columns side by side.  E2.5 itself is not rewritten.

The anonymous images are regenerated at batch 1; each must reproduce the E2.5
``anon_vs_x0_psnr_db`` value exactly, which ties the joined rows to the same
images.  lo/hi are public storage metadata, so the T2-WB attacker view is also
measured on the range-preserving read-back.

  scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id AF024_M1_STORAGE \\
      --validate-artifacts -- python -m scripts.m1_storage_compare_runner \\
      --checkpoint <ckpt> --latent-cache artifacts/runs/E2.3/<dev cache run> \\
      --e2-5-run artifacts/runs/E2.5/<E2.5 run>
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

from scripts.e1_ddim_runner import Tee, create_runtime, load_split, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics, latent_pair_metrics
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING,
    LEGACY_GUIDANCE_SCALE,
    LEGACY_PIPELINE,
    apply_legacy_key,
    legacy_generate,
    legacy_invert,
)
from scripts.e2_roundtrip_runner import SCHEMES, _timed, load_latent_cache
from scripts.m1_storage import QUANTIZATION, RANGE_PRESERVING_PNG, range_preserving_png_handoff
from scripts.run_artifacts import finalize_run, open_run, sha256_file


BOOTSTRAP_RESAMPLES = 10_000
# storage column -> {table metric name: per-sample column}
STORAGE_COLUMNS = {
    "legacy_png": "m1png",
    "range_preserving_png": "m1range",
    "float": "m1float",
}
TABLE_METRICS = (
    "vs_x0_psnr_db", "vs_x0_ssim", "vs_x0_linf", "vs_x0_uint8_pixel_equality_rate",
    "vs_p_output_psnr_db", "decrypted_latent_cosine", "decrypted_latent_rmse",
    "decrypted_latent_lowfreq32_cosine", "decrypted_latent_abs_pearson",
    "attacker_latent_abs_pearson",
)
JOINED_FROM_E2_5 = tuple(
    f"{prefix}_{metric}" for prefix in ("m1png", "m1float") for metric in TABLE_METRICS
)


def load_e2_5_rows(run_dir: Path) -> tuple[str, dict[str, dict[str, str]]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_id") != "E2.5":
        raise ValueError(f"{run_dir} is not an E2.5 run")
    with (run_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    return manifest["run_id"], rows


@torch.inference_mode()
def run_compare(args, rows, latents, e2_5_rows, diffusion, model_fn, device):
    output_rows: list[dict[str, Any]] = []
    for image_index, row in enumerate(rows):
        x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
        z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
        p_output = None
        for scheme in SCHEMES:
            sample_id = f"{row['sample_id']}|{scheme}"
            reference = e2_5_rows[sample_id]
            z_ano = apply_legacy_key(z, args.source_root, scheme, "anonymize")
            anonymous, t_anon = _timed(device, legacy_generate, diffusion, model_fn, z_ano, args.noise_level)
            if scheme == "P":
                p_output = anonymous
            anon_metrics = image_pair_metrics("anon_vs_x0", x0, anonymous)
            if anon_metrics["anon_vs_x0_psnr_db"] != float(reference["anon_vs_x0_psnr_db"]):
                raise RuntimeError(f"anonymous image differs from E2.5 for {sample_id}")
            stored, storage = range_preserving_png_handoff(anonymous)
            z_hat_ano, t_inv = _timed(device, legacy_invert, diffusion, model_fn, stored, args.noise_level)
            z_hat = apply_legacy_key(z_hat_ano, args.source_root, scheme, "deanonymize")
            recovered, t_gen = _timed(device, legacy_generate, diffusion, model_fn, z_hat, args.noise_level)
            if not all(torch.isfinite(value).all() for value in (anonymous, stored, z_hat_ano, z_hat, recovered)):
                raise RuntimeError(f"NaN/Inf for {sample_id}")
            anonymous_array = anonymous.detach().float().cpu().numpy()
            record: dict[str, Any] = {
                "schema_version": 1,
                "sample_id": sample_id,
                "image_id": row["sample_id"],
                "label": row["label"],
                "scheme": scheme,
                "seed": args.seed,
                "noise_level": args.noise_level,
                "anon_matches_e2_5": 1,
                "anonymous_float32_sha256": hashlib.sha256(np.ascontiguousarray(anonymous_array).tobytes()).hexdigest(),
                "storage_protocol": storage["storage_protocol"],
                "storage_lo": storage["lo"],
                "storage_hi": storage["hi"],
                "storage_lo_float32_bits": storage["lo_float32_bits"],
                "storage_hi_float32_bits": storage["hi_float32_bits"],
                "storage_png_sha256": storage["png_sha256"],
                "storage_png_bytes": storage["png_bytes"],
                "runtime_seconds_anonymous_generation": t_anon,
                "runtime_seconds_m1range_inversion": t_inv,
                "runtime_seconds_m1range_generation": t_gen,
                "anon_vs_x0_psnr_db": anon_metrics["anon_vs_x0_psnr_db"],
            }
            record.update(image_pair_metrics("range_png_vs_anonymous", anonymous, stored))
            record.update(latent_pair_metrics("m1range_attacker_latent", z_ano, z_hat_ano))
            record.update(latent_pair_metrics("m1range_decrypted_latent", z, z_hat))
            record.update(image_pair_metrics("m1range_vs_x0", x0, recovered))
            record.update(image_pair_metrics("m1range_vs_p_output", p_output, recovered))
            # Joined read-only from E2.5 (same anonymous image, verified above).
            for column in JOINED_FROM_E2_5:
                record[f"e2_5_{column}"] = float(reference[column])
            output_rows.append(record)
            print(
                f"{sample_id}: range [{storage['lo']:.3f},{storage['hi']:.3f}] "
                f"m1range_psnr={record['m1range_vs_x0_psnr_db']:.2f} "
                f"legacy={record['e2_5_m1png_vs_x0_psnr_db']:.2f} float={record['e2_5_m1float_vs_x0_psnr_db']:.2f}"
            )
    return output_rows


def comparison_table(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    table = []
    for scheme in SCHEMES:
        selected = [row for row in rows if row["scheme"] == scheme]
        draws = rng.integers(0, len(selected), size=(BOOTSTRAP_RESAMPLES, len(selected)))
        for storage, prefix in STORAGE_COLUMNS.items():
            for metric in TABLE_METRICS:
                column = f"{prefix}_{metric}" if storage == "range_preserving_png" else f"e2_5_{prefix}_{metric}"
                values = np.array([float(row[column]) for row in selected])
                means = values[draws].mean(axis=1)
                table.append({
                    "scheme": scheme,
                    "storage": storage,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "ci95_low": float(np.percentile(means, 2.5)),
                    "ci95_high": float(np.percentile(means, 97.5)),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "source": "AF024 run" if storage == "range_preserving_png" else "E2.5 run (joined)",
                })
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--e2-5-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1911)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-sample-output", type=Path, default=Path("results/AF024_m1_storage_per_sample.csv"))
    parser.add_argument("--table-output", type=Path, default=Path("paper_assets/tables/table_m1_storage_comparison.csv"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="AF024_M1_STORAGE")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = load_split(args.split)
        latents = load_latent_cache(args.latent_cache, rows)
        e2_5_run_id, e2_5_rows = load_e2_5_rows(args.e2_5_run)
        if args.limit:
            rows, latents = rows[: args.limit], latents[: args.limit]
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(storage_protocol=RANGE_PRESERVING_PNG, quantization=QUANTIZATION, schemes=list(SCHEMES))
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        per_sample = run_compare(args, rows, latents, e2_5_rows, diffusion, model_fn, device)

        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(per_sample)
        table = comparison_table(per_sample, args.seed)
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        with args.table_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
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
            "storage_protocol": RANGE_PRESERVING_PNG,
            "quantization": QUANTIZATION,
            "latent_cache_run_id": cache_manifest["run_id"],
            "latent_cache_sha256": sha256_file(args.latent_cache / "latents.npy"),
            "joined_e2_5_run_id": e2_5_run_id,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "raw_key_material_recorded": False,
        }
        finalize_run(
            context, Path(__file__), per_sample, numeric, manifest,
            stdout_capture.getvalue(), stderr_capture.getvalue(),
        )


if __name__ == "__main__":
    main()
