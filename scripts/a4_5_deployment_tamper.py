#!/usr/bin/env python3
"""A4.5 (deployment path): tamper the stored anonymous image, then run the full
predecessor decrypt pipeline and confirm nothing rejects it (AF-036).

For a few security_v1 images: generate the S0 anonymous image, store it as the
predecessor legacy min-max uint8 PNG, flip one stored pixel, then run the
deployment decrypt path end to end — DDIM inversion (target checkpoint) -> inverse
key -> generation — and confirm it completes with no verification/rejection
(accepted=True), quantifying the recovered-image change vs the untampered
deployment. Complements the container/latent-level table_tamper_baseline.csv.

  scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id A4.5_DEPLOY \
    --validate-artifacts -- python -m scripts.a4_5_deployment_tamper \
    --checkpoint <ckpt> --latent-cache artifacts/runs/E2.3/<security run>
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.e1_ddim_runner import Tee, create_runtime, load_split
from scripts.e2_anonymization_runner import legacy_png_handoff
from scripts.e2_legacy_wrapper import apply_legacy_key, legacy_generate, legacy_invert
from scripts.e2_roundtrip_runner import _timed, load_latent_cache
from scripts.run_artifacts import finalize_run, open_run, sha256_file

SCHEME = "S0"
TAMPER_IMAGES = (0, 199)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a.float() - b.float()) ** 2))
    return float(-10.0 * math.log10(max(mse, 1e-20)))


def _store_legacy_uint8(anonymous: torch.Tensor) -> np.ndarray:
    """Predecessor legacy storage: per-image min-max to uint8 (range discarded)."""
    image = anonymous[0, 0].detach().float().cpu().numpy()
    lo, hi = float(image.min()), float(image.max())
    return np.clip(np.round((image - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def _load_legacy_uint8(stored: np.ndarray, device: torch.device) -> torch.Tensor:
    low, high = float(stored.min()), float(stored.max())
    reloaded = (stored.astype(np.float32) - low) / (high - low)
    return torch.from_numpy(reloaded)[None, None].to(device)


@torch.inference_mode()
def decrypt_pipeline(args, diffusion, model_fn, stored_image: torch.Tensor):
    """Deployment decrypt: inversion -> inverse key -> generation. No verification."""
    z_hat_ano, t_inv = _timed(device_of(stored_image), legacy_invert, diffusion, model_fn, stored_image, args.noise_level)
    z_hat = apply_legacy_key(z_hat_ano, args.source_root, SCHEME, "deanonymize")
    recovered, t_gen = _timed(device_of(stored_image), legacy_generate, diffusion, model_fn, z_hat, args.noise_level)
    return recovered, t_inv + t_gen


def device_of(tensor: torch.Tensor) -> torch.device:
    return tensor.device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--table-output", type=Path, default=Path("paper_assets/tables/table_tamper_deployment.csv"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="A4.5_DEPLOY")
    try:
        from scripts.output_guard import add_force_flag, refuse_overwrite
    except ModuleNotFoundError:
        from output_guard import add_force_flag, refuse_overwrite
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.table_output], args.force)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows_split = load_split(args.split)
        latents = load_latent_cache(args.latent_cache, rows_split)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(scheme=SCHEME, tamper_images=list(TAMPER_IMAGES))
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)

        records: list[dict[str, Any]] = []
        for image_index in TAMPER_IMAGES:
            split_row = rows_split[image_index]
            z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
            z_ano = apply_legacy_key(z, args.source_root, SCHEME, "anonymize")
            anonymous, _ = _timed(device, legacy_generate, diffusion, model_fn, z_ano, args.noise_level)

            stored = _store_legacy_uint8(anonymous)
            clean_recovered, _ = decrypt_pipeline(args, diffusion, model_fn, _load_legacy_uint8(stored, device))

            # Tamper: flip one stored pixel by +128 (mod 256) at a fixed location.
            tampered = stored.copy()
            tampered[128, 128] = np.uint8((int(tampered[128, 128]) + 128) % 256)
            accepted = True
            try:
                tam_recovered, _ = decrypt_pipeline(args, diffusion, model_fn, _load_legacy_uint8(tampered, device))
            except Exception as error:  # any rejection would surface here
                accepted = False
                raise RuntimeError(f"unexpected rejection: {error}") from error

            finite = bool(torch.isfinite(tam_recovered).all())
            records.append({
                "schema_version": 1,
                "sample_id": f"{split_row['sample_id']}|{SCHEME}|legacy_png|deployment",
                "image_id": split_row["sample_id"],
                "scheme": SCHEME,
                "storage": "legacy_png",
                "path": "deployment: anonymous image -> store -> DDIM inversion -> inverse key -> generation",
                "tamper_case": "stored_pixel_plus_128",
                "accepted": int(accepted),
                "output_finite": int(finite),
                "recovered_psnr_vs_clean_deploy_db": _psnr(tam_recovered, clean_recovered),
                "recovered_vs_clean_max_abs": float(torch.max(torch.abs(tam_recovered - clean_recovered)).cpu()),
            })
            print(f"{records[-1]['sample_id']}: accepted={accepted} finite={finite} "
                  f"psnr_vs_clean={records[-1]['recovered_psnr_vs_clean_deploy_db']:.2f}dB")

        checks = {
            "all_tamper_accepted_no_rejection": all(r["accepted"] == 1 for r in records),
            "all_outputs_finite": all(r["output_finite"] == 1 for r in records),
            "tamper_changed_output": all(r["recovered_vs_clean_max_abs"] > 0 for r in records),
        }
        if not all(checks.values()):
            raise RuntimeError(f"deployment tamper controls failed: {checks}")
        for record in records:
            record["checks_all_true"] = int(all(checks.values()))

        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        with args.table_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)

        numeric = [k for k, v in records[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        manifest = {
            "dataset_split_hash": rows_split[0]["split_sha256"],
            "checkpoint_hash": sha256_file(args.checkpoint),
            "seed_list": [0] * len(records), "steps": args.noise_level,
            "scheme": SCHEME, "rounds": 0, "nonce_mode": "none", "container_version": "none",
            "pipeline": "legacy_anonymization deployment decrypt path",
            "storage_protocol": "legacy_png/v1_minmax_truncate_discard_range",
            "latent_cache_run_id": __import__("json").loads((args.latent_cache / "manifest.json").read_text())["run_id"],
            "tamper": "flip stored uint8 pixel (128,128) by +128 mod 256; no MAC to reject",
            "checks": checks,
        }
        finalize_run(context, Path(__file__), records, numeric, manifest,
                     stdout_capture.getvalue(), stderr_capture.getvalue())


if __name__ == "__main__":
    main()
