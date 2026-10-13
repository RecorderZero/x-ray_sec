#!/usr/bin/env python3
"""One-image P/S0/S1/S2a/S2 visual and reconstruction comparison.

P/S0/S1 deliberately use the predecessor's lossy legacy PNG hand-off
(per-image min-max, uint8 truncation, range discarded).  S2a/S2 use the new
range-preserving PNG.  S2a and S2 share identical transform material; S2 adds
authentication and therefore must have identical pixels when untampered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.e1_ddim_runner import Tee, create_runtime, load_split, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics, legacy_png_handoff
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING,
    LEGACY_GUIDANCE_SCALE,
    LEGACY_PIPELINE,
    apply_legacy_key,
    legacy_generate,
    legacy_invert,
)
from scripts.e2_roundtrip_runner import _timed, load_latent_cache
from scripts.m1_storage import RANGE_PRESERVING_PNG, range_preserving_png_handoff
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import finalize_run, open_run, sha256_file
from src.crypto.container import VERSION
from src.crypto.nonce import NonceRegistry
from src.crypto.s2 import open_anonymous_image, seal_anonymous_image
from src.crypto.sot import inverse_transform, transform


SEED = 1911
METHODS = ("P", "S0", "S1", "S2a", "S2")
LEGACY_STORAGE = "legacy_png/v1_minmax_truncate_discard_range"
FIXTURE_KEY = hashlib.sha256(b"cxr-s2/D6/public-reproducibility-fixture/v1").digest()
SENIOR_REPORTED_COSINE = 0.998978
SENIOR_REPORTED_PSNR_DB = 30.23


def nonce_for(sample_id: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/comparison/{sample_id}/R1/P0".encode()).digest()[:16]


def _to_u8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else value
    return np.rint(np.clip(np.asarray(array).reshape(256, 256), 0, 1) * 255).astype(np.uint8)


def _uint8_error(reference: torch.Tensor, other: torch.Tensor) -> tuple[int, float]:
    x, y = _to_u8(reference), _to_u8(other)
    difference = np.abs(x.astype(np.int16) - y.astype(np.int16))
    return int(difference.max()), float(np.mean(x == y))


def _legacy_raw_cosine(reference: torch.Tensor, other: torch.Tensor) -> float:
    """The predecessor's no-mean-subtraction flattened image cosine formula."""
    x = reference.detach().float().cpu().numpy().reshape(-1)
    y = other.detach().float().cpu().numpy().reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def _record(
    *, method: str, storage: str, x0: torch.Tensor, p_output: torch.Tensor,
    anonymous: torch.Tensor, reconstructed: torch.Tensor, timings: dict[str, float],
) -> dict[str, Any]:
    metrics = image_pair_metrics("x0_vs_reconstruction", x0, reconstructed)
    uint8_max, uint8_equal = _uint8_error(x0, reconstructed)
    return {
        "schema_version": 1,
        "sample_id": method,
        "method": method,
        "storage_protocol": storage,
        "range_preserved": int(storage == RANGE_PRESERVING_PNG),
        "psnr_db": metrics["x0_vs_reconstruction_psnr_db"],
        "ssim": metrics["x0_vs_reconstruction_ssim"],
        "l_inf_float": metrics["x0_vs_reconstruction_linf"],
        "max_pixel_error_float": metrics["x0_vs_reconstruction_linf"],
        "max_pixel_error_uint8_levels": uint8_max,
        "mse": metrics["x0_vs_reconstruction_mse"],
        "rmse": metrics["x0_vs_reconstruction_rmse"],
        "mae": metrics["x0_vs_reconstruction_mae"],
        "x0_vs_reconstruction_raw_pixel_cosine": _legacy_raw_cosine(x0, reconstructed),
        "p_output_vs_reconstruction_raw_pixel_cosine": _legacy_raw_cosine(p_output, reconstructed),
        "uint8_pixel_equality_rate": uint8_equal,
        "float32_bit_exact_rate": metrics["x0_vs_reconstruction_float32_bit_exact_rate"],
        "runtime_seconds_anonymous_generation": timings["anonymous"],
        "runtime_seconds_inversion": timings["inversion"],
        "runtime_seconds_reconstruction_generation": timings["reconstruction"],
        "anonymous_finite": int(bool(torch.isfinite(anonymous).all())),
        "reconstruction_finite": int(bool(torch.isfinite(reconstructed).all())),
    }


@torch.inference_mode()
def run_comparison(args, row, z, diffusion, model_fn, device, registry):
    x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    records: list[dict[str, Any]] = []

    # P is generated first and serves as the base-flow reference used by this project.
    p_latent = apply_legacy_key(z, args.source_root, "P", "anonymize")
    p_output, _ = _timed(device, legacy_generate, diffusion, model_fn, p_latent, args.noise_level)

    for method in ("P", "S0", "S1"):
        z_anonymous = apply_legacy_key(z, args.source_root, method, "anonymize")
        anonymous, t_anonymous = _timed(
            device, legacy_generate, diffusion, model_fn, z_anonymous, args.noise_level,
        )
        stored = legacy_png_handoff(anonymous)
        z_hat_anonymous, t_inversion = _timed(
            device, legacy_invert, diffusion, model_fn, stored, args.noise_level,
        )
        z_hat = apply_legacy_key(z_hat_anonymous, args.source_root, method, "deanonymize")
        reconstructed, t_reconstruction = _timed(
            device, legacy_generate, diffusion, model_fn, z_hat, args.noise_level,
        )
        records.append(_record(
            method=method, storage=LEGACY_STORAGE, x0=x0, p_output=p_output,
            anonymous=stored, reconstructed=reconstructed,
            timings={"anonymous": t_anonymous, "inversion": t_inversion, "reconstruction": t_reconstruction},
        ))
        outputs[method] = (stored, reconstructed)

    transform_nonce = nonce_for(row["sample_id"])
    z_s2 = transform(
        z, FIXTURE_KEY, transform_nonce, rounds=1, use_permutation=False,
    )
    anonymous_s2, t_anonymous = _timed(
        device, legacy_generate, diffusion, model_fn, z_s2, args.noise_level,
    )

    stored_s2a, _ = range_preserving_png_handoff(anonymous_s2)
    z_hat_s2a, t_inversion_s2a = _timed(
        device, legacy_invert, diffusion, model_fn, stored_s2a, args.noise_level,
    )
    decrypted_s2a = inverse_transform(
        z_hat_s2a, FIXTURE_KEY, transform_nonce, rounds=1, use_permutation=False,
    )
    reconstructed_s2a, t_reconstruction_s2a = _timed(
        device, legacy_generate, diffusion, model_fn, decrypted_s2a, args.noise_level,
    )
    records.append(_record(
        method="S2a", storage=RANGE_PRESERVING_PNG, x0=x0, p_output=p_output,
        anonymous=stored_s2a, reconstructed=reconstructed_s2a,
        timings={"anonymous": t_anonymous, "inversion": t_inversion_s2a, "reconstruction": t_reconstruction_s2a},
    ))
    outputs["S2a"] = (stored_s2a, reconstructed_s2a)

    aad = {
        "study_instance_uid": f"research:{row['patient_id']}",
        "sop_instance_uid": f"research:{row['sample_id']}",
        "model_sha256": sha256_file(args.checkpoint),
        "sampler_version": LEGACY_PIPELINE,
    }
    sealed = seal_anonymous_image(
        anonymous_s2, FIXTURE_KEY, transform_nonce, rounds=1, use_permutation=False,
        latent_shape=tuple(z.shape), latent_dtype=str(z.detach().cpu().numpy().dtype),
        aad=aad, registry=registry,
    )
    inversion_seconds = [0.0]
    opened_images: list[torch.Tensor] = []

    def inverter(image):
        opened_images.append(image.clone())
        result, inversion_seconds[0] = _timed(
            device, legacy_invert, diffusion, model_fn, image, args.noise_level,
        )
        return result

    decrypted_s2 = open_anonymous_image(sealed, FIXTURE_KEY, inverter, device=device)
    reconstructed_s2, t_reconstruction_s2 = _timed(
        device, legacy_generate, diffusion, model_fn, decrypted_s2, args.noise_level,
    )
    if len(opened_images) != 1:
        raise RuntimeError("authenticated S2 opener must invoke inversion exactly once")
    displayed_s2 = opened_images[0]
    records.append(_record(
        method="S2", storage=RANGE_PRESERVING_PNG, x0=x0, p_output=p_output,
        anonymous=displayed_s2, reconstructed=reconstructed_s2,
        timings={"anonymous": t_anonymous, "inversion": inversion_seconds[0], "reconstruction": t_reconstruction_s2},
    ))
    outputs["S2"] = (displayed_s2, reconstructed_s2)

    if not torch.equal(stored_s2a, displayed_s2) or not torch.equal(reconstructed_s2a, reconstructed_s2):
        raise RuntimeError("untampered S2a/S2 pixels must be bit-identical")
    return x0, p_output, records, outputs, hashlib.sha256(sealed).hexdigest()


def save_grid(x0: torch.Tensor, outputs, path: Path, sample_id: str) -> None:
    columns = ("Original x0", "Stored anonymous image", "Reconstructed image")
    cell, label = 256, 32
    canvas = Image.new("RGB", (cell * 3, (cell + label) * len(METHODS) + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), f"{sample_id} | fixed display range [0,1] | rows: P, S0, S1, S2a, S2", fill="black")
    for column, title in enumerate(columns):
        draw.text((column * cell + 4, 20), title, fill="black")
    original = _to_u8(x0)
    for row_index, method in enumerate(METHODS):
        y0 = 34 + row_index * (cell + label)
        anonymous, reconstructed = outputs[method]
        draw.text((4, y0 + 5), f"{method} | {'range-preserving' if method in {'S2a', 'S2'} else 'legacy range-discarding'}", fill="black")
        for column, image in enumerate((original, _to_u8(anonymous), _to_u8(reconstructed))):
            canvas.paste(Image.fromarray(image, mode="L").convert("RGB"), (column * cell, y0 + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_report(path: Path, args, row, records) -> None:
    headers = (
        "method", "storage_protocol", "psnr_db", "ssim", "l_inf_float",
        "max_pixel_error_uint8_levels", "x0_vs_reconstruction_raw_pixel_cosine",
        "p_output_vs_reconstruction_raw_pixel_cosine",
    )
    lines = [
        "# P／S0／S1／S2a／S2 單圖比較",
        "",
        f"固定樣本：`{row['sample_id']}`（label=`{row['label_name']}`，未依結果挑圖）。",
        "",
        "依本次比較規則，P 也使用 legacy range-discarding PNG，作為與學長流程一致的無加密基準；因此此圖是完整 pipeline 比較，不是只隔離 transform 的公平消融。",
        "",
        f"![五種方法比較](../{args.grid_output.as_posix()})",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for record in records:
        values = []
        for header in headers:
            value = record[header]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "## Cosine 對照的正確解讀",
        "",
        f"學長論文 Signed Permutation 報告 cosine=`{SENIOR_REPORTED_COSINE}`、PSNR=`{SENIOR_REPORTED_PSNR_DB} dB`。但是原評估程式比較的是 Base `samples` 與加解密流程 `samples` 的 raw-pixel cosine，不是 x0 對重構圖，也不是 latent cosine。上表因此把 `x0_vs_reconstruction_raw_pixel_cosine` 與較接近前作比較概念的 `p_output_vs_reconstruction_raw_pixel_cosine` 分欄，不能直接把兩者與 0.998978 當成相同 estimand。",
        "",
        "PSNR、SSIM、L∞、MSE/MAE 都直接使用未 clip 的 float 輸出；`max_pixel_error_uint8_levels` 才先 clip 到 [0,1] 並以 8-bit 灰階計算。L∞ 與 float 最大 pixel error 在數學上是同一個量，因此 CSV 同時保留兩個名稱但數值相同。",
        "",
        "S2a 與 S2 在未竄改輸入下逐位元相同；S2 的額外價值是 HMAC、metadata authentication 與 verify-before-inversion，不是提升影像品質。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="dev_v1.1_010")
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--table-output", type=Path, default=Path("paper_assets/tables/table_method_comparison.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/method_comparison.json"))
    parser.add_argument("--grid-output", type=Path, default=Path("image/method_comparison.png"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/method_comparison.md"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="D6.COMPARE")
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.table_output, args.json_output, args.grid_output, args.report_output], args.force)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = load_split(args.split)
        matches = [index for index, row in enumerate(rows) if row["sample_id"] == args.sample_id]
        if len(matches) != 1:
            raise ValueError("sample ID must occur exactly once in the split")
        latents = load_latent_cache(args.latent_cache, rows)
        index, row = matches[0], rows[matches[0]]
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(methods=list(METHODS), s2_rounds=1, s2_permutation=False)
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        z = torch.from_numpy(latents[index]).unsqueeze(0).to(device)
        with NonceRegistry(context.run_dir / "nonce_registry.sqlite3") as registry:
            x0, p_output, records, outputs, container_sha256 = run_comparison(
                args, row, z, diffusion, model_fn, device, registry,
            )
        write_csv(args.table_output, records)
        save_grid(x0, outputs, args.grid_output, row["sample_id"])
        write_report(args.report_output, args, row, records)
        result = {
            "schema_version": 1,
            "task_id": args.task_id,
            "sample_id": row["sample_id"],
            "label": row["label_name"],
            "methods": list(METHODS),
            "storage_assignment": {"P": LEGACY_STORAGE, "S0": LEGACY_STORAGE, "S1": LEGACY_STORAGE, "S2a": RANGE_PRESERVING_PNG, "S2": RANGE_PRESERVING_PNG},
            "s2_parameters": {"rounds": 1, "permutation": False},
            "s2_container_sha256": container_sha256,
            "s2a_s2_anonymous_bit_exact": True,
            "s2a_s2_reconstruction_bit_exact": True,
            "senior_reference": {
                "reported_signed_permutation_cosine": SENIOR_REPORTED_COSINE,
                "reported_psnr_db": SENIOR_REPORTED_PSNR_DB,
                "comparison_pair": "Base samples vs encrypted/decrypted samples; not x0 vs xrec",
            },
            "table": str(args.table_output), "grid": str(args.grid_output), "report": str(args.report_output),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        cache_manifest = json.loads((args.latent_cache / "manifest.json").read_text(encoding="utf-8"))
        numeric = [
            key for key, value in records[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        finalize_run(context, Path(__file__), records, numeric, {
            "dataset_split_hash": row["split_sha256"],
            "checkpoint_hash": sha256_file(args.checkpoint),
            "source_sampler_sha256": sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py"),
            "seed_list": [args.seed] * len(records), "steps": args.noise_level,
            "scheme": "P/S0/S1/S2a/S2", "rounds": "legacy/legacy/legacy/1/1",
            "nonce_mode": "unique deterministic public comparison fixture",
            "container_version": VERSION, "pipeline": LEGACY_PIPELINE,
            "guidance_scale": LEGACY_GUIDANCE_SCALE,
            "generation_conditioning": GENERATION_CONDITIONING,
            "inversion_conditioning": "null=True (unconditional)",
            "latent_cache_run_id": cache_manifest["run_id"],
            "latent_cache_sha256": sha256_file(args.latent_cache / "latents.npy"),
            "raw_key_material_recorded": False,
        }, stdout_capture.getvalue(), stderr_capture.getvalue())
        print(json.dumps({record["method"]: {"psnr": record["psnr_db"], "ssim": record["ssim"]} for record in records}, indent=2))


if __name__ == "__main__":
    main()
