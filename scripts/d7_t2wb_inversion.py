#!/usr/bin/env python3
"""D7.1 target-model inversion of S2 anonymous images.

Each source latent is transformed once with the frozen D6 candidate
R=1/permutation-off and a unique per-image nonce, generated once, then passed
through both legacy (range discarded) and range-preserving PNG handoffs before
target-model inversion.  S2a and S2 have identical anonymous pixels; HMAC is
evaluated separately by D7.8 and does not alter confidentiality metrics.
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

from scripts.e1_ddim_runner import Tee, create_runtime, load_split
from scripts.e2_anonymization_runner import latent_pair_metrics, legacy_png_handoff
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING, LEGACY_GUIDANCE_SCALE, LEGACY_PIPELINE,
    legacy_generate, legacy_invert,
)
from scripts.e2_roundtrip_runner import _timed, load_latent_cache
from scripts.m1_storage import QUANTIZATION, RANGE_PRESERVING_PNG, range_preserving_png_handoff
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import finalize_run, open_run, sha256_file
from src.crypto.sot import transform


SEED = 1911
ROUNDS = 1
USE_PERMUTATION = False
STORAGES = ("legacy_png", "range_preserving_png")
LEGACY_STORAGE = "legacy_png/v1_minmax_truncate_discard_range"
LEGACY_QUANTIZATION = "truncate((x-min)/(max-min)*255); discard min/max"
BOOTSTRAP_RESAMPLES = 10_000
FIXTURE_KEY = hashlib.sha256(b"cxr-s2/D7/public-reproducibility-fixture/v1").digest()
SUMMARY_METRICS = (
    "attacker_latent_cosine", "attacker_latent_mse", "attacker_latent_rmse",
    "attacker_latent_max_abs", "attacker_latent_float32_bit_exact_rate",
    "attacker_latent_norm_ratio", "attacker_latent_lowfreq32_cosine",
    "attacker_latent_abs_pearson", "runtime_seconds_anonymous_generation",
    "runtime_seconds_inversion",
)


def nonce_for(sample_id: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/D7/formal/{sample_id}".encode()).digest()[:16]


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for storage_index, storage in enumerate(STORAGES):
        selected = [row for row in rows if row["storage"] == storage]
        rng = np.random.default_rng(SEED + storage_index)
        draws = rng.integers(0, len(selected), size=(BOOTSTRAP_RESAMPLES, len(selected)))
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            means = values[draws].mean(axis=1)
            output.append({
                "method": "S2a/S2", "storage": storage, "metric": metric, "n": len(values),
                "mean": float(values.mean()), "ci95_low": float(np.percentile(means, 2.5)),
                "ci95_high": float(np.percentile(means, 97.5)), "median": float(np.median(values)),
                "min": float(values.min()), "max": float(values.max()),
            })
    return output


@torch.inference_mode()
def run(args, rows, latents, diffusion, model_fn, device):
    records: list[dict[str, Any]] = []
    queries: list[np.ndarray] = []
    query_index: list[dict[str, Any]] = []
    for image_index, split_row in enumerate(rows):
        z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
        nonce = nonce_for(split_row["sample_id"])
        z_ano = transform(
            z, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        )
        anonymous, generation_seconds = _timed(
            device, legacy_generate, diffusion, model_fn, z_ano, args.noise_level,
        )
        if not bool(torch.isfinite(anonymous).all()):
            raise RuntimeError(f"non-finite anonymous image: {split_row['sample_id']}")
        for storage in STORAGES:
            if storage == "legacy_png":
                stored = legacy_png_handoff(anonymous)
                storage_record = {
                    "storage_protocol": LEGACY_STORAGE, "quantization": LEGACY_QUANTIZATION,
                    "metadata_available": 0, "storage_lo": 0.0, "storage_hi": 1.0,
                    "storage_lo_float32_bits": "", "storage_hi_float32_bits": "",
                    "storage_png_sha256": "", "storage_png_bytes": 0,
                }
            else:
                stored, encoded = range_preserving_png_handoff(anonymous)
                storage_record = {
                    "storage_protocol": encoded["storage_protocol"],
                    "quantization": encoded["quantization"], "metadata_available": 1,
                    "storage_lo": encoded["lo"], "storage_hi": encoded["hi"],
                    "storage_lo_float32_bits": encoded["lo_float32_bits"],
                    "storage_hi_float32_bits": encoded["hi_float32_bits"],
                    "storage_png_sha256": encoded["png_sha256"],
                    "storage_png_bytes": encoded["png_bytes"],
                }
            zhat_ano, inversion_seconds = _timed(
                device, legacy_invert, diffusion, model_fn, stored, args.noise_level,
            )
            if not bool(torch.isfinite(zhat_ano).all()):
                raise RuntimeError(f"non-finite inversion: {split_row['sample_id']} {storage}")
            query = zhat_ano.detach().float().cpu().numpy()[0]
            query_position = len(queries)
            queries.append(query)
            sample_id = f"{split_row['sample_id']}|S2|{storage}"
            query_index.append({
                "sample_id": sample_id, "image_id": split_row["sample_id"],
                "storage": storage, "query_index": query_position,
                "query_sha256": sha256_array(query), "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
            })
            record: dict[str, Any] = {
                "schema_version": 1, "sample_id": sample_id, "image_id": split_row["sample_id"],
                "patient_id": split_row["patient_id"], "label": split_row["label"],
                "method": "S2a/S2", "storage": storage, "seed": args.seed,
                "noise_level": args.noise_level, "rounds": ROUNDS,
                "permutation": int(USE_PERMUTATION), "query_index": query_position,
                "query_sha256": query_index[-1]["query_sha256"],
                "nonce_sha256": query_index[-1]["nonce_sha256"],
                "runtime_seconds_anonymous_generation": generation_seconds,
                "runtime_seconds_inversion": inversion_seconds, **storage_record,
            }
            record.update(latent_pair_metrics("attacker_latent", z_ano, zhat_ano))
            records.append(record)
            print(
                f"{sample_id}: cos={record['attacker_latent_cosine']:.6f} "
                f"norm_ratio={record['attacker_latent_norm_ratio']:.6f}", flush=True,
            )
    return records, np.stack(queries).astype(np.float32, copy=False), query_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0, help="first N images; smoke only")
    parser.add_argument("--table-output", type=Path, default=Path("paper_assets/tables/table_D7_T2WB_inversion.csv"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="D7.1_T2WB")
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.table_output], args.force)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = load_split(args.split)
        latents = load_latent_cache(args.latent_cache, rows)
        cache_manifest = json.loads((args.latent_cache / "manifest.json").read_text(encoding="utf-8"))
        checkpoint_hash = sha256_file(args.checkpoint)
        if checkpoint_hash != cache_manifest.get("checkpoint_hash"):
            raise ValueError("checkpoint hash differs from E2.3 cache")
        # ``--limit 0`` only means "use the whole selected split".  It does not
        # make the 20-image dev split a formal run.  Evidence roles must follow
        # the frozen split contract, not merely the absence of a smoke limit.
        formal = args.limit == 0 and len(rows) == 200 and args.split.name == "security_v1.csv"
        evidence_role = "formal" if formal else ("smoke" if args.limit else "dev_calibration")
        if args.limit:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            rows, latents = rows[:args.limit], latents[:args.limit]
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(storages=list(STORAGES), formal=formal, evidence_role=evidence_role, batch_size=1,
                      pipeline=LEGACY_PIPELINE, guidance_scale=LEGACY_GUIDANCE_SCALE)
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        per_sample, queries, query_index = run(args, rows, latents, diffusion, model_fn, device)
        write_csv(args.table_output, summarize(per_sample))
        query_path = context.run_dir / "query_latents.npy"
        np.save(query_path, queries)
        (context.run_dir / "query_index.json").write_text(json.dumps(query_index, indent=2) + "\n")
        reloaded = np.load(query_path, mmap_mode="r")
        for entry in query_index:
            if sha256_array(reloaded[entry["query_index"]]) != entry["query_sha256"]:
                raise RuntimeError(f"saved query hash mismatch: {entry['sample_id']}")
        numeric = [key for key, value in per_sample[0].items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)]
        manifest = {
            "dataset_split_hash": rows[0]["split_sha256"], "checkpoint_hash": checkpoint_hash,
            "source_sampler_sha256": sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py"),
            "seed_list": [args.seed] * len(per_sample), "steps": args.noise_level,
            "scheme": "S2a/S2", "rounds": ROUNDS,
            "nonce_mode": "unique deterministic public D7 reproducibility fixture",
            "container_version": "S2 confidentiality path; D7.8 authenticates separately",
            "pipeline": LEGACY_PIPELINE, "guidance_scale": LEGACY_GUIDANCE_SCALE,
            "generation_conditioning": GENERATION_CONDITIONING,
            "inversion_conditioning": "null=True (unconditional)",
            "storage_protocols": [LEGACY_STORAGE, RANGE_PRESERVING_PNG],
            "range_quantization": QUANTIZATION, "batch_size": 1, "formal": formal,
            "evidence_role": evidence_role,
            "latent_cache_run_id": cache_manifest["run_id"],
            "latent_cache_sha256": sha256_file(args.latent_cache / "latents.npy"),
            "query_cache_file": query_path.name, "query_cache_sha256": sha256_file(query_path),
            "query_cache_shape": list(queries.shape), "query_index_file": "query_index.json",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "raw_key_material_recorded": False,
            "public_fixture_key_sha256": hashlib.sha256(FIXTURE_KEY).hexdigest(),
        }
        finalize_run(context, Path(__file__), per_sample, numeric, manifest,
                     stdout_capture.getvalue(), stderr_capture.getvalue())


if __name__ == "__main__":
    main()
