#!/usr/bin/env python3
"""AF-025 T2-WB runner for fixed-norm S2 v2 with private M1 range."""

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
from scripts.e2_anonymization_runner import latent_pair_metrics
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING, LEGACY_GUIDANCE_SCALE, LEGACY_PIPELINE,
    legacy_generate, legacy_invert,
)
from scripts.e2_roundtrip_runner import _timed, load_latent_cache
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import finalize_run, open_run, sha256_file
from src.crypto.container import deserialize_container
from src.crypto.nonce import NonceRegistry
from src.crypto.s2_fixed_norm import (
    FIXED_NORM, normalize_to_fixed_norm, open_anonymous_image,
    public_anonymous_image, seal_anonymous_image,
)
from src.crypto.sot import transform


SEED = 1911
ROUNDS = 1
USE_PERMUTATION = False
BOOTSTRAP_RESAMPLES = 10_000
FIXTURE_KEY = hashlib.sha256(b"cxr-s2/D7/public-reproducibility-fixture/v1").digest()
SUMMARY_METRICS = (
    "attacker_latent_norm", "attacker_latent_norm_ratio_to_C",
    "legitimate_latent_cosine", "legitimate_latent_rmse", "legitimate_latent_max_abs",
    "runtime_seconds_anonymous_generation", "runtime_seconds_attacker_inversion",
    "runtime_seconds_legitimate_inversion",
)


def nonce_for(sample_id: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/AF025/fixed-norm/{sample_id}".encode()).digest()[:16]


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(rows), size=(BOOTSTRAP_RESAMPLES, len(rows)))
    output = []
    for metric in SUMMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        means = values[draws].mean(axis=1)
        output.append({
            "method": "S2-fixed-norm/v2", "storage": "private_range_png", "metric": metric,
            "n": len(values), "mean": float(values.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
            "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()),
        })
    return output


@torch.inference_mode()
def run(args, rows, latents, diffusion, model_fn, device, registry):
    records, queries, query_index = [], [], []
    model_hash = sha256_file(args.checkpoint)
    for image_index, split_row in enumerate(rows):
        z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
        normalized, original_norm = normalize_to_fixed_norm(z)
        nonce = nonce_for(split_row["sample_id"])
        protected = transform(
            normalized, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        )
        anonymous, generation_seconds = _timed(
            device, legacy_generate, diffusion, model_fn, protected, args.noise_level,
        )
        aad = {
            "study_instance_uid": f"research:{split_row['patient_id']}",
            "sop_instance_uid": f"research:{split_row['sample_id']}",
            "model_sha256": model_hash, "sampler_version": LEGACY_PIPELINE,
        }
        sealed = seal_anonymous_image(
            anonymous, original_norm, FIXTURE_KEY, nonce, rounds=ROUNDS,
            use_permutation=USE_PERMUTATION, latent_shape=tuple(z.shape),
            latent_dtype=str(z.detach().cpu().numpy().dtype), aad=aad, registry=registry,
        )
        public_image = public_anonymous_image(sealed, device=device)
        attacker_query, attacker_seconds = _timed(
            device, legacy_invert, diffusion, model_fn, public_image, args.noise_level,
        )
        legitimate_seconds = [0.0]

        def inverter(image):
            value, legitimate_seconds[0] = _timed(
                device, legacy_invert, diffusion, model_fn, image, args.noise_level,
            )
            return value

        legitimate = open_anonymous_image(sealed, FIXTURE_KEY, inverter, device=device)
        if not all(bool(torch.isfinite(value).all()) for value in (anonymous, attacker_query, legitimate)):
            raise RuntimeError(f"non-finite AF-025 output: {split_row['sample_id']}")
        query = attacker_query.detach().float().cpu().numpy()[0]
        query_position = len(queries); queries.append(query)
        query_index.append({
            "sample_id": split_row["sample_id"], "query_index": query_position,
            "query_sha256": sha256_array(query), "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
        })
        parsed = deserialize_container(sealed)
        record: dict[str, Any] = {
            "schema_version": 1, "sample_id": split_row["sample_id"],
            "patient_id": split_row["patient_id"], "label": split_row["label"],
            "method": "S2-fixed-norm/v2", "storage": "private_range_png",
            "seed": args.seed, "noise_level": args.noise_level, "rounds": ROUNDS,
            "permutation": int(USE_PERMUTATION), "fixed_norm": float(FIXED_NORM),
            "query_index": query_position, "query_sha256": query_index[-1]["query_sha256"],
            "nonce_sha256": query_index[-1]["nonce_sha256"],
            "public_lo_hi_available": 0, "container_bytes": len(sealed),
            "public_png_bytes": len(parsed.ciphertext),
            "runtime_seconds_anonymous_generation": generation_seconds,
            "runtime_seconds_attacker_inversion": attacker_seconds,
            "runtime_seconds_legitimate_inversion": legitimate_seconds[0],
            "attacker_latent_norm": float(torch.linalg.vector_norm(attacker_query.double())),
            "attacker_latent_norm_ratio_to_C": float(torch.linalg.vector_norm(attacker_query.double()) / float(FIXED_NORM)),
        }
        record.update(latent_pair_metrics("legitimate_latent", z, legitimate))
        records.append(record)
        print(
            f"{split_row['sample_id']}: attacker_norm={record['attacker_latent_norm']:.6f} "
            f"legit_cos={record['legitimate_latent_cosine']:.6f}", flush=True,
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--table-output", type=Path, default=Path("paper_assets/tables/table_AF025_fixed_norm_T2WB_inversion.csv"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="AF025_FIXED_NORM_T2WB")
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.table_output], args.force)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        rows = load_split(args.split); latents = load_latent_cache(args.latent_cache, rows)
        cache_manifest = json.loads((args.latent_cache / "manifest.json").read_text(encoding="utf-8"))
        checkpoint_hash = sha256_file(args.checkpoint)
        if checkpoint_hash != cache_manifest.get("checkpoint_hash"):
            raise ValueError("checkpoint hash differs from E2.3 cache")
        formal = args.limit == 0 and len(rows) == 200 and args.split.name == "security_v1.csv"
        evidence_role = "formal" if formal else ("smoke" if args.limit else "dev_calibration")
        if args.limit:
            if args.limit < 1: raise ValueError("--limit must be positive")
            rows, latents = rows[:args.limit], latents[:args.limit]
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(formal=formal, evidence_role=evidence_role, batch_size=1,
                      fixed_norm=float(FIXED_NORM), public_lo_hi=False,
                      pipeline=LEGACY_PIPELINE, guidance_scale=LEGACY_GUIDANCE_SCALE)
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        with NonceRegistry(context.run_dir / "nonces.sqlite3") as registry:
            per_sample, queries, query_index = run(
                args, rows, latents, diffusion, model_fn, device, registry,
            )
        write_csv(args.table_output, summarize(per_sample))
        query_path = context.run_dir / "query_latents.npy"; np.save(query_path, queries)
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
            "scheme": "S2-fixed-norm/v2", "rounds": ROUNDS, "permutation": USE_PERMUTATION,
            "container_version": "cxr-auth-container/v1 with S2-fixed-norm/v2 parameters",
            "nonce_mode": "unique deterministic public AF-025 reproducibility fixture",
            "fixed_norm": float(FIXED_NORM), "private_metadata": "ChaCha20-Poly1305",
            "public_lo_hi": False, "batch_size": 1, "formal": formal,
            "evidence_role": evidence_role, "pipeline": LEGACY_PIPELINE,
            "guidance_scale": LEGACY_GUIDANCE_SCALE,
            "generation_conditioning": GENERATION_CONDITIONING,
            "inversion_conditioning": "null=True (unconditional)",
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
