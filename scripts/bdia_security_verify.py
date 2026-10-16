#!/usr/bin/env python3
"""Week-7 appendix: sampler-dependent T2-WB security re-verification for a BDIA cell.

To claim a BDIA cell keeps the frozen S2 (DDIM+asymmetric, AF-025/D7) security, only
the SAMPLER-dependent arm has to be re-measured; the latent/container-only arms
(T1, norm-leakage on the sealed container, geometry QᵀQ=I, cross-nonce, structured
probe, repeat, HMAC tamper) are independent of which sampler produced the anonymous
image and are inherited unchanged from the frozen battery.  The sampler-dependent arm
is T2-WB white-box re-inversion re-identification: an attacker re-inverts the public
anonymous image back to a latent and tries to match it to the right patient.

This runner is AF-025's ``af025_fixed_norm_t2wb`` with the generator/inverter swapped
for a 2x2 cell's sampler (``scripts.bdia_2x2_runner.build_backend``).  EVERYTHING else
is held identical to the frozen battery -- same public reproducibility FIXTURE_KEY,
same deterministic nonces, same E2.3 latents, same fixed-norm scaling, SOT transform
and HMAC/AEAD container -- so the ONLY difference from AF-025 is the sampler.  It emits
the same per-sample schema and ``query_latents.npy`` as AF-025, so the existing CPU
analyzers (``af025_fixed_norm_analyze`` / ``d7_t2wb_analyze``) score re-identification
without change.  Under fixed-norm every anonymous latent has ‖·‖=C, so the attacker's
re-inverted norm carries no identity and T2-WB is expected to stay at the random floor
for BDIA exactly as for DDIM; this run confirms that empirically.

Cells (``--cell``): 2 = DDIM+symmetric-null, 3 = BDIA+asymmetric, 4 = BDIA+symmetric-
null.  Cell 1 (DDIM+asymmetric) is AF-025 itself and is not re-run here.

Scope note (skill 8/9): this re-verifies security of the *scheme as deployed for the
cell*.  Promoting any BDIA cell to the method of record is a research-design change and
needs Cowork sign-off; this appendix run does not make that change.

CPU self-test (no model/GPU), writes under the scratchpad::

  scripts/run_cfg_ddim.sh python -m scripts.bdia_security_verify --synthetic-selftest \\
      --cell 3 --limit 4 --artifacts-root <scratchpad>/bdia_sec_selftest \\
      --table-output <scratchpad>/bdia_sec_selftest/table.csv
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.af025_fixed_norm_t2wb import (
    FIXTURE_KEY, ROUNDS, SEED, USE_PERMUTATION, nonce_for, sha256_array, summarize, write_csv,
)
from scripts.bdia_2x2_runner import CELLS, RUNNABLE_CELLS, build_backend
from scripts.e2_anonymization_runner import latent_pair_metrics
from scripts.e2_legacy_wrapper import LEGACY_GUIDANCE_SCALE, LEGACY_PIPELINE
from scripts.e2_roundtrip_runner import _timed
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.q8_pilot_runner import SyntheticBackend
from scripts.run_artifacts import finalize_run, open_run, sha256_file
from src.crypto.container import deserialize_container
from src.crypto.nonce import NonceRegistry
from src.crypto.s2_fixed_norm import (
    FIXED_NORM, normalize_to_fixed_norm, open_anonymous_image, public_anonymous_image,
    seal_anonymous_image,
)
from src.crypto.sot import transform


@torch.inference_mode()
def run(args, cell: int, rows, latents, backend, device, registry):
    """AF-025 T2-WB loop with the cell's sampler; identical key/nonce/latents/container."""
    spec = CELLS[cell]
    sampler_version = f"{LEGACY_PIPELINE}|cell{cell}:{spec['sampler']}/{spec['conditioning']}"
    records, queries, query_index = [], [], []
    model_hash = args.checkpoint_hash
    for image_index, split_row in enumerate(rows):
        z = torch.from_numpy(latents[image_index]).unsqueeze(0).to(device)
        normalized, original_norm = normalize_to_fixed_norm(z)
        nonce = nonce_for(split_row["sample_id"])
        protected = transform(normalized, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION)
        anonymous, generation_seconds = _timed(device, backend.generate, protected)
        aad = {
            "study_instance_uid": f"research:{split_row['patient_id']}",
            "sop_instance_uid": f"research:{split_row['sample_id']}",
            "model_sha256": model_hash, "sampler_version": sampler_version,
        }
        sealed = seal_anonymous_image(
            anonymous, original_norm, FIXTURE_KEY, nonce, rounds=ROUNDS,
            use_permutation=USE_PERMUTATION, latent_shape=tuple(z.shape),
            latent_dtype=str(z.detach().cpu().numpy().dtype), aad=aad, registry=registry,
        )
        public_image = public_anonymous_image(sealed, device=device)
        attacker_query, attacker_seconds = _timed(device, backend.invert, public_image)
        legitimate_seconds = [0.0]

        def inverter(image):
            value, legitimate_seconds[0] = _timed(device, backend.invert, image)
            return value

        legitimate = open_anonymous_image(sealed, FIXTURE_KEY, inverter, device=device)
        if not all(bool(torch.isfinite(v).all()) for v in (anonymous, attacker_query, legitimate)):
            raise RuntimeError(f"non-finite output: {split_row['sample_id']}")
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
            "cell": cell, "sampler": spec["sampler"], "conditioning": spec["conditioning"],
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
        print(f"cell {cell} {split_row['sample_id']}: attacker_norm={record['attacker_latent_norm']:.6f} "
              f"legit_cos={record['legitimate_latent_cosine']:.6f}", flush=True)
    return records, np.stack(queries).astype(np.float32, copy=False), query_index


# --------------------------------------------------------------------------- inputs
def build_real_inputs(args):
    from scripts.e1_ddim_runner import create_runtime, load_split
    from scripts.e2_roundtrip_runner import load_latent_cache

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (use --synthetic-selftest for the CPU plumbing check)")
    rows = load_split(args.split)
    latents = load_latent_cache(args.latent_cache, rows)
    cache_manifest = json.loads((args.latent_cache / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != cache_manifest.get("checkpoint_hash"):
        raise ValueError("checkpoint hash differs from E2.3 cache")
    args.checkpoint_hash = checkpoint_hash
    args.cache_manifest = cache_manifest
    if args.limit:
        rows, latents = rows[:args.limit], latents[:args.limit]
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    backend, _nfe = build_backend(args.cell, diffusion, model_fn, args.noise_level, device)
    return rows, latents, backend, device


def build_synthetic_inputs(args):
    n = args.limit or 4
    rng = np.random.default_rng(SEED)
    rows = [{"sample_id": f"synthetic_{i:04d}", "patient_id": f"patient{i:04d}", "label": str(i % 2),
             "split_sha256": hashlib.sha256(b"bdia-sec-synthetic").hexdigest()} for i in range(n)]
    latents = rng.standard_normal((n, 1, 256, 256)).astype(np.float32)
    args.checkpoint_hash = hashlib.sha256(b"bdia-sec-synthetic-ckpt").hexdigest()
    args.cache_manifest = {"run_id": "synthetic", "checkpoint_hash": args.checkpoint_hash}

    class _Synth(SyntheticBackend):
        def __init__(self, noise_level):
            self.noise_level = noise_level

    return rows, latents, _Synth(args.noise_level), torch.device("cpu")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cell", type=int, required=True, choices=RUNNABLE_CELLS)
    parser.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--latent-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--synthetic-selftest", action="store_true")
    parser.add_argument("--table-output", type=Path, default=None)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default=None)
    add_force_flag(parser)
    args = parser.parse_args(argv)
    if not args.synthetic_selftest and not (args.latent_cache and args.checkpoint):
        raise SystemExit("--latent-cache and --checkpoint are required for real runs")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    args.task_id = args.task_id or f"BDIA_SEC_VERIFY_CELL{args.cell}"
    args.table_output = args.table_output or Path(
        f"paper_assets/appendix_bdia/table_bdia_cell{args.cell}_T2WB_inversion.csv")
    return args


def main(argv=None) -> None:
    from scripts.e1_ddim_runner import Tee

    args = parse_args(argv)
    refuse_overwrite([args.table_output], args.force)
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        inputs = build_synthetic_inputs if args.synthetic_selftest else build_real_inputs
        rows, latents, backend, device = inputs(args)
        formal = (not args.synthetic_selftest and args.limit == 0 and len(rows) == 200
                  and args.split.name == "security_v1.csv")
        evidence_role = ("synthetic_selftest" if args.synthetic_selftest
                         else "formal" if formal else ("smoke" if args.limit else "dev_calibration"))
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                  if k not in ("cache_manifest",)}
        config.update(cell=args.cell, cell_spec=CELLS[args.cell], formal=formal, evidence_role=evidence_role,
                      fixed_norm=float(FIXED_NORM), public_lo_hi=False, pipeline=LEGACY_PIPELINE,
                      guidance_scale=LEGACY_GUIDANCE_SCALE)
        context = open_run(args.task_id, config, args.artifacts_root)
        with NonceRegistry(context.run_dir / "nonces.sqlite3") as registry:
            per_sample, queries, query_index = run(args, args.cell, rows, latents, backend, device, registry)
        write_csv(args.table_output, summarize(per_sample))
        query_path = context.run_dir / "query_latents.npy"; np.save(query_path, queries)
        (context.run_dir / "query_index.json").write_text(json.dumps(query_index, indent=2) + "\n")
        reloaded = np.load(query_path, mmap_mode="r")
        for entry in query_index:
            if sha256_array(reloaded[entry["query_index"]]) != entry["query_sha256"]:
                raise RuntimeError(f"saved query hash mismatch: {entry['sample_id']}")
        numeric = [k for k, v in per_sample[0].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        manifest = {
            "dataset_split_hash": rows[0]["split_sha256"], "checkpoint_hash": args.checkpoint_hash,
            "source_sampler_sha256": (hashlib.sha256(b"bdia-sec-synthetic").hexdigest()
                                      if args.synthetic_selftest
                                      else sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py")),
            "bdia_backend_sha256": sha256_file(Path("scripts/bdia_backend.py")),
            "seed_list": [args.seed] * len(per_sample), "steps": args.noise_level,
            "scheme": "S2-fixed-norm/v2", "cell": args.cell, "cell_spec": CELLS[args.cell],
            "rounds": ROUNDS, "permutation": USE_PERMUTATION,
            "container_version": "cxr-auth-container/v1 with S2-fixed-norm/v2 parameters",
            "nonce_mode": "unique deterministic public AF-025 reproducibility fixture (shared with AF-025)",
            "fixed_norm": float(FIXED_NORM), "private_metadata": "ChaCha20-Poly1305",
            "public_lo_hi": False, "formal": formal, "evidence_role": evidence_role,
            "pipeline": LEGACY_PIPELINE, "guidance_scale": LEGACY_GUIDANCE_SCALE,
            "generation_conditioning": CELLS[args.cell]["conditioning"],
            "inversion_conditioning": "null=True (unconditional)",
            "latent_cache_run_id": args.cache_manifest["run_id"],
            "query_cache_file": query_path.name, "query_cache_sha256": sha256_file(query_path),
            "query_cache_shape": list(queries.shape), "query_index_file": "query_index.json",
            "raw_key_material_recorded": False,
            "public_fixture_key_sha256": hashlib.sha256(FIXTURE_KEY).hexdigest(),
            "sampler_dependent_arm": "T2-WB white-box re-inversion re-identification",
            "inherited_arms": "T1, norm-leakage, geometry (QtQ=I), cross-nonce, structured-probe, "
                             "repeat, HMAC tamper are sampler-independent; inherited from AF-025/D7",
            "analyzer": "af025_fixed_norm_analyze / d7_t2wb_analyze (same query schema)",
        }
        finalize_run(context, Path(__file__), per_sample, numeric, manifest,
                     stdout_capture.getvalue(), stderr_capture.getvalue())


if __name__ == "__main__":
    main()
