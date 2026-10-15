#!/usr/bin/env python3
"""Derive the P/S0/S1 legacy-PNG deployment cell from a validated Q8 parent.

The parent already contains the expensive anonymous generator outputs.  This
runner reads those hash-verified floats and performs only the predecessor PNG
handoff, target-model inversion, inverse legacy transform and reconstruction.
It deliberately never opens or recomputes S2, whose per-run random key is not
persisted.  The resulting artifact is joined by :mod:`scripts.q8_analyze`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts import e2_legacy_wrapper as legacy
from scripts.artifact_schema import validate_run
from scripts.q8_analyze import Run
from scripts.q8_metrics import METHODS, col
from scripts.q8_pilot_runner import (
    FAILURE_EXIT_CODE, IDENTITY_FIELDS, Meter, SyntheticBackend, decode_legacy, encode_legacy,
    image_columns, latent_columns, legacy_password, sha256_array, synthetic_samples,
)
from scripts.run_artifacts import finalize_run, open_run, sha256_file


BASELINES = ("P", "S0", "S1")
SUPPLEMENT_SCHEMA = 1
ARRAY_SPECS = {
    "anonymous_public_q": ("uint8", BASELINES),
    "anonymous_public_available": ("float32", BASELINES),
    "recon_m1": ("float32", BASELINES),
    "recon_wrong_key": ("float32", ("S0", "S1")),
}


def _parent_hashes(parent: Path) -> dict[str, str]:
    return {name: sha256_file(parent / name) for name in ("manifest.json", "config.json", "per_sample.csv", "array_index.json")}


def _real_latents(parent: Run, config: dict[str, Any]) -> np.ndarray:
    cache = Path(config["latent_cache"])
    validation = validate_run(cache)
    if not validation["valid"]:
        raise SystemExit(f"parent latent cache invalid: {validation['errors']}")
    expected = {
        "manifest.json": parent.manifest["latent_cache_manifest_sha256"],
        "config.json": parent.manifest["latent_cache_config_sha256"],
        "per_sample.csv": parent.manifest["latent_cache_index_sha256"],
        "latents.npy": parent.manifest["latent_cache_sha256"],
    }
    for name, digest in expected.items():
        if sha256_file(cache / name) != digest:
            raise SystemExit(f"latent cache {name} no longer matches the parent provenance")
    cache_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if cache_manifest["sample_ids"][:parent.n] != parent.sample_ids:
        raise SystemExit("latent cache sample order differs from the Q8 parent")
    values = np.load(cache / "latents.npy", mmap_mode="r")[:parent.n]
    for i, sample_id in enumerate(parent.sample_ids):
        if sha256_array(values[i]) != parent.row_hash("latent_sha256", i):
            raise SystemExit(f"latent cache hash differs for {sample_id}")
    return values


def build_inputs(parent: Run):
    config = json.loads((parent.run_dir / "config.json").read_text(encoding="utf-8"))
    frozen_modules = parent.manifest.get("imported_project_modules", {})
    relevant = {path: digest for path, digest in frozen_modules.items()
                if path == "scripts/e2_legacy_wrapper.py" or path.startswith("past/SourceCode/")}
    if not relevant:
        raise SystemExit("parent lacks hashes for the legacy/model source modules")
    for relative, digest in relevant.items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise SystemExit(f"legacy/model source changed since the parent run: {relative}")
    if parent.manifest["synthetic_backend"]:
        samples = synthetic_samples(parent.n)
        for i, sample in enumerate(samples):
            if sample.sample_id != parent.sample_ids[i] or sha256_array(sample.z.numpy()[0]) != parent.row_hash("latent_sha256", i):
                raise SystemExit("synthetic source cannot reproduce parent latent evidence")
        return np.stack([s.z.numpy()[0] for s in samples]), SyntheticBackend(), Path(config.get("source_root", "past/SourceCode"))

    checkpoint = Path(config["checkpoint"])
    source_root = Path(config["source_root"])
    sampler = source_root / "guided_diffusion" / "gaussian_diffusion.py"
    if sha256_file(checkpoint) != parent.manifest["checkpoint_hash"]:
        raise SystemExit("checkpoint no longer matches the parent hash")
    if sha256_file(sampler) != parent.manifest["source_sampler_sha256"]:
        raise SystemExit("source sampler no longer matches the parent hash")
    latents = _real_latents(parent, config)
    from scripts.e1_ddim_runner import create_runtime
    from scripts.q8_pilot_runner import DiffusionBackend
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for a real supplement")
    _, diffusion, model_fn = create_runtime(source_root, checkpoint, device)
    return latents, DiffusionBackend(diffusion, model_fn, int(parent.manifest["steps"]), device), source_root


def _allocate(run_dir: Path, n: int):
    directory = run_dir / "arrays"
    directory.mkdir()
    maps = {}
    for kind, (dtype, methods) in ARRAY_SPECS.items():
        maps[kind] = np.lib.format.open_memmap(directory / f"{kind}.npy", mode="w+", dtype=dtype,
                                               shape=(n, len(methods), 256, 256))
    return maps


def _close_arrays(run_dir: Path, maps) -> dict[str, Any]:
    evidence = {}
    for kind, mmap in maps.items():
        mmap.flush(); del mmap
        path = run_dir / "arrays" / f"{kind}.npy"
        array = np.load(path, mmap_mode="r")
        evidence[kind] = {"file": str(path.relative_to(run_dir)), "dtype": str(array.dtype),
                          "shape": list(array.shape), "methods": list(ARRAY_SPECS[kind][1]),
                          "sha256": sha256_file(path),
                          "slice_sha256": [[sha256_array(array[i, j]) for j in range(array.shape[1])]
                                           for i in range(array.shape[0])]}
    return evidence


def execute(args, parent: Run, latents, backend, source_root: Path, stdout_capture, stderr_capture) -> Path:
    if tuple(parent.manifest["methods"]) != METHODS or tuple(parent.manifest["storages"]) != ("range_preserving",):
        raise SystemExit("supplement requires a P/S0/S1/S2 parent whose sole cell is range_preserving")
    parent_hashes = _parent_hashes(parent.run_dir)
    config = {"parent_run": str(parent.run_dir), "parent_run_id": parent.manifest["run_id"],
              "parent_hashes": parent_hashes, "deployment_storage": {"P": "legacy", "S0": "legacy", "S1": "legacy"},
              "warmup": args.warmup, "n_samples": parent.n}
    context = open_run(args.task_id, config, args.artifacts_root)
    arrays = _allocate(context.run_dir, parent.n)
    rows = []
    device = backend.device
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                first = torch.from_numpy(np.asarray(parent.arrays["anonymous_float"][0, 0])).to(device)[None, None]
                backend.generate(backend.invert(first))
            for i, sample_id in enumerate(parent.sample_ids):
                row: dict[str, Any] = {"schema_version": SUPPLEMENT_SCHEMA, "sample_id": sample_id,
                                      "patient_id": parent.patients[i], "label": parent.labels[i]}
                z = torch.from_numpy(np.asarray(latents[i]).copy()).to(device)[None]
                x0 = torch.from_numpy(np.asarray(parent.arrays["x0"][i]).copy()).to(device)[None, None]
                p_output = torch.from_numpy(np.asarray(parent.arrays["anonymous_float"][i, 0]).copy()).to(device)[None, None]
                meter = Meter(device, row)
                for method_index, method in enumerate(BASELINES):
                    anonymous = torch.from_numpy(np.asarray(parent.arrays["anonymous_float"][i, method_index]).copy()).to(device)[None, None]
                    payload, q = meter(method, "storage_encode", encode_legacy, anonymous)
                    stored = meter(method, "storage_decode", decode_legacy, payload, device)
                    z_hat_ano = meter(method, "diffusion_invert", backend.invert, stored)
                    z_hat = meter(method, "transform_inverse", legacy.apply_legacy_key,
                                  z_hat_ano, source_root, method, "deanonymize")
                    recon = meter(method, "diffusion_generate_recon", backend.generate, z_hat)
                    row.update(image_columns(method, "anonpub", x0, stored))
                    row.update(image_columns(method, "m1", x0, recon))
                    row.update(image_columns(method, "m1_vs_p", p_output, recon))
                    row.update(latent_columns(method, "m1lat", z, z_hat))
                    row[col(method, "store", "m1_image_payload_bytes")] = len(payload)
                    parent_forward = parent.get(method, "runtime", "transform_forward_seconds")[i] + parent.get(method, "runtime", "diffusion_generate_anonymous_seconds")[i]
                    row[col(method, "runtime", "cached_latent_to_m1_seconds")] = parent_forward + row[col(method, "runtime", "storage_encode_seconds")]
                    row[col(method, "runtime", "m1_to_reconstruction_seconds")] = sum(row[col(method, "runtime", f"{stage}_seconds")] for stage in
                        ("storage_decode", "diffusion_invert", "transform_inverse", "diffusion_generate_recon"))
                    arrays["anonymous_public_q"][i, method_index] = q
                    arrays["anonymous_public_available"][i, method_index] = stored[0, 0].cpu().numpy()
                    arrays["recon_m1"][i, method_index] = recon[0, 0].cpu().numpy()
                    if method != "P":
                        with legacy_password(f"Q8-wrong-key/{sample_id}"):
                            z_wrong = legacy.apply_legacy_key(z_hat_ano, source_root, method, "deanonymize")
                        wrong = meter(method, "control_generate_wrong_key", backend.generate, z_wrong)
                        row.update(image_columns(method, "wrongkey", x0, wrong))
                        row.update(latent_columns(method, "wrongkeylat", z, z_wrong))
                        arrays["recon_wrong_key"][i, ("S0", "S1").index(method)] = wrong[0, 0].cpu().numpy()
                rows.append(row)
                print(f"{sample_id}: completed P/S0/S1 legacy deployment supplement", flush=True)
    except Exception as error:
        failure = {"sample_id": parent.sample_ids[len(rows)] if len(rows) < parent.n else "unknown", "completed": len(rows),
                   "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        (context.run_dir / "failures.json").write_text(json.dumps([failure], indent=2) + "\n", encoding="utf-8")
        raise SystemExit(FAILURE_EXIT_CODE) from error
    evidence = _close_arrays(context.run_dir, arrays)
    index_path = context.run_dir / "array_index.json"
    index_path.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    numeric = [key for key, value in rows[0].items() if key not in IDENTITY_FIELDS and isinstance(value, (int, float))]
    manifest = {"dataset_split_hash": parent.manifest["dataset_split_hash"], "checkpoint_hash": parent.manifest["checkpoint_hash"],
                "seed_list": parent.manifest["seed_list"], "steps": parent.manifest["steps"],
                "scheme": "Q8 legacy deployment supplement: P/S0/S1", "rounds": 0,
                "nonce_mode": "not applicable: predecessor deterministic legacy schemes", "container_version": "none",
                "q8_supplement_schema": SUPPLEMENT_SCHEMA, "parent_run_id": parent.manifest["run_id"],
                "parent_hashes": parent_hashes, "source_sampler_sha256": parent.manifest["source_sampler_sha256"],
                "evidence_role": parent.role, "synthetic_backend": parent.manifest["synthetic_backend"],
                "deployment_storage_by_method": {"P": "legacy_png_handoff", "S0": "legacy_png_handoff", "S1": "legacy_png_handoff"},
                "arrays": {kind: {k: v for k, v in entry.items() if k != "slice_sha256"} for kind, entry in evidence.items()},
                "array_index_file": index_path.name, "array_index_sha256": sha256_file(index_path),
                "failure_policy": "fail closed; parent/source/cache hashes verified before model construction", "failures": 0}
    return finalize_run(context, Path(__file__), rows, numeric, manifest, stdout_capture.getvalue(), stderr_capture.getvalue())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--task-id", default="Q8_LEGACY_SUPPLEMENT")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    return args


def main(argv=None):
    from scripts.e1_ddim_runner import Tee
    args = parse_args(argv)
    parent = Run(args.parent_run, allow_synthetic=True)
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(__import__('sys').stdout, stdout_capture)), redirect_stderr(Tee(__import__('sys').stderr, stderr_capture)):
        latents, backend, source_root = build_inputs(parent)
        execute(args, parent, latents, backend, source_root, stdout_capture, stderr_capture)


if __name__ == "__main__":
    main()
