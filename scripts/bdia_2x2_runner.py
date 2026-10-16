#!/usr/bin/env python3
"""Week-7 appendix: S2 BDIA/DDIM x asymmetric/symmetric-null 2x2 quality ablation.

This is a QUALITY / reversibility ablation of the *scheme of record* (S2 fixed-norm
v2).  It swaps only the sampler and the generation conditioning; the fixed-norm
scaling, the SOT transform, the HMAC/AEAD container and all storage are byte-for-byte
the frozen S2 settings (ROUNDS=1, USE_PERMUTATION=False, private range_preserving
container).  It does NOT re-establish any anonymity/security claim -- the symmetric
cells remove the pseudo-healthy conditioning and so change anonymization semantics;
see ``scripts/bdia_security_verify.py`` and the appendix design note for the honest
security boundary.

Two axes, four cells (cell 1 is the main table and is REUSED, never re-run here):

                asymmetric (invert null / gen y=0 cond)   symmetric null (both null)
    DDIM        (1) scheme of record = Q8_FORMAL S2        (2) run
    BDIA(g=1)   (3) run  <- pure-BDIA honest number        (4) run

Per (sample, cell) it records, in one wide row keyed by method token ``S2`` plus a
``cell`` identity field, exactly the quality columns of the main S2 row:

  anon            anonymous generator output vs x0
  m1 / m1lat      image hand-off path recon vs x0, and recovered latent vs z
  m1_vs_p         recon vs the P base flow (same cell's P-flow is not run; see note)
  anonpub         public view (q/255; lo/hi are encrypted) vs x0
  runtime / nfe   per-stage seconds, GPU peak, and model-eval count (DDIM==BDIA)

Inversion x0->z is null-conditioned in every cell, so it depends only on the
sampler: DDIM-null z is bit-identical to the E2.3 cache (deterministic), BDIA-null z
is this sampler's own inversion.  By default z is computed on the fly; ``--ddim-
latent-cache`` lets a DDIM cell reuse and provenance-pin the frozen E2.3 latents.

Failure policy (fail closed): any exception or non-finite value aborts the run,
writes ``failures.json`` and exits 4 before per_sample/summary, so a partial run can
never validate.  Retries use a NEW run id; seeds/samples/settings are never changed
to obtain better numbers (loop_engineering_spec 4.4).

CPU self-test (no model, no GPU), writes under the scratchpad, exercises the full
crypto + metric + artifact wiring with the pointwise SyntheticBackend::

  scripts/run_cfg_ddim.sh python -m scripts.bdia_2x2_runner --synthetic-selftest \\
      --cells 2 3 4 --limit 3 --artifacts-root <scratchpad>/bdia2x2_selftest

A real GPU run is gated (CLAUDE.md 3) and launched only on the user's go; it mirrors
the q8 managed-run invocation with ``--checkpoint`` and the real split.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from scripts.bdia_backend import NFE, BDIABackend
from scripts.e2_anonymization_runner import _sync
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING, LEGACY_GUIDANCE_SCALE, LEGACY_PIPELINE,
)
from scripts.q8_metrics import SCHEMA_VERSION, col, slice_hashes
from scripts.q8_pilot_runner import (
    FIXTURE_KEY, KEY_MODE_TEXT, ROUNDS, SEED, USE_PERMUTATION, VS_P_METRIC_NAMES,
    KeyPolicy, Meter, SyntheticBackend, image_columns, latent_columns, require_finite,
    resolve_key_mode, sha256_array, synthetic_samples, timed_call,
)
from scripts.run_artifacts import finalize_run, open_run, sha256_file
from src.crypto.container import VERSION
from src.crypto.nonce import NonceRegistry
from src.crypto.s2_fixed_norm import (
    FIXED_NORM, SCHEME as S2_SCHEME, normalize_to_fixed_norm, open_anonymous_image,
    public_anonymous_image, seal_anonymous_image,
)
from src.crypto.sot import transform

FAILURE_EXIT_CODE = 4
IMAGE_SHAPE_HW = (256, 256)
EVIDENCE_ROLES = ("smoke", "pilot", "formal", "synthetic_selftest")
FIXTURE_ROLES = ("synthetic_selftest", "smoke")

# Cell 1 (DDIM + asymmetric) is the main table (Q8_FORMAL S2); it is reused, never
# run here, so the runnable cells are 2, 3, 4.
CELLS: dict[int, dict[str, str]] = {
    1: {"sampler": "ddim", "conditioning": "asymmetric", "note": "scheme of record = Q8_FORMAL S2 (reused, not run here)"},
    2: {"sampler": "ddim", "conditioning": "symmetric_null", "note": "pure conditioning change (DDIM)"},
    3: {"sampler": "bdia", "conditioning": "asymmetric", "note": "pure BDIA, semantics unchanged (the honest paper number)"},
    4: {"sampler": "bdia", "conditioning": "symmetric_null", "note": "BDIA + symmetric null"},
}
RUNNABLE_CELLS = (2, 3, 4)
ARRAY_KINDS = ("x0", "anonymous_float", "anonymous_public_available", "recon_m1")
IDENTITY_FIELDS = {"schema_version", "seed", "noise_level", "rounds", "permutation", "cell", "sampler", "conditioning"}


# --------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Sample:
    sample_id: str
    patient_id: str
    label: str
    x0: torch.Tensor  # (1,1,256,256) float32 in [0,1], CPU


def base_kwargs(reference: torch.Tensor) -> dict[str, Any]:
    """Model kwargs the predecessor passes (y=0 healthy class, no CFG mixing)."""
    return {"y": torch.zeros(reference.shape[0], device=reference.device, dtype=torch.int64),
            "p_uncond": -1, "clf_free": True}


class CountingModel:
    """Wraps the model callable to count evaluations (DDIM/BDIA cost-parity check)."""

    def __init__(self, model_fn: Callable, nfe: NFE):
        self.model_fn, self.nfe = model_fn, nfe

    def __call__(self, *args, **kwargs):
        self.nfe.bump()
        return self.model_fn(*args, **kwargs)


class DDIMBackend:
    """Predecessor DDIM sampler; ``generate_null`` switches generation to unconditional.

    ``generate_null=False`` reproduces the frozen S2 generation (y=0 conditional,
    guidance_scale=-1, no CFG mixing); ``True`` is the symmetric-null generation.
    Inversion is always null (``ddim_anonymization_forward`` forces it).
    """

    def __init__(self, diffusion, model_fn, noise_level: int, device: torch.device,
                 generate_null: bool, nfe: NFE):
        self.diffusion, self.noise_level, self.device = diffusion, noise_level, device
        self.generate_null, self.nfe = generate_null, nfe
        self.model_fn = CountingModel(model_fn, nfe)

    def invert(self, image: torch.Tensor) -> torch.Tensor:
        x_t, _ = self.diffusion.ddim_anonymization_forward(
            self.model_fn, image, self.noise_level, clip_denoised=True,
            model_kwargs=base_kwargs(image))
        return x_t

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        kwargs = base_kwargs(latent)
        if self.generate_null:
            kwargs["null"] = True
        return self.diffusion.ddim_anonymization_backward(
            self.model_fn, latent, self.noise_level, clip_denoised=True,
            model_kwargs=kwargs, guidance_scale=LEGACY_GUIDANCE_SCALE)


def build_backend(cell: int, diffusion, model_fn, noise_level: int, device: torch.device):
    """Sampler+conditioning backend for a cell, plus a shared NFE counter."""
    spec = CELLS[cell]
    generate_null = spec["conditioning"] == "symmetric_null"
    nfe = NFE()
    if spec["sampler"] == "ddim":
        return DDIMBackend(diffusion, model_fn, noise_level, device, generate_null, nfe), nfe
    # BDIA: invert kwargs always null; generate kwargs carry y=0 conditioning unless
    # symmetric-null.  A dummy sample on the right device seeds the y tensor; the y
    # tensor is rebuilt per call inside bdia_backend from these templates' device.
    ref = torch.zeros(1, 1, *IMAGE_SHAPE_HW, device=device)
    invert_kwargs = {**base_kwargs(ref), "null": True}
    generate_kwargs = {**base_kwargs(ref), **({"null": True} if generate_null else {})}
    counting = CountingModel(model_fn, nfe)
    return BDIABackend(diffusion, counting, noise_level, device, invert_kwargs, generate_kwargs), nfe


# --------------------------------------------------------------------------- array store
class LeanArrayStore:
    """One ``.npy`` per image kind (N,256,256) with per-slice SHA-256, flushed on close.

    Unlike q8's ArrayStore this run has a single method (S2) across cells, so arrays
    are not method-indexed; each cell writes its own run directory.
    """

    def __init__(self, directory: Path, count: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.maps = {kind: np.lib.format.open_memmap(
            self.directory / f"{kind}.npy", mode="w+", dtype="float32",
            shape=(count, *IMAGE_SHAPE_HW)) for kind in ARRAY_KINDS}

    def put(self, kind: str, index: int, array: np.ndarray) -> None:
        value = np.asarray(array).reshape(IMAGE_SHAPE_HW).astype("float32", copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite values for array {kind}[{index}]")
        self.maps[kind][index] = value

    def close(self) -> dict[str, Any]:
        evidence = {}
        for kind, memmap in self.maps.items():
            memmap.flush()
            path = self.directory / f"{kind}.npy"
            evidence[kind] = {"file": f"{self.directory.name}/{kind}.npy", "sha256": sha256_file(path),
                              "shape": list(memmap.shape), "dtype": "float32",
                              "slice_sha256": slice_hashes(memmap)}
        return evidence


# --------------------------------------------------------------------------- one sample x cell
def run_cell_sample(cell: int, sample: Sample, index: int, backend, nfe: NFE,
                    keys: KeyPolicy, registry: NonceRegistry, source_hashes: dict,
                    store: LeanArrayStore, device: torch.device,
                    cached_latent: torch.Tensor | None) -> dict[str, Any]:
    """Quality path for one sample in one cell (mirrors q8 run_s2's quality columns)."""
    torch.manual_seed(SEED)
    nfe.count = 0
    spec = CELLS[cell]
    row: dict[str, Any] = {
        # sample_id is unique per (image, cell) as the artifact validator requires; the
        # real image id and patient id are kept for paired cross-cell / patient-cluster analysis.
        "schema_version": SCHEMA_VERSION, "sample_id": f"{sample.sample_id}__cell{cell}",
        "base_sample_id": sample.sample_id, "patient_id": sample.patient_id,
        "label": sample.label, "seed": SEED, "noise_level": backend.noise_level, "rounds": ROUNDS,
        "permutation": int(USE_PERMUTATION), "cell": cell, "sampler": spec["sampler"],
        "conditioning": spec["conditioning"], "x0_sha256": sha256_array(sample.x0.numpy()),
    }
    meter = Meter(device, row)
    method = "S2"
    key, nonce = keys.master_key, keys.nonce(sample.sample_id, f"cell{cell}/m1")
    x0 = sample.x0.to(device)
    aad = {"model_sha256": source_hashes["checkpoint_hash"],
           "sampler_version": f"{LEGACY_PIPELINE}|cell{cell}:{spec['sampler']}/{spec['conditioning']}",
           "study_instance_uid": f"research:{sample.patient_id}",
           "sop_instance_uid": f"research:{sample.sample_id}"}

    if cached_latent is not None:
        z = cached_latent.to(device)
        meter.record(method, "diffusion_invert_x0", 0.0)
    else:
        z = meter(method, "diffusion_invert_x0", backend.invert, x0)
    row["latent_sha256"] = sha256_array(z.detach().cpu().numpy())
    latent_dtype = str(z.detach().cpu().numpy().dtype)

    normalized, original_norm = meter(method, "fixed_norm_scale", normalize_to_fixed_norm, z)
    protected = meter(method, "transform_forward", transform, normalized, key, nonce,
                      rounds=ROUNDS, use_permutation=USE_PERMUTATION)
    anonymous = meter(method, "diffusion_generate_anonymous", backend.generate, protected)
    row.update(image_columns(method, "anon", x0, anonymous))
    store.put("anonymous_float", index, anonymous[0, 0].cpu().numpy())

    sealed = meter(method, "container_seal_total", seal_anonymous_image, anonymous, original_norm,
                   key, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION, latent_shape=tuple(z.shape),
                   latent_dtype=latent_dtype, aad=aad, registry=registry)
    public = public_anonymous_image(sealed, device="cpu")[0, 0].numpy()
    store.put("anonymous_public_available", index, public)
    row.update(image_columns(method, "anonpub", x0, torch.from_numpy(public.copy())[None, None]))

    captured: dict[str, torch.Tensor] = {}

    def inverter(image: torch.Tensor) -> torch.Tensor:
        captured["z_hat_ano"] = meter(method, "diffusion_invert", backend.invert, image)
        return captured["z_hat_ano"]

    z_hat, open_wall = timed_call(device, open_anonymous_image, sealed, key, inverter, device=device)
    invert_seconds = row[col(method, "runtime", "diffusion_invert_seconds")]
    meter.record(method, "container_open_total", max(open_wall - invert_seconds, 0.0))
    recon = meter(method, "diffusion_generate_recon", backend.generate, z_hat)
    row.update(image_columns(method, "m1", x0, recon))
    row.update(latent_columns(method, "m1lat", z, z_hat))
    store.put("recon_m1", index, recon[0, 0].cpu().numpy())
    store.put("x0", index, sample.x0[0, 0].numpy())
    require_finite(sample.sample_id, z, protected, anonymous, captured["z_hat_ano"], z_hat, recon)

    row[col(method, "nfe", "model_evals")] = int(nfe.count)
    row[col(method, "nfe", "per_pass")] = int(backend.noise_level - 1)
    row[col("ALL", "mem", "process_peak_rss_mb")] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    for name, value in row.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise RuntimeError(f"non-finite {name} in {sample.sample_id} cell {cell}")
    return row


# --------------------------------------------------------------------------- driver
def execute(args, samples: list[Sample], diffusion, model_fn, device: torch.device,
            source_hashes: dict, stdout_capture, stderr_capture, backend_factory=None) -> Path:
    backend_factory = backend_factory or build_backend
    if len(samples) < 1:
        raise SystemExit("need at least one sample")
    if len({s.patient_id for s in samples}) != len(samples):
        raise SystemExit("patient IDs must be unique (CI treats each image as one patient)")
    key_mode = resolve_key_mode(args.evidence_role, args.key_mode)
    if key_mode == "fixture" and args.evidence_role not in FIXTURE_ROLES:
        raise SystemExit("fixture key mode is limited to synthetic_selftest/smoke")
    cells = tuple(args.cells)
    if not cells or set(cells) - set(RUNNABLE_CELLS):
        raise SystemExit(f"--cells must be a non-empty subset of {RUNNABLE_CELLS} (cell 1 is reused)")

    keys = KeyPolicy(key_mode)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(cells=list(cells), rounds=ROUNDS, permutation=USE_PERMUTATION,
                  fixed_norm=float(FIXED_NORM), n_samples=len(samples), key_mode=key_mode,
                  master_key_sha256=keys.master_key_sha256, **source_hashes)
    context = open_run(args.task_id, config, args.artifacts_root)
    run_dir = context.run_dir
    store = LeanArrayStore(run_dir / "arrays", len(samples) * len(cells))
    rows: list[dict[str, Any]] = []
    slot = 0
    with NonceRegistry(run_dir / "nonce_registry.sqlite3") as registry:
        with torch.inference_mode():
            for cell in cells:
                backend, nfe = backend_factory(cell, diffusion, model_fn, args.noise_level, device)
                for _ in range(args.warmup):  # untimed allocator / CUDA warm-up
                    backend.invert(backend.generate(samples[0].x0.to(device)))
                for sample in samples:
                    cached = args.ddim_latents.get(sample.sample_id) if (
                        args.ddim_latents and CELLS[cell]["sampler"] == "ddim") else None
                    try:
                        rows.append(run_cell_sample(cell, sample, slot, backend, nfe, keys, registry,
                                                    source_hashes, store, device, cached))
                    except Exception as error:  # fail closed
                        failure = {"sample_id": sample.sample_id, "cell": cell, "slot": slot,
                                   "completed": len(rows), "error": f"{type(error).__name__}: {error}",
                                   "traceback": traceback.format_exc()}
                        (run_dir / "failures.json").write_text(json.dumps([failure], indent=2) + "\n", encoding="utf-8")
                        print(f"FAILED {sample.sample_id} cell {cell}: {failure['error']}", file=sys.stderr)
                        raise SystemExit(FAILURE_EXIT_CODE) from error
                    print(f"cell {cell} {sample.sample_id}: m1 psnr={rows[-1][col('S2','m1','psnr_db')]:.2f} "
                          f"anon psnr={rows[-1][col('S2','anon','psnr_db')]:.2f} "
                          f"nfe={rows[-1][col('S2','nfe','model_evals')]}", flush=True)
                    slot += 1
        nonce_count = registry.count()

    evidence = store.close()
    index_path = run_dir / "array_index.json"
    index_path.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    numeric = [k for k, v in rows[0].items()
               if k not in IDENTITY_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool)]
    manifest = {
        "scheme": f"appendix-2x2:{S2_SCHEME}", "rounds": ROUNDS, "permutation": USE_PERMUTATION,
        "cells": list(cells), "cell_specs": {str(c): CELLS[c] for c in cells},
        "fixed_norm": float(FIXED_NORM), "seed_list": [SEED] * len(rows), "steps": args.noise_level,
        "key_mode": key_mode, "nonce_mode": KEY_MODE_TEXT[key_mode], "master_key_sha256": keys.master_key_sha256,
        "container_version": f"{VERSION} with {S2_SCHEME} parameters", "schema_version": SCHEMA_VERSION,
        "evidence_role": args.evidence_role, "formal": args.evidence_role == "formal",
        "pipeline": LEGACY_PIPELINE, "guidance_scale": LEGACY_GUIDANCE_SCALE,
        "generation_conditioning_asymmetric": GENERATION_CONDITIONING,
        "generation_conditioning_symmetric": "null=True (unconditional) both directions",
        "inversion_conditioning": "null=True (unconditional)", "warmup_runs": args.warmup,
        "gpu_memory_measured": device.type == "cuda", "synthetic_backend": args.synthetic_selftest,
        "array_index_file": index_path.name, "array_index_sha256": sha256_file(index_path),
        "arrays": {k: {kk: vv for kk, vv in e.items() if kk != "slice_sha256"} for k, e in evidence.items()},
        "failure_policy": "fail closed: any sample failure aborts without per_sample.csv", "failures": 0,
        "nonce_registry_count": nonce_count, "raw_key_material_recorded": False,
        "reuses_main_table_cell": "cell 1 (DDIM+asymmetric) is Q8_FORMAL S2; not re-run here",
        "ddim_latent_cache_reused": args.ddim_latent_cache and str(args.ddim_latent_cache) or "none (on-the-fly inversion)",
        "security_claim": "QUALITY ablation only; symmetric cells change anonymization semantics; "
                          "no anonymity/security equivalence is claimed here (see bdia_security_verify.py)",
        **source_hashes,
    }
    return finalize_run(context, Path(__file__), rows, numeric, manifest,
                        stdout_capture.getvalue(), stderr_capture.getvalue())


# --------------------------------------------------------------------------- inputs
def build_real_inputs(args):
    from scripts.e1_ddim_runner import create_runtime, load_split, preprocess

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (use --synthetic-selftest for the CPU plumbing check)")
    rows = load_split(args.split)
    selected = rows[:args.limit] if args.limit else rows
    samples = [Sample(r["sample_id"], r["patient_id"], r["label"], preprocess(Path(r["local_path"]))[0].unsqueeze(0))
               for r in selected]
    checkpoint_hash = sha256_file(args.checkpoint)
    sampler_hash = sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py")
    bdia_hash = sha256_file(Path("scripts/bdia_backend.py"))
    source_hashes = {"dataset_split_hash": rows[0]["split_sha256"], "checkpoint_hash": checkpoint_hash,
                     "source_sampler_sha256": sampler_hash, "bdia_backend_sha256": bdia_hash}
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    args.ddim_latents = load_ddim_latent_cache(args, rows, selected) if args.ddim_latent_cache else {}
    return samples, diffusion, model_fn, device, source_hashes


def load_ddim_latent_cache(args, rows, selected) -> dict[str, torch.Tensor]:
    """Reuse the frozen E2.3 DDIM-null latents for DDIM cells (provenance-pinned)."""
    from scripts.e2_roundtrip_runner import load_latent_cache

    latents = load_latent_cache(Path(args.ddim_latent_cache), rows)
    by_id = {r["sample_id"]: torch.from_numpy(np.ascontiguousarray(latents[i])).unsqueeze(0)
             for i, r in enumerate(rows)}
    return {r["sample_id"]: by_id[r["sample_id"]] for r in selected}


def build_synthetic_inputs(args):
    synthetic = synthetic_samples(args.limit or 3)
    samples = [Sample(s.sample_id, s.patient_id, s.label, s.x0) for s in synthetic]
    h = hashlib.sha256(b"bdia-2x2-synthetic-selftest").hexdigest()
    source_hashes = {"dataset_split_hash": h, "checkpoint_hash": h, "source_sampler_sha256": h,
                     "bdia_backend_sha256": sha256_file(Path("scripts/bdia_backend.py"))}
    return samples, None, None, torch.device("cpu"), source_hashes


class _SyntheticCellBackend(SyntheticBackend):
    """SyntheticBackend with the DDIM/BDIA backend surface (noise_level, no model)."""

    def __init__(self, noise_level: int):
        self.noise_level = noise_level


def parse_args(argv=None):
    from scripts.output_guard import add_force_flag

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evidence-role", choices=EVIDENCE_ROLES, default="synthetic_selftest")
    parser.add_argument("--cells", type=int, nargs="+", default=list(RUNNABLE_CELLS))
    parser.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--ddim-latent-cache", type=Path, default=None,
                        help="E2.3 DDIM-null cache to reuse for DDIM cells (optional; else on-the-fly)")
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="first N split rows; 0 = all")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--key-mode", choices=("fixture", "random"), default=None)
    parser.add_argument("--synthetic-selftest", action="store_true")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="BDIA2X2")
    add_force_flag(parser)
    args = parser.parse_args(argv)
    if args.synthetic_selftest != (args.evidence_role == "synthetic_selftest"):
        raise SystemExit("--synthetic-selftest and --evidence-role synthetic_selftest must be used together")
    if not args.synthetic_selftest and not args.checkpoint:
        raise SystemExit("--checkpoint is required for real runs")
    if args.limit < 0 or args.warmup < 0:
        raise SystemExit("--limit/--warmup must be non-negative")
    args.ddim_latents = {}
    return args


def main(argv=None) -> None:
    from contextlib import redirect_stderr, redirect_stdout
    from scripts.e1_ddim_runner import Tee

    args = parse_args(argv)
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        if args.synthetic_selftest:
            samples, diffusion, model_fn, device, source_hashes = build_synthetic_inputs(args)
            # In synthetic mode every cell runs the same pointwise backend (wiring test).
            def synthetic_factory(cell, _diffusion, _model_fn, noise_level, _device):
                return _SyntheticCellBackend(noise_level), NFE()
            execute(args, samples, diffusion, model_fn, device, source_hashes,
                    stdout_capture, stderr_capture, backend_factory=synthetic_factory)
        else:
            samples, diffusion, model_fn, device, source_hashes = build_real_inputs(args)
            execute(args, samples, diffusion, model_fn, device, source_hashes, stdout_capture, stderr_capture)


if __name__ == "__main__":
    main()
