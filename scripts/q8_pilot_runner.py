#!/usr/bin/env python3
"""Q8.1-Q8.5 / P8.1 pilot runner: P, S0, S1 and S2-fixed-norm/v2 on one split.

Per image and method it records, in ONE wide per-sample row (WORKFLOW 3.2 schema):

  anon          anonymous generator output vs x0                   (Q8.2)
  m1 / m1lat    image-only path: store -> invert -> decrypt -> generate, image
                vs x0 and decrypted latent vs z                    (Q8.1)
  t1 / t1lat    exact-latent (M2) path: Enc -> Dec -> generate     (Q8.1)
  xform         transform-only latent round trip                   (Q8.1)
  wrongkey / wrongnonce   decrypt-without-authentication counterfactuals: the
                inverted latent is inverse-transformed with a wrong key / nonce
                while the decryptor is still GIVEN the true value range and norm
                (a conservative oracle)                             (Q8.3)
  auth          S2 only: authenticated open with a wrong key / forged nonce is
                rejected before the inverter is ever called         (Q8.3)
  runtime/mem   disjoint transform / KDF / MAC / container / codec / diffusion
                stages plus GPU peak memory                         (Q8.4)
  store         nonce / tag / manifest / private-metadata / latent-payload /
                image bytes                                         (Q8.5)

Image arrays (x0, anonymous, attacker-visible stored pixels, reconstructions) are
written beside the rows with per-slice SHA-256 so the CPU analyzer
(``scripts/q8_analyze.py``) recomputes NN rank, LPIPS and the P8 grid without a
GPU.  LPIPS is deliberately NOT computed here.

Storage cells: ``range_preserving`` is the externally claimed main cell (S2 uses
the authenticated private-range container, P/S0/S1 the public-range PNG);
``legacy`` (optional, ``--storages range_preserving legacy``) is the predecessor's
range-discarding PNG as an attribution cell.  For S2 the legacy cell is a
counterfactual that bypasses the container (no lo/hi exist to protect).

Failure policy (fail closed): any exception or non-finite value in any sample
aborts the run, writes ``failures.json`` and exits 4 without per_sample/summary,
so a partial run can never validate.  Retries use a NEW run ID; seeds, samples and
settings are never changed to obtain better numbers (loop_engineering_spec 4.4).

Example (GPU; asks the user first when the estimate exceeds 30 minutes):
  scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id Q8_PILOT_SMOKE \\
      --validate-artifacts -- python -m scripts.q8_pilot_runner --evidence-role smoke \\
      --limit 2 --split splits/dev_v1.1.csv --checkpoint <ckpt> --latent-cache <E2.3 dev cache>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import resource
import secrets
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from scripts import e2_legacy_wrapper as legacy
from scripts.e2_anonymization_runner import _sync, image_pair_metrics, latent_pair_metrics
from scripts.e2_legacy_wrapper import (
    GENERATION_CONDITIONING, LEGACY_GUIDANCE_SCALE, LEGACY_PIPELINE, apply_legacy_key,
    legacy_generate, legacy_invert,
)
from scripts.m1_storage import decode_range_preserving_png, encode_range_preserving_png
from scripts.q8_metrics import (
    ARRAY_KINDS, METHODS, SCHEMA_VERSION, SELECTION_RULE, SHARED, STORAGES, ArrayStore, col,
    container_storage_breakdown, load_verified_arrays,
)
from scripts.run_artifacts import finalize_run, open_run, sha256_file
from src.crypto.container import VERSION, deserialize_container
from src.crypto.errors import AuthenticationError
from src.crypto.kdf import MAC_LABEL, PRIVATE_METADATA_LABEL, derive_subkey, round_label
from src.crypto.mac import compute_tag, open_authenticated
from src.crypto.nonce import NonceRegistry
from src.crypto.s2_fixed_norm import (
    FIXED_NORM, SCHEME as S2_SCHEME, normalize_to_fixed_norm, open_anonymous_image, open_latent,
    public_anonymous_image, restore_original_norm, seal_anonymous_image, seal_latent,
)
from src.crypto.sot import inverse_transform, transform


SEED = 1911
ROUNDS = 1  # frozen D6/D7 S2 setting
USE_PERMUTATION = False
MAIN_STORAGE = STORAGES[0]
FIXTURE_KEY = hashlib.sha256(b"cxr-s2/Q8/public-reproducibility-fixture/v1").digest()
FAILURE_EXIT_CODE = 4
EVIDENCE_ROLES = ("smoke", "pilot", "formal", "synthetic_selftest")
IMAGE_METRIC_NAMES = (
    "psnr_db", "ssim", "mse", "rmse", "mae", "linf", "cosine",
    "uint8_pixel_equality_rate", "float32_bit_exact_rate",
)
VS_P_METRIC_NAMES = ("psnr_db", "ssim", "cosine", "linf")
IMAGE_SHAPE_HW = (256, 256)
IDENTITY_FIELDS = {"schema_version", "seed", "noise_level", "rounds", "permutation"}


KEY_MODES = ("fixture", "random")
FIXTURE_ROLES = ("synthetic_selftest", "smoke")  # the only roles allowed to use deterministic fixtures
KEY_MODE_TEXT = {
    "fixture": "deterministic public reproducibility fixture (synthetic_selftest/smoke only)",
    "random": "per-run random 256-bit master key and random unique 128-bit nonces, NonceRegistry-enforced",
}


def resolve_key_mode(role: str, requested: str | None) -> str:
    """Pilot/formal never reuse a cross-run deterministic key or nonce stream."""
    mode = requested or ("fixture" if role in FIXTURE_ROLES else "random")
    if mode not in KEY_MODES:
        raise SystemExit(f"--key-mode must be one of {KEY_MODES}")
    if mode == "fixture" and role not in FIXTURE_ROLES:
        raise SystemExit(f"deterministic fixture key/nonces are limited to roles {FIXTURE_ROLES}; {role} needs --key-mode random")
    return mode


class KeyPolicy:
    """Master key, nonces and wrong-key/nonce material for one run.

    ``fixture``: public deterministic values (reproducible plumbing checks only).
    ``random``: a fresh CSPRNG master key and fresh random nonces per run; the nonce
    registry still rejects reuse at seal time and this class additionally keeps all
    drawn nonces of the run distinct (the wrong nonce is never registered).  The raw
    key never leaves this object; only its SHA-256 is recorded.
    """

    def __init__(self, mode: str):
        if mode not in KEY_MODES:
            raise ValueError(f"unknown key mode: {mode}")
        self.mode = mode
        self._master = FIXTURE_KEY if mode == "fixture" else secrets.token_bytes(32)
        self._drawn: set[bytes] = set()

    def __repr__(self) -> str:  # never print key material
        return f"KeyPolicy(mode={self.mode!r}, master_key_sha256={self.master_key_sha256[:12]})"

    @property
    def master_key(self) -> bytes:
        return self._master

    @property
    def master_key_sha256(self) -> str:
        return hashlib.sha256(self._master).hexdigest()

    def nonce(self, sample_id: str, role: str) -> bytes:
        if self.mode == "fixture":
            return hashlib.sha256(f"cxr-s2/Q8/{sample_id}/{role}".encode()).digest()[:16]
        while True:
            candidate = secrets.token_bytes(16)
            if candidate not in self._drawn:
                self._drawn.add(candidate)
                return candidate

    def wrong_key(self, sample_id: str) -> bytes:
        if self.mode == "fixture":
            return hashlib.sha256(f"cxr-s2/Q8/wrong-key/{sample_id}".encode()).digest()
        while True:
            candidate = secrets.token_bytes(32)
            if candidate != self._master:
                return candidate


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# --------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Sample:
    sample_id: str
    patient_id: str
    label: str
    x0: torch.Tensor  # (1,1,256,256) float32 in [0,1], CPU
    z: torch.Tensor   # (1,1,256,256) float32 cached key-independent inversion latent, CPU


class DiffusionBackend:
    """The only GPU-touching component: the predecessor's generate/invert calls."""

    def __init__(self, diffusion, model_fn, noise_level: int, device: torch.device):
        self.diffusion, self.model_fn, self.noise_level, self.device = diffusion, model_fn, noise_level, device

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return legacy_generate(self.diffusion, self.model_fn, latent, self.noise_level)

    def invert(self, image: torch.Tensor) -> torch.Tensor:
        return legacy_invert(self.diffusion, self.model_fn, image, self.noise_level)


class SyntheticBackend:
    """CPU stand-in for plumbing tests only: a pointwise invertible 'generator'.

    It is NOT a model of the diffusion pipeline and its numbers carry no scientific
    meaning; runs using it are labelled ``synthetic_selftest`` and rejected by the
    analyzer unless ``--allow-synthetic`` is given.
    """

    device = torch.device("cpu")
    noise_level = 0

    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * torch.tanh(0.5 * latent)

    def invert(self, image: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.atanh((2 * image - 1).clamp(-0.999, 0.999))


def synthetic_samples(count: int, seed: int = SEED) -> list[Sample]:
    """Smooth deterministic fake 'images' and their exact synthetic latents."""
    backend = SyntheticBackend()
    samples = []
    for index in range(count):
        generator = torch.Generator().manual_seed(seed + index)
        noise = torch.randn(1, 1, 32, 32, generator=generator)
        smooth = torch.nn.functional.interpolate(noise, size=(256, 256), mode="bilinear", align_corners=False)
        x0 = (smooth - smooth.min()) / (smooth.max() - smooth.min())
        samples.append(Sample(
            f"synthetic_{index:04d}", f"patient{index:04d}", str(index % 2), x0.float(),
            backend.invert(x0).float(),
        ))
    return samples


# --------------------------------------------------------------------------- metering
class Meter:
    """Times one stage (device-synchronised) and records its GPU peak memory."""

    def __init__(self, device: torch.device, row: dict[str, Any]):
        self.device, self.row, self.cuda = device, row, device.type == "cuda"

    def __call__(self, method: str, stage: str, function: Callable, *args, **kwargs):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        _sync(self.device)
        start = time.perf_counter()
        value = function(*args, **kwargs)
        _sync(self.device)
        self.row[col(method, "runtime", f"{stage}_seconds")] = time.perf_counter() - start
        if self.cuda:
            self.row[col(method, "mem", f"{stage}_gpu_peak_mb")] = torch.cuda.max_memory_allocated(self.device) / 2**20
        return value

    def record(self, method: str, stage: str, seconds: float) -> None:
        self.row[col(method, "runtime", f"{stage}_seconds")] = seconds


@contextlib.contextmanager
def legacy_password(password: str):
    """Temporarily replace the frozen legacy S0/S1 password (wrong-key control only)."""
    original = legacy.LEGACY_PASSWORD
    legacy.LEGACY_PASSWORD = password
    try:
        yield
    finally:
        legacy.LEGACY_PASSWORD = original


# --------------------------------------------------------------------------- storage hand-offs
def _png_bytes(quantized: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(quantized, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def encode_legacy(image: torch.Tensor) -> tuple[bytes, np.ndarray]:
    """Predecessor storage: per-image min-max, uint8 truncation, range discarded."""
    low, high = image.amin(dim=(1, 2, 3), keepdim=True), image.amax(dim=(1, 2, 3), keepdim=True)
    quantized = ((image - low) / (high - low) * 255).to(torch.uint8)[0, 0].cpu().numpy()
    return _png_bytes(quantized), quantized


def decode_legacy(payload: bytes, device: torch.device) -> torch.Tensor:
    with Image.open(io.BytesIO(payload)) as handle:
        reloaded = torch.from_numpy(np.asarray(handle, dtype=np.uint8).astype(np.float32) / 255.0)
    reloaded = reloaded[None, None].to(device)
    low, high = reloaded.amin(dim=(1, 2, 3), keepdim=True), reloaded.amax(dim=(1, 2, 3), keepdim=True)
    return (reloaded - low) / (high - low)


def encode_range(image: torch.Tensor) -> tuple[bytes, np.ndarray]:
    payload, _record = encode_range_preserving_png(image[0, 0].detach().float().cpu().numpy())
    with Image.open(io.BytesIO(payload)) as handle:
        return payload, np.asarray(handle, dtype=np.uint8).copy()


def decode_range(payload: bytes, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(decode_range_preserving_png(payload).copy())[None, None].to(device)


STORE_CODECS = {"range_preserving": (encode_range, decode_range), "legacy": (encode_legacy, decode_legacy)}


# --------------------------------------------------------------------------- row helpers
def image_columns(method, group, reference, other, names=IMAGE_METRIC_NAMES) -> dict[str, float]:
    metrics = image_pair_metrics("", reference, other)
    return {col(method, group, name): float(metrics[f"_{name}"]) for name in names}


def latent_columns(method, group, reference, other) -> dict[str, float]:
    return {col(method, group, key[1:]): float(value) for key, value in latent_pair_metrics("", reference, other).items()}


def require_finite(sample_id: str, *tensors: torch.Tensor) -> None:
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise RuntimeError(f"non-finite tensor in sample {sample_id}")


def add_stage_sums(row: dict[str, Any], method: str, anonymize: tuple[str, ...], decrypt: tuple[str, ...]) -> None:
    """Stage sums starting from the CACHED key-independent latent z (E2.3).

    ``cached_latent_to_m1``  = transform/seal + generation + M1 storage encode.
    ``m1_to_reconstruction`` = M1 decode/open + inversion + inverse transform + generation.
    Both are sums of disjoint measured stages of the same sample (no extra clock) and
    EXCLUDE the initial inversion of the original image x0 -> z, which the E2.3 cache
    performed once, key-independently, before this run.
    """
    def total(stages):
        return sum(row[col(method, "runtime", f"{stage}_seconds")] for stage in stages)
    row[col(method, "runtime", "cached_latent_to_m1_seconds")] = total(anonymize)
    row[col(method, "runtime", "m1_to_reconstruction_seconds")] = total(decrypt)


@dataclass
class Context:
    backend: Any
    device: torch.device
    source_root: Path
    storages: tuple[str, ...]
    store: ArrayStore
    registry: NonceRegistry
    model_sha256: str
    aad_template: dict[str, str]
    keys: "KeyPolicy"


# --------------------------------------------------------------------------- P / S0 / S1
def run_baseline(method: str, sample: Sample, index: int, ctx: Context, meter: Meter, row: dict[str, Any], p_output: dict) -> None:
    device, backend, store = ctx.device, ctx.backend, ctx.store
    x0, z = sample.x0.to(device), sample.z.to(device)
    z_ano = meter(method, "transform_forward", apply_legacy_key, z, ctx.source_root, method, "anonymize")
    anonymous = meter(method, "diffusion_generate_anonymous", backend.generate, z_ano)
    if method == "P":
        p_output["value"] = anonymous
    row.update(image_columns(method, "anon", x0, anonymous))
    store.put("anonymous_float", index, anonymous[0, 0].cpu().numpy(), method)

    z_roundtrip = apply_legacy_key(z_ano, ctx.source_root, method, "deanonymize")
    row.update(latent_columns(method, "xform", z, z_roundtrip))
    exact = bool(torch.equal(z_roundtrip, z))
    row[col(method, "t1", "reused_p_output")] = int(exact)
    if exact:  # identical generator input => identical P output; do not re-run, do not fake a time
        meter.record(method, "diffusion_generate_t1", 0.0)
        t1_output = p_output["value"]
    else:
        t1_output = meter(method, "diffusion_generate_t1", backend.generate, z_roundtrip)
    row.update(image_columns(method, "t1", x0, t1_output))
    row.update(latent_columns(method, "t1lat", z, z_roundtrip))
    store.put("recon_t1", index, t1_output[0, 0].cpu().numpy(), method)
    row[col(method, "store", "m2_latent_payload_bytes")] = int(z_ano.numel() * 4)  # float32, no container

    main_inverted = None
    for storage in ctx.storages:
        encode, decode = STORE_CODECS[storage]
        main = storage == MAIN_STORAGE
        group, suffix = ("m1", "") if main else ("m1leg", "_legacy")
        payload, quantized = meter(method, f"storage_encode{suffix}", encode, anonymous)
        stored = meter(method, f"storage_decode{suffix}", decode, payload, device)
        if main:  # public_available view: the range-restored PNG (lo/hi are public for P/S0/S1)
            row.update(image_columns(method, "anonpub", x0, stored))
            store.put("anonymous_public_available", index, stored[0, 0].cpu().numpy(), method)
        z_hat_ano = meter(method, f"diffusion_invert{suffix}", backend.invert, stored)
        z_hat = meter(method, f"transform_inverse{suffix}", apply_legacy_key, z_hat_ano, ctx.source_root, method, "deanonymize")
        recon = meter(method, f"diffusion_generate_recon{suffix}", backend.generate, z_hat)
        row.update(image_columns(method, group, x0, recon))
        row.update(image_columns(method, f"{group}_vs_p", p_output["value"], recon, VS_P_METRIC_NAMES))
        row.update(latent_columns(method, f"{group}lat", z, z_hat))
        require_finite(sample.sample_id, stored, z_hat_ano, z_hat, recon)
        if main:
            row[col(method, "store", "m1_image_payload_bytes")] = len(payload)
            store.put("anonymous_public_q", index, quantized, method)
            store.put("recon_m1", index, recon[0, 0].cpu().numpy(), method)
            main_inverted = z_hat_ano
        else:
            store.put("recon_m1_legacy", index, recon[0, 0].cpu().numpy(), method)

    if method != "P":  # P has no key; nothing to get wrong
        with legacy_password(f"Q8-wrong-key/{sample.sample_id}"):
            z_wrong = apply_legacy_key(main_inverted, ctx.source_root, method, "deanonymize")
        wrong = meter(method, "control_generate_wrong_key", backend.generate, z_wrong)
        row.update(image_columns(method, "wrongkey", x0, wrong))
        row.update(latent_columns(method, "wrongkeylat", z, z_wrong))
        store.put("recon_wrong_key", index, wrong[0, 0].cpu().numpy(), method)
        require_finite(sample.sample_id, z_wrong, wrong)
    require_finite(sample.sample_id, anonymous, t1_output)
    add_stage_sums(
        row, method,
        ("transform_forward", "diffusion_generate_anonymous", "storage_encode"),
        ("storage_decode", "diffusion_invert", "transform_inverse", "diffusion_generate_recon"),
    )


# --------------------------------------------------------------------------- S2-fixed-norm/v2
def timed_call(sync_device: torch.device, function: Callable, /, *args, **kwargs):
    _sync(sync_device)
    start = time.perf_counter()
    value = function(*args, **kwargs)
    _sync(sync_device)
    return value, time.perf_counter() - start


def run_s2(sample: Sample, index: int, ctx: Context, meter: Meter, row: dict[str, Any], p_output: dict) -> None:
    method, device, backend, store = "S2", ctx.device, ctx.backend, ctx.store
    key, wrong_key = ctx.keys.master_key, ctx.keys.wrong_key(sample.sample_id)
    nonce_m1, nonce_m2 = ctx.keys.nonce(sample.sample_id, "m1"), ctx.keys.nonce(sample.sample_id, "m2")
    wrong_nonce = ctx.keys.nonce(sample.sample_id, "wrong-nonce")
    x0, z = sample.x0.to(device), sample.z.to(device)
    aad = {**ctx.aad_template, "study_instance_uid": f"research:{sample.patient_id}",
           "sop_instance_uid": f"research:{sample.sample_id}"}
    latent_dtype = str(z.detach().cpu().numpy().dtype)

    normalized_and_norm = meter(method, "fixed_norm_scale", normalize_to_fixed_norm, z)
    normalized, original_norm = normalized_and_norm
    protected = meter(method, "transform_forward", transform, normalized, key, nonce_m1,
                      rounds=ROUNDS, use_permutation=USE_PERMUTATION)
    anonymous = meter(method, "diffusion_generate_anonymous", backend.generate, protected)
    row.update(image_columns(method, "anon", x0, anonymous))
    store.put("anonymous_float", index, anonymous[0, 0].cpu().numpy(), method)
    sealed = meter(method, "container_seal_total", seal_anonymous_image, anonymous, original_norm, key,
                   nonce_m1, rounds=ROUNDS, use_permutation=USE_PERMUTATION, latent_shape=tuple(z.shape),
                   latent_dtype=latent_dtype, aad=aad, registry=ctx.registry)

    public = public_anonymous_image(sealed, device="cpu")[0, 0].numpy()  # q/255: lo/hi are encrypted
    store.put("anonymous_public_q", index, np.rint(public * 255).astype(np.uint8), method)
    store.put("anonymous_public_available", index, public, method)
    row.update(image_columns(method, "anonpub", x0, torch.from_numpy(public.copy())[None, None]))
    for key_name, value in container_storage_breakdown(sealed).items():
        if key_name != "mode":
            row[col(method, "store", f"m1_{key_name}")] = int(value)

    transform_only = restore_original_norm(inverse_transform(
        protected, key, nonce_m1, rounds=ROUNDS, use_permutation=USE_PERMUTATION), original_norm)
    row.update(latent_columns(method, "xform", z, transform_only))

    # Legitimate image-only open: verify -> metadata AEAD -> decode -> invert -> inverse -> restore norm.
    captured: dict[str, torch.Tensor] = {}

    def inverter(image: torch.Tensor) -> torch.Tensor:
        captured["z_hat_ano"] = meter(method, "diffusion_invert", backend.invert, image)
        return captured["z_hat_ano"]

    z_hat, open_wall = timed_call(device, open_anonymous_image, sealed, key, inverter, device=device)
    invert_seconds = row[col(method, "runtime", "diffusion_invert_seconds")]
    meter.record(method, "container_open_total", max(open_wall - invert_seconds, 0.0))
    recon = meter(method, "diffusion_generate_recon", backend.generate, z_hat)
    row.update(image_columns(method, "m1", x0, recon))
    row.update(image_columns(method, "m1_vs_p", p_output["value"], recon, VS_P_METRIC_NAMES))
    row.update(latent_columns(method, "m1lat", z, z_hat))
    store.put("recon_m1", index, recon[0, 0].cpu().numpy(), method)
    require_finite(sample.sample_id, protected, anonymous, captured["z_hat_ano"], z_hat, recon)

    # Component micro-benchmarks on the same inputs (not additive with the pipeline stages).
    def derive_all():
        labels = [round_label(r, p) for r in range(ROUNDS) for p in ("sign-pre", "sign-post")]
        for label in (*labels, MAC_LABEL, PRIVATE_METADATA_LABEL):
            derive_subkey(key, nonce_m1, label)
    meter(method, "micro_kdf_subkeys", derive_all)
    meter(method, "micro_mac_compute", compute_tag, key, replace_tag(deserialize_container(sealed)))
    meter(method, "micro_mac_verify", open_authenticated, sealed, key, lambda container: None)
    meter(method, "micro_transform_inverse", inverse_transform, captured["z_hat_ano"], key, nonce_m1,
          rounds=ROUNDS, use_permutation=USE_PERMUTATION)

    # T1 / M2: exact latent payload through the authenticated M2 container.
    sealed_m2 = meter(method, "m2_seal_total", seal_latent, z, key, nonce_m2, rounds=ROUNDS,
                      use_permutation=USE_PERMUTATION, aad=aad, registry=ctx.registry)
    opened = meter(method, "m2_open_total", open_latent, sealed_m2, key, device=device)
    t1_output = meter(method, "diffusion_generate_t1", backend.generate, opened)
    row[col(method, "t1", "reused_p_output")] = 0
    row.update(image_columns(method, "t1", x0, t1_output))
    row.update(latent_columns(method, "t1lat", z, opened))
    store.put("recon_t1", index, t1_output[0, 0].cpu().numpy(), method)
    for key_name, value in container_storage_breakdown(sealed_m2).items():
        if key_name != "mode":
            row[col(method, "store", f"m2_{key_name}")] = int(value)
    require_finite(sample.sample_id, opened, t1_output)

    # Wrong key / wrong nonce: authenticated rejection (fail closed, inverter never called).
    forged = replace_nonce(sealed, wrong_nonce)
    for condition, blob, open_key, stage in (
        ("wrong_key", sealed, wrong_key, "control_reject_wrong_key"),
        ("wrong_nonce", forged, key, "control_reject_wrong_nonce"),
    ):
        calls = [0]

        def counting_inverter(image: torch.Tensor, calls=calls) -> torch.Tensor:
            calls[0] += 1
            return torch.zeros(tuple(z.shape), device=device)

        start = time.perf_counter()
        try:
            open_anonymous_image(blob, open_key, counting_inverter, device=device)
            rejected = 0
        except AuthenticationError:
            rejected = 1
        meter.record(method, stage, time.perf_counter() - start)
        row[col(method, "auth", f"{condition}_rejected")] = rejected
        row[col(method, "auth", f"{condition}_inverter_calls")] = calls[0]
        # Engineering correctness gate (verify-before-open): abort, never record, a violation.
        if rejected != 1 or calls[0] != 0:
            raise RuntimeError(
                f"authenticated {condition} open for {sample.sample_id} violated fail-closed rules: "
                f"rejected={rejected}, inverter_calls={calls[0]}")

    # Bypass-authentication counterfactuals (what a wrong key/nonce would decrypt to).
    for condition, wrong_key_value, wrong_nonce_value, stage, kind in (
        ("wrongkey", wrong_key, nonce_m1, "control_generate_wrong_key", "recon_wrong_key"),
        ("wrongnonce", key, wrong_nonce, "control_generate_wrong_nonce", "recon_wrong_nonce"),
    ):
        z_wrong = restore_original_norm(inverse_transform(
            captured["z_hat_ano"], wrong_key_value, wrong_nonce_value,
            rounds=ROUNDS, use_permutation=USE_PERMUTATION), original_norm)
        wrong = meter(method, stage, backend.generate, z_wrong)
        row.update(image_columns(method, condition, x0, wrong))
        row.update(latent_columns(method, f"{condition}lat", z, z_wrong))
        store.put(kind, index, wrong[0, 0].cpu().numpy(), method)
        require_finite(sample.sample_id, z_wrong, wrong)

    if "legacy" in ctx.storages:  # attribution cell: S2 transform with the predecessor's PNG
        payload, _q = meter(method, "storage_encode_legacy", encode_legacy, anonymous)
        stored = meter(method, "storage_decode_legacy", decode_legacy, payload, device)
        z_hat_ano_leg = meter(method, "diffusion_invert_legacy", backend.invert, stored)
        z_hat_leg = meter(method, "transform_inverse_legacy", lambda value: restore_original_norm(
            inverse_transform(value, key, nonce_m1, rounds=ROUNDS, use_permutation=USE_PERMUTATION),
            original_norm), z_hat_ano_leg)
        recon_leg = meter(method, "diffusion_generate_recon_legacy", backend.generate, z_hat_leg)
        row.update(image_columns(method, "m1leg", x0, recon_leg))
        row.update(image_columns(method, "m1leg_vs_p", p_output["value"], recon_leg, VS_P_METRIC_NAMES))
        row.update(latent_columns(method, "m1leglat", z, z_hat_leg))
        store.put("recon_m1_legacy", index, recon_leg[0, 0].cpu().numpy(), method)
        require_finite(sample.sample_id, stored, z_hat_ano_leg, z_hat_leg, recon_leg)

    add_stage_sums(
        row, method,
        ("fixed_norm_scale", "transform_forward", "diffusion_generate_anonymous", "container_seal_total"),
        ("container_open_total", "diffusion_invert", "diffusion_generate_recon"),
    )


def replace_tag(container):
    """Same container with an empty tag, i.e. the exact bytes the MAC covers."""
    return container.with_tag(b"")


def replace_nonce(serialized: bytes, nonce: bytes) -> bytes:
    """Re-serialise with a different nonce but the ORIGINAL tag (a forged nonce)."""
    container = deserialize_container(serialized)
    return type(container)(
        nonce=nonce, parameters=container.parameters, aad=container.aad, storage=container.storage,
        ciphertext=container.ciphertext, tag=container.tag, version=container.version,
    ).serialize()


# --------------------------------------------------------------------------- driver
def process_sample(sample: Sample, partner: Sample, index: int, ctx: Context) -> dict[str, Any]:
    start = time.perf_counter()
    torch.manual_seed(SEED)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "sample_id": sample.sample_id, "patient_id": sample.patient_id,
        "label": sample.label, "seed": SEED, "noise_level": ctx.backend.noise_level,
        "rounds": ROUNDS, "permutation": int(USE_PERMUTATION),
        "x0_sha256": sha256_array(sample.x0.numpy()), "latent_sha256": sha256_array(sample.z.numpy()),
    }
    meter = Meter(ctx.device, row)
    ctx.store.put("x0", index, sample.x0[0, 0].numpy())
    p_output: dict[str, torch.Tensor] = {}
    for method in METHODS:  # P first: its output is the base-flow reference for the others
        if method == "S2":
            run_s2(sample, index, ctx, meter, row, p_output)
        else:
            run_baseline(method, sample, index, ctx, meter, row, p_output)
    # Negative control: an unrelated image/latent pair gives the numeric floor for "wrong".
    row.update(image_columns(SHARED, "ctrl_unrelated", sample.x0, partner.x0))
    row.update(latent_columns(SHARED, "ctrl_unrelated_lat", sample.z, partner.z))
    row[col(SHARED, "mem", "process_peak_rss_mb")] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    row[col(SHARED, "run", "sample_wall_seconds")] = time.perf_counter() - start
    for key, value in row.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise RuntimeError(f"non-finite {key} in {sample.sample_id}")
    return row


def array_kinds_for(storages: tuple[str, ...]) -> list[str]:
    return [kind for kind in ARRAY_KINDS if kind != "recon_m1_legacy" or "legacy" in storages]


def validate_request(args, samples: list[Sample]) -> None:
    if len(samples) < 2:
        raise SystemExit("Q8 needs at least 2 samples (unrelated-image control and NN gallery)")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise SystemExit("duplicate sample IDs")
    if len({sample.patient_id for sample in samples}) != len(samples):
        raise SystemExit("patient IDs must be unique (the CI treats each image as one independent patient)")
    if args.storages[0] != MAIN_STORAGE or len(set(args.storages)) != len(args.storages) or not set(args.storages) <= set(STORAGES):
        raise SystemExit(f"--storages must start with {MAIN_STORAGE} and use {STORAGES} without repeats")
    count = len(samples)
    if args.evidence_role == "smoke" and count > 4:
        raise SystemExit("smoke role is limited to 4 samples")
    if args.evidence_role == "formal" and (args.split.name != "security_v1.csv" or count != 200 or args.limit):
        raise SystemExit("formal role requires the complete 200-image security_v1 split")
    if args.evidence_role == "pilot" and count >= 200 and args.split.name == "security_v1.csv":
        raise SystemExit("a full 200-image security_v1 run must use --evidence-role formal")


def execute(args, samples: list[Sample], backend, provenance: dict[str, Any], stdout_capture, stderr_capture) -> Path:
    """Run every sample; write run artifacts or fail closed."""
    validate_request(args, samples)
    key_mode = resolve_key_mode(args.evidence_role, args.key_mode)
    keys = KeyPolicy(key_mode)
    device = backend.device
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        methods=list(METHODS), storages=list(args.storages), rounds=ROUNDS, permutation=USE_PERMUTATION,
        fixed_norm=float(FIXED_NORM), pipeline=LEGACY_PIPELINE, guidance_scale=LEGACY_GUIDANCE_SCALE,
        n_samples=len(samples), selection_rule=SELECTION_RULE, key_mode=key_mode,
        master_key_sha256=keys.master_key_sha256, **provenance.get("config", {}),
    )
    context = open_run(args.task_id, config, args.artifacts_root)
    run_dir = context.run_dir
    store = ArrayStore(run_dir / "arrays", len(samples), array_kinds_for(tuple(args.storages)))
    ctx_model_hash = provenance["checkpoint_hash"]
    rows: list[dict[str, Any]] = []
    with NonceRegistry(run_dir / "nonce_registry.sqlite3") as registry:
        ctx = Context(
            backend, device, args.source_root, tuple(args.storages), store, registry, ctx_model_hash,
            {"model_sha256": ctx_model_hash, "sampler_version": LEGACY_PIPELINE}, keys,
        )
        with torch.inference_mode():
            for _ in range(args.warmup):  # untimed, discarded: CUDA context / allocator warm-up
                warm = samples[0]
                backend.invert(backend.generate(warm.z.to(device)))
            for index, sample in enumerate(samples):
                partner = samples[(index + 1) % len(samples)]
                try:
                    rows.append(process_sample(sample, partner, index, ctx))
                except Exception as error:  # fail closed; keep the evidence of why
                    failure = {"sample_id": sample.sample_id, "index": index, "completed": len(rows),
                               "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
                    (run_dir / "failures.json").write_text(json.dumps([failure], indent=2) + "\n", encoding="utf-8")
                    print(f"FAILED {sample.sample_id}: {failure['error']}", file=sys.stderr)
                    raise SystemExit(FAILURE_EXIT_CODE) from error
                print(
                    f"{sample.sample_id}: S2 m1 psnr={rows[-1][col('S2', 'm1', 'psnr_db')]:.2f} "
                    f"P m1 psnr={rows[-1][col('P', 'm1', 'psnr_db')]:.2f} "
                    f"sample_wall={rows[-1][col(SHARED, 'run', 'sample_wall_seconds')]:.1f}s", flush=True,
                )
        nonce_count = registry.count()
    evidence = store.close()
    index_path = run_dir / "array_index.json"
    index_path.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    load_verified_arrays(run_dir, evidence)  # re-read from disk and verify every slice hash
    numeric = [key for key, value in rows[0].items()
               if key not in IDENTITY_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool)]
    manifest = {
        "scheme": f"Q8:P/S0/S1/{S2_SCHEME}", "rounds": ROUNDS, "permutation": USE_PERMUTATION,
        "dataset_split_hash": provenance["dataset_split_hash"], "checkpoint_hash": provenance["checkpoint_hash"],
        "source_sampler_sha256": provenance["source_sampler_sha256"], "seed_list": [SEED] * len(rows),
        "steps": backend.noise_level, "nonce_mode": KEY_MODE_TEXT[key_mode], "key_mode": key_mode,
        "master_key_sha256": keys.master_key_sha256,
        "reproducibility": (
            "fixture: ciphertexts/protected images are bit-reproducible across runs (plumbing checks only). "
            "random: protected images differ per run by design; reproducibility rests on the recorded split/"
            "checkpoint/cache hashes, per-sample x0/latent hashes, hash-verified arrays and the deterministic analyzer"),
        "container_version": f"{VERSION} with {S2_SCHEME} parameters",
        "q8_schema_version": SCHEMA_VERSION, "evidence_role": args.evidence_role,
        "formal": args.evidence_role == "formal", "methods": list(METHODS), "storages": list(args.storages),
        "main_storage": MAIN_STORAGE, "fixed_norm": float(FIXED_NORM), "private_metadata": "ChaCha20-Poly1305",
        "pipeline": LEGACY_PIPELINE, "guidance_scale": LEGACY_GUIDANCE_SCALE,
        "generation_conditioning": GENERATION_CONDITIONING, "inversion_conditioning": "null=True (unconditional)",
        "batch_size": 1, "warmup_runs": args.warmup, "gpu_memory_measured": device.type == "cuda",
        "synthetic_backend": isinstance(backend, SyntheticBackend), "lpips_computed_in_runner": False,
        "selection_rule": SELECTION_RULE, "array_index_file": index_path.name,
        "array_index_sha256": sha256_file(index_path),
        "arrays": {kind: {key: value for key, value in entry.items() if key != "slice_sha256"}
                   for kind, entry in evidence.items()},
        "failure_policy": "fail closed: any sample failure aborts without per_sample.csv",
        "failures": 0, "nonce_registry_count": nonce_count,
        "raw_key_material_recorded": False,
        **({"public_fixture_key_sha256": hashlib.sha256(FIXTURE_KEY).hexdigest()} if key_mode == "fixture" else {}),
        "timing_notes": {
            "cached_latent_to_m1": "sum of measured stages from the cached key-independent latent z (E2.3): transform/seal + generation + M1 encode; EXCLUDES the initial x0->z inversion",
            "m1_to_reconstruction": "sum of measured stages: M1 decode/open + inversion + inverse transform + generation; EXCLUDES the initial x0->z inversion",
            "micro_*": "isolated re-timing of one primitive; not additive with pipeline stages",
            "control_*": "wrong-key/nonce controls; excluded from deployment totals",
            "S0_S1_transform": "includes the wrapper's per-call legacy key construction (password KDF)",
            "S2_container": "container_* totals include KDF, private-metadata AEAD, HMAC, PNG codec and framing",
            "peak_rss": "process high-water mark (ru_maxrss), not per stage",
        },
        **provenance.get("manifest", {}),
    }
    return finalize_run(context, Path(__file__), rows, numeric, manifest,
                        stdout_capture.getvalue(), stderr_capture.getvalue())


def validate_latent_cache(
    cache_dir: Path, rows: list[dict[str, str]], x0_arrays: list[np.ndarray], *, noise_level: int, split_name: str,
    source_sampler_sha256: str, checkpoint_hash: str,
) -> dict[str, Any]:
    """Fail closed unless the E2.3 cache is exactly the one this run's settings imply.

    ``rows`` is the FULL split (order is part of the contract); ``x0_arrays`` are the
    freshly preprocessed originals (1,H,W) of the first ``len(x0_arrays)`` rows.  Uses the
    actual E2.3 manifest/config/index fields; nothing is inferred or defaulted.
    """
    from scripts.artifact_schema import validate_run
    from scripts.e2_roundtrip_runner import load_latent_cache

    cache_dir = Path(cache_dir)
    validation = validate_run(cache_dir)
    if not validation["valid"]:
        raise SystemExit(f"E2.3 cache run artifacts invalid: {validation['errors']}")
    status_path = cache_dir / "status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("state") != "succeeded":
        raise SystemExit("E2.3 cache managed run did not succeed")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "task_id": "E2.3", "steps": noise_level, "pipeline": LEGACY_PIPELINE,
        "guidance_scale": LEGACY_GUIDANCE_SCALE, "generation_conditioning": GENERATION_CONDITIONING,
        "inversion_conditioning": "null=True (unconditional)", "source_sampler_sha256": source_sampler_sha256,
        "checkpoint_hash": checkpoint_hash, "dataset_split_hash": rows[0]["split_sha256"], "exit_code": 0,
        "failure_reason": "", "skip_reason": "", "dtype": "float32",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"E2.3 cache manifest {key}={manifest.get(key)!r} differs from this run's {expected!r}")
    if manifest.get("sample_ids") != [row["sample_id"] for row in rows]:
        raise SystemExit("E2.3 cache sample IDs/order differ from the split")
    expected_config = {"mode": "cache", "noise_level": noise_level, "pipeline": LEGACY_PIPELINE,
                       "guidance_scale": LEGACY_GUIDANCE_SCALE, "limit": 0}
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(f"E2.3 cache config {key}={config.get(key)!r} differs from {expected!r}")
    if Path(str(config.get("split", ""))).name != split_name:
        raise SystemExit(f"E2.3 cache was built from split {config.get('split')!r}, not {split_name}")
    latents = load_latent_cache(cache_dir, rows)  # task, split hash, order and per-latent hashes
    if latents.dtype != np.float32 or latents.shape[1:] != (1, *IMAGE_SHAPE_HW):
        raise SystemExit(f"E2.3 latents have unexpected dtype/shape {latents.dtype} {latents.shape}")
    with (cache_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    if len(x0_arrays) > len(index):
        raise SystemExit("more samples requested than the E2.3 cache holds")
    for position, array in enumerate(x0_arrays):
        if array.shape != (1, *IMAGE_SHAPE_HW):
            raise SystemExit(f"preprocessed x0 has shape {array.shape}, expected (1,{IMAGE_SHAPE_HW[0]},{IMAGE_SHAPE_HW[1]})")
        if index[position]["latent_finite"] != "1":
            raise SystemExit(f"E2.3 cache latent is not finite: {index[position]['sample_id']}")
        if sha256_array(array) != index[position]["x0_sha256"]:
            raise SystemExit(f"x0 hash differs from the E2.3 cache for {index[position]['sample_id']}")
    return {
        "latents": latents,
        "hashes": {
            "latent_cache_run_id": manifest["run_id"],
            "latent_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "latent_cache_config_sha256": sha256_file(cache_dir / "config.json"),
            "latent_cache_index_sha256": sha256_file(cache_dir / "per_sample.csv"),
            "latent_cache_sha256": sha256_file(cache_dir / "latents.npy"),
            "latent_cache_x0_verified": len(x0_arrays),
        },
    }


def build_real_inputs(args):
    from scripts.e1_ddim_runner import create_runtime, load_split, preprocess

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (use --synthetic-selftest for the CPU plumbing check)")
    rows = load_split(args.split)
    checkpoint_hash = sha256_file(args.checkpoint)
    sampler_hash = sha256_file(args.source_root / "guided_diffusion" / "gaussian_diffusion.py")
    selected = rows[:args.limit] if args.limit else rows
    originals = [preprocess(Path(row["local_path"]))[0] for row in selected]  # (1,H,W) as the cache hashed it
    # Validate BEFORE any model is created or any GPU memory is taken.
    cache = validate_latent_cache(
        args.latent_cache, rows, [tensor.numpy() for tensor in originals], noise_level=args.noise_level,
        split_name=args.split.name, source_sampler_sha256=sampler_hash, checkpoint_hash=checkpoint_hash,
    )
    samples = [
        Sample(row["sample_id"], row["patient_id"], row["label"], originals[i].unsqueeze(0),
               torch.from_numpy(np.ascontiguousarray(cache["latents"][i])).unsqueeze(0))
        for i, row in enumerate(selected)
    ]
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    provenance = {
        "dataset_split_hash": rows[0]["split_sha256"], "checkpoint_hash": checkpoint_hash,
        "source_sampler_sha256": sampler_hash, "manifest": cache["hashes"],
    }
    return samples, DiffusionBackend(diffusion, model_fn, args.noise_level, device), provenance


def build_synthetic_inputs(args):
    synthetic_hash = hashlib.sha256(b"q8-synthetic-selftest").hexdigest()
    provenance = {"dataset_split_hash": synthetic_hash, "checkpoint_hash": synthetic_hash,
                  "source_sampler_sha256": synthetic_hash, "manifest": {}}
    return synthetic_samples(args.limit or 4), SyntheticBackend(), provenance


def parse_args(argv=None):
    from scripts.output_guard import add_force_flag
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evidence-role", choices=EVIDENCE_ROLES, required=True)
    parser.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--latent-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="first N split rows; 0 = all")
    parser.add_argument("--storages", nargs="+", default=[MAIN_STORAGE], choices=STORAGES)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--key-mode", choices=KEY_MODES, default=None,
                        help="default: fixture for synthetic_selftest/smoke, random for pilot/formal (fixture refused there)")
    parser.add_argument("--synthetic-selftest", action="store_true",
                        help="CPU plumbing check with a fake pointwise generator; scientifically meaningless")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="Q8_PILOT")
    add_force_flag(parser)
    args = parser.parse_args(argv)
    if args.synthetic_selftest != (args.evidence_role == "synthetic_selftest"):
        raise SystemExit("--synthetic-selftest and --evidence-role synthetic_selftest must be used together")
    if not args.synthetic_selftest and not (args.latent_cache and args.checkpoint):
        raise SystemExit("--latent-cache and --checkpoint are required for real runs")
    if args.limit < 0 or args.warmup < 0:
        raise SystemExit("--limit/--warmup must be non-negative")
    resolve_key_mode(args.evidence_role, args.key_mode)  # role policy fails before any data/model is touched
    return args


def main(argv=None) -> None:
    from scripts.e1_ddim_runner import Tee

    args = parse_args(argv)
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_capture)), redirect_stderr(Tee(sys.stderr, stderr_capture)):
        samples, backend, provenance = (build_synthetic_inputs if args.synthetic_selftest else build_real_inputs)(args)
        execute(args, samples, backend, provenance, stdout_capture, stderr_capture)


if __name__ == "__main__":
    main()
