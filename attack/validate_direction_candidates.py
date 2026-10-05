#!/usr/bin/env python3
"""CPU-only preflight for three candidate thesis directions.

This script does not claim end-to-end model security.  It checks the algebraic
properties that can be tested without a diffusion checkpoint:

1. SOT-WHT: reversibility, Gaussian preservation, magnitude/norm leakage.
2. CDF torus pad: reversibility, Gaussian preservation, key-reuse failure, and
   sensitivity to approximate recovery of the encrypted latent.
3. KCI-style injection: exact cancellation in a toy reversible two-state
   recurrence and failure with a wrong key.

The NumPy RNG below is only for a deterministic experiment.  It is not a KDF or
CSPRNG and must not be copied into the production design.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri


SEED = 1911


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    return float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))


def row_normalize(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(denom, np.finfo(np.float64).tiny)


def gallery_top1(query: np.ndarray, gallery: np.ndarray) -> float:
    q = row_normalize(np.asarray(query, dtype=np.float64))
    g = row_normalize(np.asarray(gallery, dtype=np.float64))
    return float(np.mean(np.argmax(q @ g.T, axis=1) == np.arange(len(q))))


def scalar_gallery_top1(query: np.ndarray, gallery: np.ndarray) -> float:
    distances = np.abs(query[:, None] - gallery[None, :])
    return float(np.mean(np.argmin(distances, axis=1) == np.arange(len(query))))


def fwht(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    n = out.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError("FWHT dimension must be a power of two")
    h = 1
    while h < n:
        blocks = out.reshape(*out.shape[:-1], -1, 2 * h)
        left = blocks[..., :h].copy()
        right = blocks[..., h:].copy()
        blocks[..., :h] = left + right
        blocks[..., h:] = left - right
        h *= 2
    return out


def make_sot_keys(rng: np.random.Generator, d: int, rounds: int = 2):
    return [
        (
            rng.choice(np.array([-1.0, 1.0]), size=d),
            rng.permutation(d),
            rng.choice(np.array([-1.0, 1.0]), size=d),
        )
        for _ in range(rounds)
    ]


def sot_encrypt(z: np.ndarray, keys) -> np.ndarray:
    out = np.asarray(z, dtype=np.float64)
    scale = math.sqrt(out.shape[-1])
    for sign_left, permutation, sign_right in keys:
        out = fwht(out * sign_left) / scale
        out = out[..., permutation] * sign_right
    return out


def sot_decrypt(c: np.ndarray, keys) -> np.ndarray:
    out = np.asarray(c, dtype=np.float64)
    d = out.shape[-1]
    scale = math.sqrt(d)
    for sign_left, permutation, sign_right in reversed(keys):
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(d)
        out = out * sign_right
        out = fwht(out[..., inverse]) / scale
        out = out * sign_left
    return out


def clip_unit_interval(u: np.ndarray) -> np.ndarray:
    low = np.nextafter(np.float64(0.0), np.float64(1.0))
    high = np.nextafter(np.float64(1.0), np.float64(0.0))
    return np.clip(u, low, high)


def cdf_encrypt(z: np.ndarray, pad: np.ndarray) -> np.ndarray:
    uniform = np.mod(ndtr(np.asarray(z, dtype=np.float64)) + pad, 1.0)
    return ndtri(clip_unit_interval(uniform))


def cdf_decrypt(c: np.ndarray, pad: np.ndarray) -> np.ndarray:
    uniform = np.mod(ndtr(np.asarray(c, dtype=np.float64)) - pad, 1.0)
    return ndtri(clip_unit_interval(uniform))


def circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.mod(a - b + 0.5, 1.0) - 0.5)


def validate_sot(d: int, gallery_size: int) -> dict:
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((gallery_size, d)).astype(np.float32)
    keys = make_sot_keys(np.random.default_rng(SEED + 1), d, rounds=2)

    start = time.perf_counter()
    c = sot_encrypt(z, keys)
    elapsed = time.perf_counter() - start
    recovered = sot_decrypt(c, keys)
    recovered32 = recovered.astype(np.float32)

    plain_norm = np.linalg.norm(z.astype(np.float64), axis=1)
    cipher_norm = np.linalg.norm(c, axis=1)
    ks = stats.kstest(c.ravel()[:: max(1, c.size // 100_000)], "norm")

    return {
        "dimension": d,
        "gallery_size": gallery_size,
        "rounds": 2,
        "float32_bit_exact_rate": float(
            np.mean(
                np.all(
                    recovered32.view(np.uint32) == z.view(np.uint32),
                    axis=1,
                )
            )
        ),
        "max_abs_roundtrip_error_float64": float(
            np.max(np.abs(recovered - z.astype(np.float64)))
        ),
        "gaussian_ks_statistic": float(ks.statistic),
        "gaussian_ks_pvalue": float(ks.pvalue),
        "norm_correlation": float(np.corrcoef(plain_norm, cipher_norm)[0, 1]),
        "magnitude_gallery_top1": gallery_top1(np.abs(c), np.abs(z)),
        "norm_only_gallery_top1": scalar_gallery_top1(cipher_norm, plain_norm),
        "encrypt_ms_per_vector_python": 1000.0 * elapsed / gallery_size,
        "interpretation": (
            "SOT removes elementwise magnitude matching but exactly preserves "
            "the L2 norm; the scalar norm can therefore link exact latent pairs."
        ),
    }


def validate_cdf(d: int, gallery_size: int) -> dict:
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((gallery_size, d))
    pads = rng.random((gallery_size, d))

    start = time.perf_counter()
    c = cdf_encrypt(z, pads)
    elapsed = time.perf_counter() - start
    recovered = cdf_decrypt(c, pads)

    plain_norm = np.linalg.norm(z, axis=1)
    cipher_norm = np.linalg.norm(c, axis=1)
    ks = stats.kstest(c.ravel()[:: max(1, c.size // 100_000)], "norm")

    # With an independently sampled one-time pad per value, even a shifted and
    # rescaled input is mapped to a standard-normal marginal.  This statement is
    # over pad randomness; it is not a fixed-key distributional claim.
    non_gaussian_z = 0.25 + 1.13 * rng.standard_normal((gallery_size, d))
    non_gaussian_c = cdf_encrypt(non_gaussian_z, rng.random((gallery_size, d)))
    non_gaussian_probe = non_gaussian_c.ravel()[
        :: max(1, non_gaussian_c.size // 100_000)
    ]
    non_gaussian_ks = stats.kstest(non_gaussian_probe, "norm")

    # Reusing one pad: one known pair reveals that pad, then decrypts another.
    reuse_rng = np.random.default_rng(SEED + 2)
    reused_pad = reuse_rng.random(d)
    known_plain = reuse_rng.standard_normal(d)
    known_cipher = cdf_encrypt(known_plain, reused_pad)
    recovered_pad = np.mod(ndtr(known_cipher) - ndtr(known_plain), 1.0)
    target_plain = reuse_rng.standard_normal(d)
    target_cipher = cdf_encrypt(target_plain, reused_pad)
    target_recovered = cdf_decrypt(target_cipher, recovered_pad)

    # Correct two-time-pad relation is a circular/mod-1 difference.
    plain_delta = np.mod(ndtr(known_plain) - ndtr(target_plain), 1.0)
    cipher_delta = np.mod(ndtr(known_cipher) - ndtr(target_cipher), 1.0)

    robustness = {}
    probe_rng = np.random.default_rng(SEED + 3)
    probe_z = probe_rng.standard_normal(65_536)
    probe_pad = probe_rng.random(65_536)
    probe_c = cdf_encrypt(probe_z, probe_pad)
    for target_cosine in (0.9989, 0.9999, 0.99999, 0.999999):
        sigma = math.sqrt(target_cosine ** -2 - 1.0)
        approximate_c = probe_c + sigma * probe_rng.standard_normal(probe_c.shape)
        approximate_z = cdf_decrypt(approximate_c, probe_pad)
        error = approximate_z - probe_z
        robustness[str(target_cosine)] = {
            "ciphertext_cosine_observed": cosine(probe_c, approximate_c),
            "decrypted_cosine": cosine(probe_z, approximate_z),
            "decrypted_rmse": float(np.sqrt(np.mean(error * error))),
            "fraction_abs_error_gt_1": float(np.mean(np.abs(error) > 1.0)),
            "max_abs_error": float(np.max(np.abs(error))),
        }

    return {
        "dimension": d,
        "gallery_size": gallery_size,
        "max_abs_roundtrip_error_exact_latent": float(
            np.max(np.abs(recovered - z))
        ),
        "median_abs_roundtrip_error_exact_latent": float(
            np.median(np.abs(recovered - z))
        ),
        "gaussian_ks_statistic": float(ks.statistic),
        "gaussian_ks_pvalue": float(ks.pvalue),
        "shifted_input_output_mean_over_fresh_pads": float(
            np.mean(non_gaussian_probe)
        ),
        "shifted_input_output_std_over_fresh_pads": float(np.std(non_gaussian_probe)),
        "shifted_input_output_ks_pvalue_over_fresh_pads": float(
            non_gaussian_ks.pvalue
        ),
        "norm_correlation": float(np.corrcoef(plain_norm, cipher_norm)[0, 1]),
        "magnitude_gallery_top1": gallery_top1(np.abs(c), np.abs(z)),
        "norm_only_gallery_top1": scalar_gallery_top1(cipher_norm, plain_norm),
        "known_pair_reused_pad_max_circular_error": float(
            np.max(circular_error(recovered_pad, reused_pad))
        ),
        "known_pair_decrypts_second_message_cosine": cosine(
            target_plain, target_recovered
        ),
        "two_time_pad_mod1_relation_max_circular_error": float(
            np.max(circular_error(plain_delta, cipher_delta))
        ),
        "approximate_ciphertext_robustness": robustness,
        "encrypt_ms_per_vector_python": 1000.0 * elapsed / gallery_size,
        "interpretation": (
            "The ideal transform removes norm and magnitude linkage, but one "
            "known pair breaks a reused pad and approximate latent recovery can "
            "be strongly amplified by inverse-CDF conditioning and wrap-around."
        ),
    }


def reversible_chain_encrypt(
    x0: np.ndarray,
    x1: np.ndarray,
    deltas: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> list[np.ndarray]:
    states = [x0.astype(np.float64), x1.astype(np.float64)]
    for i in range(1, len(deltas) + 1):
        predicted_noise = np.tanh(states[i])
        states.append(
            a * states[i - 1]
            + b * states[i]
            + c * (predicted_noise + deltas[i - 1])
        )
    return states


def reversible_chain_decrypt(
    x_last_minus_one: np.ndarray,
    x_last: np.ndarray,
    deltas: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    right = x_last.astype(np.float64)
    middle = x_last_minus_one.astype(np.float64)
    for i in range(len(deltas), 0, -1):
        predicted_noise = np.tanh(middle)
        left = (
            right - b * middle - c * (predicted_noise + deltas[i - 1])
        ) / a
        right, middle = middle, left
    return middle


def validate_kci_toy(d: int, steps: int = 20) -> dict:
    rng = np.random.default_rng(SEED)
    x0 = rng.standard_normal(d)
    x1 = rng.standard_normal(d)
    correct_deltas = rng.standard_normal((steps, d))
    wrong_deltas = np.random.default_rng(SEED + 99).standard_normal((steps, d))

    # Coefficients are chosen only to create a numerically stable reversible
    # recurrence.  They are not O-BELM coefficients.
    a, b, c = 0.90, 0.05, 0.15
    states = reversible_chain_encrypt(x0, x1, correct_deltas, a, b, c)
    recovered = reversible_chain_decrypt(
        states[-2], states[-1], correct_deltas, a, b, c
    )
    wrong = reversible_chain_decrypt(
        states[-2], states[-1], wrong_deltas, a, b, c
    )

    return {
        "dimension": d,
        "steps": steps,
        "correct_key_max_abs_error": float(np.max(np.abs(recovered - x0))),
        "correct_key_cosine": cosine(recovered, x0),
        "wrong_key_cosine": cosine(wrong, x0),
        "wrong_key_rmse": float(np.sqrt(np.mean((wrong - x0) ** 2))),
        "scope_warning": (
            "This validates only algebraic key-noise cancellation in a toy "
            "two-state recurrence.  It does not validate O-BELM coefficients, "
            "diffusion-model quality, whitening, or the paper's IND-CPA proof."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--gallery-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preflight/direction_candidates_cpu.json"),
    )
    args = parser.parse_args()

    if args.dimension & (args.dimension - 1):
        raise SystemExit("--dimension must be a power of two")

    report = {
        "schema_version": 1,
        "seed": SEED,
        "environment": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "limitations": [
            "CPU synthetic Gaussian data only; no checkpoint or CXR is used.",
            "NumPy RNG is deterministic test scaffolding, not a production KDF.",
            "Gallery attacks are diagnostics, not comprehensive security proofs.",
            "The approximate-CDF test models inversion error as independent noise.",
        ],
        "sot_wht": validate_sot(args.dimension, args.gallery_size),
        "cdf_torus_pad": validate_cdf(args.dimension, args.gallery_size),
        "kci_reversible_chain_toy": validate_kci_toy(args.dimension),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
