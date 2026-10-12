#!/usr/bin/env python3
"""A3.2 reduced-space brute force: positive control for the 32-bit seed collapse.

A3.1 shows that the predecessor's S0 / S1 key is a function of 32-bit integer
seeds.  This script demonstrates, on a REDUCED seed space (default 2^20), that
a known plaintext/ciphertext latent pair is enough to recover such a seed by
exhaustive search, using the predecessor's own key generators, and then
extrapolates the measured throughput analytically to the full 2^32.

Experiments (all CPU, single torch thread, seed 1911):
  E1  S0 naive:   d = 4096 key (1,64,64), synthetic N(0,1) latent; regenerate
                  each candidate key with RademacherKey._generate_key and test
                  k (.) z == z_enc (equivalently k == sign(z_enc * z) for z != 0).
  E2  S0 full d:  d = 65536 key (1,256,256), real cached E2.3 latent #0; the
                  candidate is first screened on the first 64 key coordinates
                  (valid because the CPU Bernoulli stream is sequential, see
                  prefix_property_check), survivors are verified on all 65536
                  coordinates with the exact oracle k (.) z == z_enc.
  E3  negative:   same as E2 but the ciphertext was made with a seed OUTSIDE the
                  searched range; the search must find nothing.
  E4  S1 split:   S1 in password-mode structure (independent sign seed and
                  permutation seed, each in a 2^18 range, d = 4096).  Known pair
                  -> recover (k, pi) exactly from magnitude ordering and signs,
                  then search the two seeds separately: ~2^19 candidates instead
                  of 2^36 for a joint search.

Everything under "results" is deterministic.  Wall times, candidates/second and
the 2^32 extrapolation depend on the machine and live under
"timing_nondeterministic".

    scripts/run_cfg_ddim.sh python -m scripts.a3_bruteforce
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# CPU only: hide every GPU before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from scripts.output_guard import add_force_flag, refuse_overwrite
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.env_guard import enforce_active_prefix  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    from env_guard import enforce_active_prefix  # type: ignore  # noqa: E402

enforce_active_prefix({"numpy": np, "torch": torch})
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "A3.2 must run on CPU only"

SOURCE_ROOT = ROOT / "past/SourceCode"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from guided_diffusion.anonymization import (  # noqa: E402
    DiffusionAnonymizer,
    PermutationKey,
    RademacherKey,
    SignedPermutationKey,
)

SEED = 1911
TASK_ID = "A3.2"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.a3_bruteforce"
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/A3.2.json"
ANON_PATH = SOURCE_ROOT / "guided_diffusion/anonymization.py"
LATENT_CACHE = ROOT / "artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a/latents.npy"
REAL_LATENT_INDEX = 0

REDUCED_BITS = 20
RANGE = 2**REDUCED_BITS
S1_BITS = 18
S1_RANGE = 2**S1_BITS
FULL_BITS = 32
SMALL_SHAPE = (1, 64, 64)  # d = 4096
FULL_SHAPE = (1, 256, 256)  # d = 65536
SCREEN_M = 64  # screened key prefix length for E2/E3
COA_BITS = 14  # ciphertext-only reduced seed space (2^14), matching the auditor's AF-027 protocol
COA_RANGE = 2**COA_BITS
COA_REAL_INDEX_S0 = 0  # real latent used for the S0 ciphertext-only attack
COA_REAL_INDEX_S1 = 1  # independent real latent for the S1 ciphertext-only attack


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def human(seconds: float) -> dict:
    return {"seconds": seconds, "hours": seconds / 3600.0, "days": seconds / 86400.0}


def encrypt(latent_shape: tuple[int, ...], z: torch.Tensor, key, use_permutation: bool) -> torch.Tensor:
    """Ciphertext through the predecessor's anonymizer (mask 'full', margin 0)."""
    anonymizer = DiffusionAnonymizer(
        latent_shape=latent_shape, mask_type="full", margin=0, use_permutation=use_permutation
    )
    return anonymizer.anonymize_latent(z, key)


def streams() -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(8)]


# ---------------------------------------------------------------------------
# E1 / E2 / E3 : S0 searches
# ---------------------------------------------------------------------------
def naive_s0_search(shape, z, z_enc, space: int) -> dict:
    """Regenerate every candidate key with the predecessor's code and compare with k = sign(z_enc * z)."""
    d = int(np.prod(shape))
    target = torch.sign(z * z_enc)
    assert bool((target != 0).all()), "z and z_enc must be non-zero for sign extraction"
    wrong_max = 0
    wrong_sum = 0
    found = None
    tested = 0
    t0 = time.perf_counter()
    for s in range(space):
        key = RademacherKey._generate_key(shape, s)
        tested += 1
        agree = int((key == target).sum())
        if agree == d:
            # exact oracle on the original equation z_enc == k (.) z
            if torch.equal(key * z, z_enc):
                found = s
                break
        wrong_sum += agree
        wrong_max = max(wrong_max, agree)
    wall = time.perf_counter() - t0
    wrong_count = tested - (1 if found is not None else 0)
    return {
        "recovered_seed": found, "candidates_tested": tested, "wrong_candidates": wrong_count,
        "max_sign_agreement_of_wrong_candidates": wrong_max,
        "mean_sign_agreement_of_wrong_candidates": (wrong_sum / wrong_count) if wrong_count else None,
        "d": d, "wall": wall,
    }


def screened_s0_search(shape, z, z_enc, space: int, m: int = SCREEN_M) -> dict:
    """Screen on the first m key coordinates, verify survivors with the exact full-d oracle."""
    target_prefix = (torch.sign(z * z_enc).flatten()[:m] > 0).float()
    assert bool(((z * z_enc).flatten()[:m] != 0).all())
    half = torch.full((m,), 0.5)
    generator = torch.Generator()
    survivors = []
    false_survivors = 0
    found = None
    tested = 0
    t0 = time.perf_counter()
    for s in range(space):
        generator.manual_seed(s)
        tested += 1
        if torch.equal(torch.bernoulli(half, generator=generator), target_prefix):
            survivors.append(s)
            key = RademacherKey._generate_key(shape, s)
            if torch.equal(key * z, z_enc):
                found = s
                break
            false_survivors += 1
    wall = time.perf_counter() - t0
    return {
        "recovered_seed": found, "candidates_tested": tested, "screen_prefix_coordinates": m,
        "screen_survivors": survivors, "screen_false_survivors": false_survivors,
        "d": int(np.prod(shape)), "wall": wall,
    }


def prefix_property_check(m: int = SCREEN_M, n_seeds: int = 64) -> dict:
    """The first m entries of the d=65536 Bernoulli key equal a stand-alone m-entry draw (needed by E2/E3)."""
    seeds = [int(x) for x in np.linspace(0, 2**32 - 1, n_seeds)]
    mismatches = 0
    for s in seeds:
        full = RademacherKey._generate_key(FULL_SHAPE, s).flatten()
        generator = torch.Generator()
        generator.manual_seed(s)
        prefix = 2 * torch.bernoulli(torch.full((m,), 0.5), generator=generator) - 1
        mismatches += int(not torch.equal(full[:m], prefix))
    # the key itself must not depend on the thread count
    thread_mismatches = 0
    for s in seeds[:8]:
        torch.set_num_threads(1)
        a = RademacherKey._generate_key(FULL_SHAPE, s)
        torch.set_num_threads(8)
        b = RademacherKey._generate_key(FULL_SHAPE, s)
        torch.set_num_threads(1)
        thread_mismatches += int(not torch.equal(a, b))
    if mismatches or thread_mismatches:
        raise SystemExit("prefix property or thread independence failed; screened search would be invalid")
    return {
        "prefix_length": m, "seeds_checked": n_seeds, "seeds": f"np.linspace(0, 2^32-1, {n_seeds}) as int",
        "prefix_mismatches": mismatches, "thread_independence_seeds_checked": 8,
        "thread_independence_mismatches": thread_mismatches,
    }


# ---------------------------------------------------------------------------
# E4 : S1 divide and conquer
# ---------------------------------------------------------------------------
def s1_kpa_recover(z: np.ndarray, z_enc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover (k, pi) from a known pair: z_enc[i] = k[pi[i]] * z[pi[i]] (anonymization.py:392-397)."""
    zf, ef = z.reshape(-1), z_enc.reshape(-1)
    assert np.unique(np.abs(zf)).size == zf.size, "tied magnitudes: permutation not uniquely recoverable"
    order_z = np.argsort(np.abs(zf), kind="stable")
    order_e = np.argsort(np.abs(ef), kind="stable")
    perm = np.empty(zf.size, dtype=np.int64)
    perm[order_e] = order_z
    sign_key = np.empty(zf.size, dtype=np.float32)
    sign_key[perm] = np.sign(ef * zf[perm])
    return sign_key, perm


def s1_split_search(z: torch.Tensor, secret_sign_seed: int, secret_perm_seed: int, z2: torch.Tensor) -> dict:
    d = int(z.numel())
    key = SignedPermutationKey(
        shape=SMALL_SHAPE,
        rademacher_key=RademacherKey(shape=SMALL_SHAPE, seed=secret_sign_seed),
        permutation_key=PermutationKey(size=d, seed=secret_perm_seed),
    )
    z_enc = encrypt(SMALL_SHAPE, z, key, use_permutation=True)
    sign_rec, perm_rec = s1_kpa_recover(z.numpy(), z_enc.numpy())
    rebuilt = (torch.from_numpy(sign_rec).reshape(-1) * z.reshape(-1))[torch.from_numpy(perm_rec)]
    kpa_exact = bool(torch.equal(rebuilt, z_enc.reshape(-1)))
    sign_target = torch.from_numpy(sign_rec).reshape(SMALL_SHAPE)
    perm_target = torch.from_numpy(perm_rec)

    t0 = time.perf_counter()
    sign_found, sign_tested = None, 0
    for s in range(S1_RANGE):
        sign_tested += 1
        if torch.equal(RademacherKey._generate_key(SMALL_SHAPE, s), sign_target):
            sign_found = s
            break
    wall_sign = time.perf_counter() - t0

    t0 = time.perf_counter()
    perm_found, perm_tested = None, 0
    for t in range(S1_RANGE):
        perm_tested += 1
        if torch.equal(PermutationKey._generate_permutation(d, t), perm_target):
            perm_found = t
            break
    wall_perm = time.perf_counter() - t0

    generalises = False
    if sign_found is not None and perm_found is not None:
        key_hat = SignedPermutationKey(
            shape=SMALL_SHAPE,
            rademacher_key=RademacherKey(shape=SMALL_SHAPE, seed=sign_found),
            permutation_key=PermutationKey(size=d, seed=perm_found),
        )
        generalises = bool(torch.equal(encrypt(SMALL_SHAPE, z2, key, True), encrypt(SMALL_SHAPE, z2, key_hat, True)))
    return {
        "d": d, "kpa_recovers_sign_and_permutation_exactly": kpa_exact,
        "recovered_sign_seed": sign_found, "sign_candidates_tested": sign_tested,
        "recovered_permutation_seed": perm_found, "permutation_candidates_tested": perm_tested,
        "total_candidates_tested": sign_tested + perm_tested,
        "joint_search_space_candidates": S1_RANGE * S1_RANGE,
        "recovered_key_encrypts_fresh_latent_identically": generalises,
        "wall_sign": wall_sign, "wall_perm": wall_perm,
    }


# ---------------------------------------------------------------------------
# E5/E6 : ciphertext-only attacks (AF-027). No plaintext is used.
# ---------------------------------------------------------------------------
def s0_coa_score(c_flat: np.ndarray, candidate_key_flat: np.ndarray) -> float:
    """Normalized correlation of a candidate sign key with one ciphertext.

    c = k (.) z with z_i having a positive mean (noise level 500, x0 in [0,1]).
    score(k') = sum_i k'_i c_i / (rms(c) * sqrt(d)).  For k' = k this is
    sqrt(d) * mean(z) / rms(z); for a random k' it is distributed N(0, 1), so the
    raw score is already in units of sigma against the wrong-key distribution.
    """
    d = c_flat.size
    rms = float(np.sqrt(np.mean(c_flat * c_flat)))
    return float(candidate_key_flat @ c_flat) / (rms * np.sqrt(d))


def _adjacent_corr_abs(flat: np.ndarray, side: int = 256) -> float:
    """Mean horizontal+vertical adjacent-pixel correlation of |flat| as a side x side image."""
    image = np.abs(flat).reshape(side, side)
    centered = image - image.mean()
    denom = float((centered * centered).sum())
    if denom == 0:
        return 0.0
    horizontal = float((centered[:, 1:] * centered[:, :-1]).sum())
    vertical = float((centered[1:, :] * centered[:-1, :]).sum())
    return (horizontal + vertical) / (2.0 * denom)


def s0_coa_search(shape, z: torch.Tensor, secret_seed: int, space: int) -> dict:
    """Plant a secret sign seed, then score every candidate against ONE ciphertext."""
    key = RademacherKey(shape=shape, seed=secret_seed)
    c_flat = encrypt(shape, z, key, False).numpy().reshape(-1).astype(np.float64)
    correct = s0_coa_score(c_flat, key.key.numpy().reshape(-1).astype(np.float64))
    t0 = time.perf_counter()
    scores = np.empty(space, dtype=np.float64)
    best_seed, best_score = -1, -np.inf
    for seed in range(space):
        score = s0_coa_score(c_flat, RademacherKey._generate_key(shape, seed).numpy().reshape(-1).astype(np.float64))
        scores[seed] = score
        if score > best_score:
            best_score, best_seed = score, seed
    wall = time.perf_counter() - t0
    wrong = np.delete(scores, secret_seed) if secret_seed < space else scores
    return {
        "d": int(z.numel()), "secret_seed": secret_seed, "secret_in_range": secret_seed < space,
        "correct_score_sigma": correct, "ranked_first": best_seed == secret_seed,
        "best_seed": best_seed, "best_score_sigma": best_score,
        "wrong_max_sigma": float(wrong.max()), "wrong_mean": float(wrong.mean()), "wrong_std": float(wrong.std()),
        "candidates_tested": space, "wall": wall,
    }


def s1_coa_search(shape, z: torch.Tensor, secret_sign_seed: int, secret_perm_seed: int, space: int) -> dict:
    """Two-stage ciphertext-only attack on S1 from ONE ciphertext (AF-027/AUD-20261011-02)."""
    d = int(z.numel())
    side = int(round(d**0.5))
    key = SignedPermutationKey(
        shape=shape,
        rademacher_key=RademacherKey(shape=shape, seed=secret_sign_seed),
        permutation_key=PermutationKey(size=d, seed=secret_perm_seed),
    )
    c_flat = encrypt(shape, z, key, True).numpy().reshape(-1).astype(np.float64)

    # Stage 1: the correct inverse permutation restores the spatial structure of |.|.
    t0 = time.perf_counter()
    perm_scores = np.empty(space, dtype=np.float64)
    best_perm, best_perm_corr = -1, -np.inf
    for seed in range(space):
        inverse = PermutationKey(size=d, seed=seed).inverse_permutation.numpy()
        corr = _adjacent_corr_abs(c_flat[inverse], side)
        perm_scores[seed] = corr
        if corr > best_perm_corr:
            best_perm_corr, best_perm = corr, seed
    wall_perm = time.perf_counter() - t0
    perm_wrong = np.delete(perm_scores, secret_perm_seed) if secret_perm_seed < space else perm_scores
    perm_sigma = (perm_scores[secret_perm_seed] - perm_wrong.mean()) / perm_wrong.std() if secret_perm_seed < space else float("nan")

    # Stage 2: with the permutation known, undo it and run the S0 score for the sign seed.
    inverse_correct = PermutationKey(size=d, seed=secret_perm_seed).inverse_permutation.numpy()
    u_flat = c_flat[inverse_correct]  # = k (.) z
    t0 = time.perf_counter()
    sign_scores = np.empty(space, dtype=np.float64)
    best_sign, best_sign_score = -1, -np.inf
    for seed in range(space):
        score = s0_coa_score(u_flat, RademacherKey._generate_key(shape, seed).numpy().reshape(-1).astype(np.float64))
        sign_scores[seed] = score
        if score > best_sign_score:
            best_sign_score, best_sign = score, seed
    wall_sign = time.perf_counter() - t0
    sign_wrong = np.delete(sign_scores, secret_sign_seed) if secret_sign_seed < space else sign_scores
    return {
        "d": d, "n_ciphertexts": 1,
        "permutation_secret_seed": secret_perm_seed, "permutation_ranked_first": best_perm == secret_perm_seed,
        "permutation_correct_sigma": float(perm_sigma), "permutation_wrong_max_corr": float(perm_wrong.max()),
        "permutation_correct_corr": float(perm_scores[secret_perm_seed]) if secret_perm_seed < space else float("nan"),
        "sign_secret_seed": secret_sign_seed, "sign_ranked_first": best_sign == secret_sign_seed,
        "sign_correct_sigma": float(sign_scores[secret_sign_seed]) if secret_sign_seed < space else float("nan"),
        "sign_wrong_max_sigma": float(sign_wrong.max()),
        "candidates_tested": 2 * space, "wall_perm": wall_perm, "wall_sign": wall_sign,
    }


def s1_coa_negative_control(shape, z: torch.Tensor, secret_perm_seed_out: int, space: int) -> dict:
    """Permutation stage when the ciphertext's permutation seed is OUTSIDE the candidate range."""
    d = int(z.numel())
    side = int(round(d**0.5))
    key = SignedPermutationKey(
        shape=shape,
        rademacher_key=RademacherKey(shape=shape, seed=0),
        permutation_key=PermutationKey(size=d, seed=secret_perm_seed_out),
    )
    c_flat = encrypt(shape, z, key, True).numpy().reshape(-1).astype(np.float64)
    corrs = np.array([_adjacent_corr_abs(c_flat[PermutationKey(size=d, seed=seed).inverse_permutation.numpy()], side)
                      for seed in range(space)])
    return {"secret_perm_seed_outside_range": secret_perm_seed_out, "candidates_tested": space,
            "max_sigma_over_candidates": float((corrs.max() - corrs.mean()) / corrs.std())}


def coa_all_images(latents_all: np.ndarray, null_perm_seeds: int = 64) -> dict:
    """Correct-key ciphertext-only score for every security_v1 latent (AF-032).

    No seed search: the correct score depends only on the latent. S0 / S1-sign
    score = sqrt(d) * mean(z) / rms(z). S1 permutation score = adjacent-pixel
    correlation of |z| standardized against a fixed null set of wrong inverse
    permutations.  Reports the per-image distribution and how many fall below
    the full-scale threshold (expected max of 2^32 wrong candidates, ~6.66 sigma).
    """
    n, _, side, _ = latents_all.shape
    d = side * side
    threshold = float((2 * np.log(2**FULL_BITS)) ** 0.5)
    null_inverses = [PermutationKey(size=d, seed=10**6 + seed).inverse_permutation.numpy()
                     for seed in range(null_perm_seeds)]

    sign_sigma = np.empty(n)
    perm_sigma = np.empty(n)
    for i in range(n):
        flat = latents_all[i].reshape(-1).astype(np.float64)
        rms = float(np.sqrt(np.mean(flat * flat)))
        sign_sigma[i] = np.sqrt(d) * float(flat.mean()) / rms  # correct key: sum z / (rms sqrt d)
        correct_corr = _adjacent_corr_abs(flat, side)
        null = np.array([_adjacent_corr_abs(flat[inv], side) for inv in null_inverses])
        perm_sigma[i] = (correct_corr - null.mean()) / null.std()

    def dist(values: np.ndarray) -> dict:
        return {"min": float(values.min()), "p05": float(np.percentile(values, 5)),
                "median": float(np.median(values)), "max": float(values.max()),
                "n_below_full_scale_threshold": int(np.sum(values < threshold))}

    return {
        "n_images": n, "d": d, "null_permutation_seeds": null_perm_seeds,
        "full_scale_threshold_sigma": threshold,
        "s0_and_s1_sign_correct_sigma": dist(sign_sigma),
        "s1_permutation_correct_sigma": dist(perm_sigma),
        "s1_permutation_weakest_image_index": int(np.argmin(perm_sigma)),
        "note": ("[fact] S0 and the S1 sign stage exceed the threshold for every image; the S1 permutation stage "
                 "exceeds it for all but the weakest image. The predecessor reuses one key across all images, so a "
                 "single ciphertext (the strongest one) suffices at deployment level."),
    }


# ---------------------------------------------------------------------------
# throughput benchmarks (for the extrapolation)
# ---------------------------------------------------------------------------
def bench(fn, n: int, warmup: int = 10) -> float:
    for i in range(warmup):
        fn(i)
    t0 = time.perf_counter()
    for i in range(n):
        fn(warmup + i)
    return n / (time.perf_counter() - t0)


def run_benchmarks() -> dict:
    half = torch.full((SCREEN_M,), 0.5)
    generator = torch.Generator()

    def screen(i: int) -> None:
        generator.manual_seed(i)
        torch.bernoulli(half, generator=generator)

    return {
        "s0_full_key_d65536_candidates_per_s": bench(lambda i: RademacherKey._generate_key(FULL_SHAPE, i), 400),
        "s0_prefix_screen_m64_candidates_per_s": bench(screen, 100_000),
        "s0_full_key_d4096_candidates_per_s": bench(lambda i: RademacherKey._generate_key(SMALL_SHAPE, i), 20_000),
        "s1_permutation_d65536_candidates_per_s": bench(lambda i: PermutationKey._generate_permutation(65536, i), 100),
        "s1_permutation_d4096_candidates_per_s": bench(lambda i: PermutationKey._generate_permutation(4096, i), 20_000),
    }


def extrapolate(rate: float) -> dict:
    full = 2.0**FULL_BITS
    return {
        "candidates_per_s_cpu_single_thread": rate,
        "unit": "CPU single thread (torch.set_num_threads(1)); not GPU",
        "worst_case_2_32": human(full / rate),
        "expected_half_2_31": human(full / 2 / rate),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=(__doc__ or '').strip().splitlines()[0])
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([OUTPUT], args.force)
    ss = streams()
    seed_rng = ss[0]
    secret_e1 = int(seed_rng.integers(0, RANGE))
    secret_e2 = int(seed_rng.integers(0, RANGE))
    secret_e3 = int(seed_rng.integers(RANGE, 2**FULL_BITS))
    secret_sign = int(seed_rng.integers(0, S1_RANGE))
    secret_perm = int(seed_rng.integers(0, S1_RANGE))

    z_small = torch.from_numpy(ss[1].standard_normal(SMALL_SHAPE).astype(np.float32))
    z_s1 = torch.from_numpy(ss[2].standard_normal(SMALL_SHAPE).astype(np.float32))
    z_s1_fresh = torch.from_numpy(ss[3].standard_normal(SMALL_SHAPE).astype(np.float32))
    latents = np.load(LATENT_CACHE, mmap_mode="r")
    assert latents.shape == (200, 1, 256, 256) and latents.dtype == np.float32
    z_real = torch.from_numpy(np.array(latents[REAL_LATENT_INDEX], dtype=np.float32))

    prefix_check = prefix_property_check()

    # E1
    key_e1 = RademacherKey(shape=SMALL_SHAPE, seed=secret_e1)
    e1 = naive_s0_search(SMALL_SHAPE, z_small, encrypt(SMALL_SHAPE, z_small, key_e1, False), RANGE)
    # E2
    key_e2 = RademacherKey(shape=FULL_SHAPE, seed=secret_e2)
    e2 = screened_s0_search(FULL_SHAPE, z_real, encrypt(FULL_SHAPE, z_real, key_e2, False), RANGE)
    # E3 (negative control)
    key_e3 = RademacherKey(shape=FULL_SHAPE, seed=secret_e3)
    e3 = screened_s0_search(FULL_SHAPE, z_real, encrypt(FULL_SHAPE, z_real, key_e3, False), RANGE)
    # E4
    e4 = s1_split_search(z_s1, secret_sign, secret_perm, z_s1_fresh)
    # E5 / E6 : ciphertext-only attacks (AF-027), independent real latents, single ciphertext
    secret_e5 = int(seed_rng.integers(0, COA_RANGE))
    secret_e6_sign = int(seed_rng.integers(0, COA_RANGE))
    secret_e6_perm = int(seed_rng.integers(0, COA_RANGE))
    secret_e6_neg = int(seed_rng.integers(COA_RANGE, 2**FULL_BITS))
    z_real_s1 = torch.from_numpy(np.array(latents[COA_REAL_INDEX_S1], dtype=np.float32))
    e5 = s0_coa_search(FULL_SHAPE, z_real, secret_e5, COA_RANGE)
    e6 = s1_coa_search(FULL_SHAPE, z_real_s1, secret_e6_sign, secret_e6_perm, COA_RANGE)
    e6_neg = s1_coa_negative_control(FULL_SHAPE, z_real_s1, secret_e6_neg, COA_RANGE)
    e7 = coa_all_images(np.array(latents))

    benchmarks = run_benchmarks()

    # legacy seeds (from A3.1 V1): where an ascending enumeration of 2^32 seeds would reach them
    legacy_sign = int.from_bytes(hashlib.sha256(b"Ki@13579").digest()[:4], "big")
    legacy_perm = int.from_bytes(hashlib.sha256(b"Ki@13579_permutation").digest()[:4], "big")

    checks = {
        "E1_recovered": e1["recovered_seed"] == secret_e1,
        "E2_recovered": e2["recovered_seed"] == secret_e2,
        "E3_nothing_found": e3["recovered_seed"] is None and not e3["screen_survivors"],
        "E4_recovered": e4["recovered_sign_seed"] == secret_sign and e4["recovered_permutation_seed"] == secret_perm,
        "E4_kpa_exact": e4["kpa_recovers_sign_and_permutation_exactly"],
        "E4_generalises": e4["recovered_key_encrypts_fresh_latent_identically"],
        "E2_no_false_survivors": e2["screen_false_survivors"] == 0,
        "E5_coa_ranked_first": e5["ranked_first"] and e5["correct_score_sigma"] > e5["wrong_max_sigma"],
        "E6_coa_permutation_first": e6["permutation_ranked_first"],
        "E6_coa_sign_first": e6["sign_ranked_first"],
        "E7_s0_all_images_above_threshold": e7["s0_and_s1_sign_correct_sigma"]["n_below_full_scale_threshold"] == 0,
        "E7_s1_permutation_at_most_one_below": e7["s1_permutation_correct_sigma"]["n_below_full_scale_threshold"] <= 1,
    }
    if not all(checks.values()):
        raise SystemExit(f"positive/negative control failed: {checks}")

    results = {
        "prefix_property_check": prefix_check,
        "e1_s0_naive_d4096": {
            "description": "S0, brute force over seeds [0, 2^20), full 4096-dim candidate keys regenerated with RademacherKey._generate_key",
            "reduced_space_bits": REDUCED_BITS, "search_range": [0, RANGE],
            "key_shape": list(SMALL_SHAPE), "d": e1["d"],
            "latent": "synthetic N(0,1) float32, numpy default_rng(SeedSequence(1911).spawn(8)[1])",
            "oracle": "known pair (z, z_enc); candidate k accepted iff k (.) z == z_enc exactly (equivalent to k == sign(z_enc*z) because z has no zero)",
            "secret_seed": secret_e1, "secret_seed_draw": "default_rng(SeedSequence(1911).spawn(8)[0]).integers(0, 2^20), draw 1",
            "secret_seed_recovered": e1["recovered_seed"] == secret_e1, "recovered_seed": e1["recovered_seed"],
            "candidates_tested": e1["candidates_tested"], "wrong_candidates": e1["wrong_candidates"],
            "max_sign_agreement_of_wrong_candidates": e1["max_sign_agreement_of_wrong_candidates"],
            "mean_sign_agreement_of_wrong_candidates": e1["mean_sign_agreement_of_wrong_candidates"],
            "expected_agreement_of_random_key": e1["d"] / 2,
        },
        "e2_s0_full_d65536_screened": {
            "description": "S0 at the real latent size d = 65536, seeds [0, 2^20); 64-coordinate prefix screen, survivors verified on all coordinates",
            "reduced_space_bits": REDUCED_BITS, "search_range": [0, RANGE],
            "key_shape": list(FULL_SHAPE), "d": e2["d"],
            "latent": f"real E2.3 inversion latent, index {REAL_LATENT_INDEX} of latents.npy",
            "oracle": "screen: first 64 key coordinates == sign(z_enc*z); verify: k (.) z == z_enc on all 65536 coordinates",
            "secret_seed": secret_e2, "secret_seed_draw": "same rng as E1, draw 2",
            "secret_seed_recovered": e2["recovered_seed"] == secret_e2, "recovered_seed": e2["recovered_seed"],
            "candidates_tested": e2["candidates_tested"], "screen_prefix_coordinates": e2["screen_prefix_coordinates"],
            "screen_survivors": e2["screen_survivors"], "screen_false_survivors": e2["screen_false_survivors"],
            "analytic_false_survivor_rate": ("[fact] a 64-coordinate prefix screen matches a random key with probability 2^-64; "
                                             "over the full 2^32 seed space the expected false survivors are 2^32 * 2^-64 = 2^-32"),
        },
        "e3_negative_control": {
            "description": "same search as E2 over the whole [0, 2^20) range, but z_enc was produced with a seed outside the range; nothing may be found",
            "reduced_space_bits": REDUCED_BITS, "search_range": [0, RANGE], "d": e3["d"],
            "secret_seed_outside_range": secret_e3, "secret_seed_draw": "same rng as E1, draw 3, integers(2^20, 2^32)",
            "recovered_seed": e3["recovered_seed"], "candidates_tested": e3["candidates_tested"],
            "screen_survivors": e3["screen_survivors"], "false_positives": len(e3["screen_survivors"]),
        },
        "e4_s1_split_search": {
            "description": ("S1 with independent sign seed and permutation seed (password-mode structure), each in [0, 2^18), d = 4096; "
                            "known pair -> exact (k, pi) via magnitude ordering and signs, then the two seeds are searched separately"),
            "reduced_space_bits_each_seed": S1_BITS, "search_range_each": [0, S1_RANGE],
            "joint_search_bits_without_splitting": 2 * S1_BITS,
            "key_shape": list(SMALL_SHAPE),
            "latent": "synthetic N(0,1) float32 with pairwise distinct magnitudes (asserted)",
            "secret_sign_seed": secret_sign, "secret_permutation_seed": secret_perm,
            "secret_seed_draw": "same rng as E1, draws 4 and 5",
            **{k: v for k, v in e4.items() if not k.startswith("wall_")},
            "worst_case_candidates_separable": 2 * S1_RANGE,
            "joint_search_candidates_without_splitting": S1_RANGE * S1_RANGE,
            "note": ("[fact] total_candidates_tested reflects early stopping at the planted seeds; the attacker's work is "
                     "2 * 2^18 (separable) versus 2^36 (joint), not log2(total_tested)"),
        },
        "e5_s0_ciphertext_only_d65536": {
            "description": ("S0 ciphertext-only: ONE exact ciphertext latent, no plaintext. Score each candidate sign key by "
                            "sum_i k'_i c_i / (rms(c) sqrt(d)); the latent's positive mean makes the correct key an outlier."),
            "reduced_space_bits": COA_BITS, "search_range": [0, COA_RANGE],
            "latent": f"real E2.3 inversion latent, index {REAL_LATENT_INDEX}",
            "secret_seed": secret_e5, "secret_seed_recovered": e5["ranked_first"], "recovered_seed": e5["best_seed"],
            "correct_score_sigma": e5["correct_score_sigma"], "wrong_candidate_max_sigma": e5["wrong_max_sigma"],
            "wrong_candidate_mean": e5["wrong_mean"], "wrong_candidate_std": e5["wrong_std"],
            "candidates_tested": e5["candidates_tested"],
            "note": "[fact] a single ciphertext latent is enough; score is already in sigma units (wrong keys ~ N(0,1))",
        },
        "e6_s1_ciphertext_only_d65536": {
            "description": ("S1 ciphertext-only from ONE ciphertext: stage 1 finds the permutation seed by the adjacent-pixel "
                            "correlation of the inverse-permuted |c| (restored spatial structure); stage 2 finds the sign seed "
                            "with the S0 score on the de-permuted latent. No plaintext, no key reuse."),
            "reduced_space_bits_each_stage": COA_BITS, "search_range_each": [0, COA_RANGE], "n_ciphertexts": 1,
            "latent": f"real E2.3 inversion latent, index {COA_REAL_INDEX_S1}",
            "permutation_secret_seed": secret_e6_perm, "permutation_ranked_first": e6["permutation_ranked_first"],
            "permutation_correct_sigma": e6["permutation_correct_sigma"], "permutation_wrong_max_corr": e6["permutation_wrong_max_corr"],
            "sign_secret_seed": secret_e6_sign, "sign_ranked_first": e6["sign_ranked_first"],
            "sign_correct_sigma": e6["sign_correct_sigma"], "sign_wrong_max_sigma": e6["sign_wrong_max_sigma"],
            "candidates_tested": e6["candidates_tested"],
            "negative_control": e6_neg,
            "note": "[fact] a single S1 ciphertext suffices; no key reuse required (stronger than the >=10 ciphertexts earlier estimated)",
        },
        "e7_ciphertext_only_all_images": {
            "description": ("AF-032: correct-key ciphertext-only score over all 200 security_v1 latents, no search. "
                            "Scope check for the 'single ciphertext suffices' claim."),
            **e7,
        },
        "checks": checks,
        "legacy_seed_positions": {
            "note": ("[inference] positions of the legacy password-derived seeds (A3.1 V1) in an ascending enumeration of all 2^32 seeds; "
                     "an attacker without the password would not know the position and expects to test 2^31 candidates on average"),
            "sign_seed": legacy_sign, "sign_seed_ascending_candidates": legacy_sign + 1,
            "permutation_seed": legacy_perm, "permutation_seed_ascending_candidates": legacy_perm + 1,
        },
    }

    timing = {
        "note": "machine dependent; excluded from determinism comparison",
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(), "processor": platform.processor(),
        "wall_time_s": {
            "e1_s0_naive_d4096": e1["wall"], "e2_s0_screened_d65536": e2["wall"], "e3_negative_screened_d65536": e3["wall"],
            "e4_sign_search_d4096": s1_wall(e4, "sign"), "e4_permutation_search_d4096": s1_wall(e4, "perm"),
        },
        "candidates_per_s_in_controls": {
            "e1_s0_naive_d4096": e1["candidates_tested"] / e1["wall"],
            "e2_s0_screened_d65536": e2["candidates_tested"] / e2["wall"],
            "e3_negative_screened_d65536": e3["candidates_tested"] / e3["wall"],
        },
        "benchmarks": benchmarks,
        "extrapolation_to_2_32": {
            "assumptions": ("[assumption] linear in the number of candidates, single thread, no per-candidate oracle cost beyond "
                            "key regeneration/screening; scales ~linearly with cores only if candidates are split across workers"),
            "s0_naive_full_key_d65536": extrapolate(benchmarks["s0_full_key_d65536_candidates_per_s"]),
            "s0_prefix_screen_m64_d65536": extrapolate(benchmarks["s0_prefix_screen_m64_candidates_per_s"]),
            "s1_permutation_naive_randperm_d65536": extrapolate(benchmarks["s1_permutation_d65536_candidates_per_s"]),
            "legacy_sign_seed_ascending_with_screen": human(
                (legacy_sign + 1) / benchmarks["s0_prefix_screen_m64_candidates_per_s"]
            ),
        },
        "ciphertext_only_extrapolation": {
            "assumptions": ("[assumption] wrong candidates score ~ N(0,1); the expected maximum over M candidates is about "
                            "sqrt(2 ln M). For M = 2^32 that is ~6.66 sigma, far below the correct key's score, so the correct "
                            "key stays rank 1 at full scale. CPU single thread."),
            "expected_max_wrong_sigma_2_14": float((2 * np.log(COA_RANGE)) ** 0.5),
            "expected_max_wrong_sigma_2_32": float((2 * np.log(2**FULL_BITS)) ** 0.5),
            "s0_coa_score_per_s": extrapolate(benchmarks["s0_full_key_d65536_candidates_per_s"]),
            "s1_coa_permutation_score_per_s": extrapolate(benchmarks["s1_permutation_d65536_candidates_per_s"]),
        },
    }

    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "created_at": created_at,
        "command": COMMAND,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "seed": SEED,
        "seed_streams": "np.random.default_rng for each of np.random.SeedSequence(1911).spawn(8); torch seeds are the candidate seeds themselves",
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cpu (CUDA_VISIBLE_DEVICES='' set before importing torch)",
        "inputs": {
            "predecessor_source": "past/SourceCode/guided_diffusion/anonymization.py",
            "predecessor_source_sha256": sha256_file(ANON_PATH),
            "predecessor_functions_used": [
                "RademacherKey._generate_key", "PermutationKey._generate_permutation", "RademacherKey", "PermutationKey",
                "SignedPermutationKey", "DiffusionAnonymizer.anonymize_latent",
            ],
            "real_latent_cache": str(LATENT_CACHE.relative_to(ROOT)),
            "real_latent_cache_sha256": sha256_file(LATENT_CACHE),
            "real_latent_index": REAL_LATENT_INDEX,
            "synthetic_latents": "N(0,1) float32, seed 1911 streams (E1: stream 1, E4: streams 2 and 3)",
        },
        "oracle_assumption": ("E1-E4 use one known plaintext/ciphertext pair (T3 KPA). E5/E6 use ciphertext only (T1 COA): a single exact "
                              "ciphertext latent. In all experiments the secret seed is planted inside the reduced range, so the controls verify "
                              "the search and the oracle, not the base rate of a real attack."),
        "reduced_space_bits": REDUCED_BITS,
        "secret_seed_recovered": checks["E1_recovered"] and checks["E2_recovered"],
        "candidates_tested": {"e1_s0_naive_d4096": e1["candidates_tested"], "e2_s0_screened_d65536": e2["candidates_tested"]},
        "results": results,
        "timing_nondeterministic": timing,
        "conclusion": (
            f"[事實] 在縮小至 2^{REDUCED_BITS} 的 seed 空間中，以一對已知明文/密文 latent，窮舉重新產生前作金鑰即可找回祕密 seed："
            f"d=4096 簡單窮舉測試 {e1['candidates_tested']:,} 個候選後找回 seed {secret_e1}；d=65536（真實 latent）以 64 座標預篩加全維驗證，"
            f"測試 {e2['candidates_tested']:,} 個候選後找回 seed {secret_e2}；祕密 seed 在範圍外的 negative control 掃完 2^{REDUCED_BITS} 個候選皆無結果；"
            f"S1 把符號 seed 與置換 seed 分開搜尋，共 {e4['total_candidates_tested']:,} 個候選（聯合搜尋為 2^{2 * S1_BITS}）即找回兩者。"
            f"[事實] 唯密文（COA，無任何明文）：S0 單一密文 latent 的正確金鑰分數 {e5['correct_score_sigma']:.1f}σ，"
            f"2^{COA_BITS} 個錯誤候選最高僅 {e5['wrong_max_sigma']:.1f}σ，正確金鑰排第 1；"
            f"S1 單一密文分兩階段，置換 seed {e6['permutation_correct_sigma']:.1f}σ、符號 seed {e6['sign_correct_sigma']:.1f}σ，皆排第 1，"
            "不需金鑰重用。分數本身即以 σ 為單位（錯誤金鑰 ~ N(0,1)）。"
            "[推定] 2^32 個 N(0,1) 錯誤候選的期望最大值約 6.7σ，仍遠低於正確金鑰，故全尺度下正確金鑰仍排第 1；"
            "外推時間見 timing_nondeterministic（CPU 單執行緒，非 GPU）。"
            f"[事實] 全樣本（security_v1 200 張，e7）：S0 與 S1 符號階段對 "
            f"{e7['s0_and_s1_sign_correct_sigma']['n_below_full_scale_threshold'] == 0 and '200/200' or 'most'} 張皆高於 2^32 門檻（{e7['full_scale_threshold_sigma']:.2f}σ）；"
            f"S1 置換階段有 {e7['s1_permutation_correct_sigma']['n_below_full_scale_threshold']}/200 張低於門檻"
            f"（最弱為 index {e7['s1_permutation_weakest_image_index']}），其餘單張即可。"
            "[推定] 前作所有影像共用一把金鑰，攻擊者可挑訊號最強的一張密文找回金鑰再解開全部，故部署層級『單一密文即可』成立；"
            "不寫成『每一張影像單張都可以』。"
            "[結論] 前作在 T1 下不需要已知明文：單一密文即可在 32-bit seed 空間中辨識金鑰，密碼強度幾乎無關。"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    def _json_default(value):
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"E1 tested={e1['candidates_tested']:,} recovered={e1['recovered_seed']} (secret {secret_e1}) wall={e1['wall']:.1f}s")
    print(f"E2 tested={e2['candidates_tested']:,} recovered={e2['recovered_seed']} (secret {secret_e2}) wall={e2['wall']:.1f}s")
    print(f"E3 tested={e3['candidates_tested']:,} recovered={e3['recovered_seed']} wall={e3['wall']:.1f}s")
    print(f"E4 sign tested={e4['sign_candidates_tested']:,} perm tested={e4['permutation_candidates_tested']:,}")
    for name, value in benchmarks.items():
        print(f"{name}: {value:,.0f}")
    for name, value in timing["extrapolation_to_2_32"].items():
        if "worst_case_2_32" in value:
            print(f"{name}: worst-case {value['worst_case_2_32']['days']:.2f} d, {value['worst_case_2_32']['hours']:.1f} h")


def s1_wall(e4: dict, which: str) -> float:
    return e4["wall_sign"] if which == "sign" else e4["wall_perm"]


if __name__ == "__main__":
    main()
