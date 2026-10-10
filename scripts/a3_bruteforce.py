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
        "candidates_per_s": rate,
        "worst_case_2_32": human(full / rate),
        "expected_half_2_31": human(full / 2 / rate),
    }


# ---------------------------------------------------------------------------
def main() -> None:
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
            "split_search_bits_equivalent": float(np.log2(e4["total_candidates_tested"])),
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
        "oracle_assumption": ("The attacker holds one known plaintext/ciphertext latent pair (z, z_enc); the secret seed is chosen by this "
                              "script inside the reduced range, so the controls verify the search procedure, not the success rate of a real attack"),
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
            "[推定] 外推到 2^32 的時間見 timing_nondeterministic.extrapolation_to_2_32（吞吐量隨機器而異，假設線性縮放、單執行緒）。"
            "[假設] 真實攻擊需要一對已知明文/密文；無已知明文時是否存在可用 oracle 本任務未驗證。"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
