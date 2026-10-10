#!/usr/bin/env python3
"""A3.3 S0 / S1 magnitude invariants, verified on the predecessor's key objects.

Analytic identities (anonymization.py line numbers refer to
past/SourceCode/guided_diffusion/anonymization.py):

  S0  z_enc = M (.) (k (.) z) + (1 - M) (.) z with M = 1 (anonymization.py:680-682),
      k_i in {-1, +1}  =>  |z_enc_i| = |z_i| for every coordinate i.  IEEE-754
      negation is exact, so the equality is bit-exact, and ||z_enc|| = ||z||.
  S1  z_enc = P(k (.) z), i.e. z_enc[i] = k[pi(i)] z[pi(i)] (anonymization.py:392-397,
      225)  =>  |z_enc| = |z| o pi is a rearrangement of |z|:  sort(|z_enc|) = sort(|z|)
      and ||z_enc|| = ||z||, while the per-index equality |z_enc_i| = |z_i| holds
      only at fixed points of pi (and exact ties).

The script measures both, for synthetic N(0,1) latents (seed 1911) and cached
real inversion latents (E2.3), with the legacy password key (Ki@13579, seed 42
ignored) and with the seed-only key (seed 42), via DiffusionAnonymizer
(latent_shape=(1,256,256), mask_type='full', margin=0).  Negative controls
compare a ciphertext with the magnitudes of a different latent.  CPU only.

    scripts/run_cfg_ddim.sh python -m scripts.a3_invariants
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
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
assert not torch.cuda.is_available(), "A3.3 must run on CPU only"

SOURCE_ROOT = ROOT / "past/SourceCode"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from guided_diffusion.anonymization import (  # noqa: E402
    DiffusionAnonymizer,
    RademacherKey,
    SignedPermutationKey,
)

SEED = 1911
TASK_ID = "A3.3"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.a3_invariants"
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/A3.3.json"
ANON_PATH = SOURCE_ROOT / "guided_diffusion/anonymization.py"
LATENT_CACHE = ROOT / "artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a/latents.npy"
REAL_INDICES = (0, 49, 99, 149, 199)
N_SYNTHETIC = 5
KEY_SHAPE = (1, 256, 256)
LEGACY_PASSWORD = "Ki@13579"
LEGACY_SEED = 42

ANALYTIC_IDENTITY = (
    "S0：z_enc = M⊙(k⊙z) + (1−M)⊙z，全遮罩 M=1（anonymization.py:680-682），k_i∈{−1,+1}，故對每個座標 |z_enc_i| = |k_i||z_i| = |z_i|；"
    "IEEE-754 的變號是精確運算，所以逐位元相等，且 ||z_enc||₂ = ||z||₂。"
    "S1：z_enc = P(k⊙z)，即 z_enc[i] = k[π(i)]·z[π(i)]（anonymization.py:392-397、225），故 |z_enc| = |z|∘π 是 |z| 的重排："
    "sort(|z_enc|) = sort(|z|)、幅值多重集合與 ||z_enc||₂ 不變，但逐座標 |z_enc_i| = |z_i| 只在 π 的不動點（或剛好相同的幅值）成立。"
    "浮點歸約的加總順序會使 S1 的 ||·||₂ 在 float64 約 1e-16、float32 約 1e-7 的相對尺度上有順序效應，與解析等式無關。"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build_keys() -> dict:
    """Legacy password key (seed also passed but ignored) and the seed-only key, for S0 and S1."""
    return {
        "legacy_password": {
            "S0": RademacherKey(shape=KEY_SHAPE, seed=LEGACY_SEED, password=LEGACY_PASSWORD),
            "S1": SignedPermutationKey(shape=KEY_SHAPE, seed=LEGACY_SEED, password=LEGACY_PASSWORD),
        },
        "seed_only_42": {
            "S0": RademacherKey(shape=KEY_SHAPE, seed=LEGACY_SEED),
            "S1": SignedPermutationKey(shape=KEY_SHAPE, seed=LEGACY_SEED),
        },
    }


def l2_f64(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float64).ravel()))


def sumsq_fsum(x: np.ndarray) -> float:
    """Order-independent exactly rounded sum of squares (float64 squares of float32 values are exact)."""
    return math.fsum((x.astype(np.float64).ravel() ** 2).tolist())


def measure(z: np.ndarray, z_enc: np.ndarray, scheme: str, z_back: np.ndarray) -> dict:
    """z, z_enc: float32 arrays of shape (1,256,256)."""
    a, e = np.abs(z).ravel(), np.abs(z_enc).ravel()
    a64, e64 = a.astype(np.float64), e.astype(np.float64)
    sorted_a, sorted_e = np.sort(a64), np.sort(e64)
    changed = e != a
    n1, n2 = l2_f64(z), l2_f64(z_enc)
    f32_a = float(torch.linalg.norm(torch.from_numpy(z.ravel())))
    f32_e = float(torch.linalg.norm(torch.from_numpy(z_enc.ravel())))
    record = {
        "abs_elementwise_max_diff": float(np.max(np.abs(e64 - a64))),
        "abs_elementwise_bit_exact": bool(np.array_equal(e.view(np.uint32), a.view(np.uint32))),
        "sorted_abs_max_diff": float(np.max(np.abs(sorted_e - sorted_a))),
        "sorted_abs_bit_exact": bool(np.array_equal(np.sort(e).view(np.uint32), np.sort(a).view(np.uint32))),
        "n_indices": int(a.size),
        "n_indices_magnitude_changed": int(changed.sum()),
        "frac_indices_magnitude_changed": float(changed.mean()),
        "norm_l2_plain": n1,
        "norm_l2_encrypted": n2,
        "norm_abs_diff_float64": abs(n2 - n1),
        "norm_rel_diff_float64": abs(n2 - n1) / n1,
        "sumsq_fsum_equal": sumsq_fsum(z) == sumsq_fsum(z_enc),
        "norm_rel_diff_float32_torch": abs(f32_e - f32_a) / f32_a,
        "roundtrip_max_abs_error": float(np.max(np.abs(z_back.astype(np.float64) - z.astype(np.float64)))),
    }
    flat_z, flat_e = z.ravel(), z_enc.ravel()
    if scheme == "S0":
        record["frac_sign_flipped"] = float(((flat_e == -flat_z) & (flat_z != 0)).mean())
    else:
        # fixed points of the permutation implied by exact magnitude matches
        record["n_indices_magnitude_equal"] = int(a.size - changed.sum())
    return record


def encrypt_decrypt(anonymizer: DiffusionAnonymizer, z: np.ndarray, key) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline-shaped call: batch of one, (1,1,256,256)."""
    zt = torch.from_numpy(z).unsqueeze(0)
    z_enc = anonymizer.anonymize_latent(zt, key)
    z_back = anonymizer.deanonymize_latent(z_enc, key)
    return z_enc.squeeze(0).numpy(), z_back.squeeze(0).numpy()


def input_stats(z: np.ndarray) -> dict:
    return {
        "mean": float(z.mean(dtype=np.float64)), "std": float(z.std(dtype=np.float64)),
        "abs_max": float(np.abs(z).max()), "n_zero": int((z == 0).sum()),
        "n_distinct_magnitudes": int(np.unique(np.abs(z)).size),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=(__doc__ or '').strip().splitlines()[0])
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([OUTPUT], args.force)
    keys = build_keys()
    anonymizers = {
        "S0": DiffusionAnonymizer(latent_shape=KEY_SHAPE, mask_type="full", margin=0, use_permutation=False),
        "S1": DiffusionAnonymizer(latent_shape=KEY_SHAPE, mask_type="full", margin=0, use_permutation=True),
    }

    rng = np.random.default_rng(SEED)
    inputs: list[dict] = []
    for i in range(N_SYNTHETIC):
        z = rng.standard_normal(KEY_SHAPE).astype(np.float32)
        inputs.append({"kind": "synthetic", "id": f"synthetic_{i}", "z": z})
    latents = np.load(LATENT_CACHE, mmap_mode="r")
    assert latents.shape == (200, 1, 256, 256) and latents.dtype == np.float32
    for idx in REAL_INDICES:
        z = np.array(latents[idx], dtype=np.float32)
        assert np.isfinite(z).all()
        inputs.append({"kind": "real", "id": f"E2.3_latent_{idx}", "z": z, "cache_index": idx})

    records = []
    for item in inputs:
        z = item["z"]
        for variant, key_pair in keys.items():
            for scheme in ("S0", "S1"):
                z_enc, z_back = encrypt_decrypt(anonymizers[scheme], z, key_pair[scheme])
                rec = {"input": item["id"], "input_kind": item["kind"], "scheme": scheme, "key_variant": variant}
                rec.update(measure(z, z_enc, scheme, z_back))
                records.append(rec)

    # negative controls: compare a ciphertext with the magnitudes of a DIFFERENT latent
    negative = []
    for kind in ("synthetic", "real"):
        pair = [it for it in inputs if it["kind"] == kind][:2]
        za, zb = pair[0]["z"], pair[1]["z"]
        for scheme in ("S0", "S1"):
            z_enc, _ = encrypt_decrypt(anonymizers[scheme], za, keys["legacy_password"][scheme])
            ea, bb = np.abs(z_enc).ravel().astype(np.float64), np.abs(zb).ravel().astype(np.float64)
            negative.append({
                "input_kind": kind, "scheme": scheme, "ciphertext_of": pair[0]["id"], "compared_with_magnitudes_of": pair[1]["id"],
                "abs_elementwise_max_diff": float(np.max(np.abs(ea - bb))),
                "sorted_abs_max_diff": float(np.max(np.abs(np.sort(ea) - np.sort(bb)))),
                "expected": "both differences are clearly non-zero, i.e. the tests can fail",
            })

    # aggregate per (input kind, scheme)
    summary = {}
    for kind in ("synthetic", "real"):
        for scheme in ("S0", "S1"):
            sel = [r for r in records if r["input_kind"] == kind and r["scheme"] == scheme]
            entry = {
                "n_records": len(sel),
                "n_inputs": len({r["input"] for r in sel}),
                "max_abs_elementwise_max_diff": max(r["abs_elementwise_max_diff"] for r in sel),
                "max_sorted_abs_max_diff": max(r["sorted_abs_max_diff"] for r in sel),
                "max_norm_rel_diff_float64": max(r["norm_rel_diff_float64"] for r in sel),
                "max_norm_abs_diff_float64": max(r["norm_abs_diff_float64"] for r in sel),
                "max_norm_rel_diff_float32_torch": max(r["norm_rel_diff_float32_torch"] for r in sel),
                "all_sumsq_fsum_equal": all(r["sumsq_fsum_equal"] for r in sel),
                "max_roundtrip_max_abs_error": max(r["roundtrip_max_abs_error"] for r in sel),
                "min_frac_indices_magnitude_changed": min(r["frac_indices_magnitude_changed"] for r in sel),
                "max_frac_indices_magnitude_changed": max(r["frac_indices_magnitude_changed"] for r in sel),
            }
            if scheme == "S0":
                entry["all_abs_elementwise_bit_exact"] = all(r["abs_elementwise_bit_exact"] for r in sel)
                entry["frac_sign_flipped_min"] = min(r["frac_sign_flipped"] for r in sel)
                entry["frac_sign_flipped_max"] = max(r["frac_sign_flipped"] for r in sel)
            else:
                entry["all_sorted_abs_bit_exact"] = all(r["sorted_abs_bit_exact"] for r in sel)
                entry["max_n_indices_magnitude_equal"] = max(r["n_indices_magnitude_equal"] for r in sel)
            summary[f"{kind}_{scheme}"] = entry

    checks = {
        "S0_abs_invariant_exact_all": all(
            r["abs_elementwise_max_diff"] == 0.0 and r["abs_elementwise_bit_exact"] for r in records if r["scheme"] == "S0"
        ),
        "S1_sorted_abs_invariant_exact_all": all(
            r["sorted_abs_max_diff"] == 0.0 and r["sorted_abs_bit_exact"] for r in records if r["scheme"] == "S1"
        ),
        "S1_per_index_magnitudes_moved": all(
            r["frac_indices_magnitude_changed"] > 0.99 for r in records if r["scheme"] == "S1"
        ),
        "S0_signs_actually_flipped": all(0.45 < r["frac_sign_flipped"] < 0.55 for r in records if r["scheme"] == "S0"),
        "norm_rel_diff_float64_below_1e-13": all(r["norm_rel_diff_float64"] < 1e-13 for r in records),
        "norm_sumsq_fsum_exact": all(r["sumsq_fsum_equal"] for r in records),
        "roundtrip_exact": all(r["roundtrip_max_abs_error"] == 0.0 for r in records),
        "negative_controls_fail_as_expected": all(
            n["abs_elementwise_max_diff"] > 1e-3 and n["sorted_abs_max_diff"] > 1e-3 for n in negative
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"invariant check failed: {checks}")

    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "seed": SEED,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cpu (CUDA_VISIBLE_DEVICES='' set before importing torch)",
        "inputs": {
            "predecessor_source": "past/SourceCode/guided_diffusion/anonymization.py",
            "predecessor_source_sha256": sha256_file(ANON_PATH),
            "anonymizer": "DiffusionAnonymizer(latent_shape=(1,256,256), mask_type='full', margin=0, use_permutation=False for S0 / True for S1).anonymize_latent / deanonymize_latent on a (1,1,256,256) batch",
            "key_variants": {
                "legacy_password": f"password={LEGACY_PASSWORD!r} with seed={LEGACY_SEED} also passed (ignored by the predecessor, see A3.1 V3)",
                "seed_only_42": f"seed={LEGACY_SEED}, no password (S1 permutation seed = 42 + 1000000)",
            },
            "synthetic": {
                "description": f"{N_SYNTHETIC} N(0,1) float32 latents of shape (1,256,256), numpy default_rng(1911) drawn sequentially",
                "ids": [it["id"] for it in inputs if it["kind"] == "synthetic"],
                "stats": {it["id"]: input_stats(it["z"]) for it in inputs if it["kind"] == "synthetic"},
            },
            "real": {
                "description": "cached E2.3 inversion latents (float32, shape (1,256,256) each), not regenerated",
                "latent_cache": str(LATENT_CACHE.relative_to(ROOT)),
                "latent_cache_sha256": sha256_file(LATENT_CACHE),
                "cache_indices": list(REAL_INDICES),
                "ids": [it["id"] for it in inputs if it["kind"] == "real"],
                "stats": {it["id"]: input_stats(it["z"]) for it in inputs if it["kind"] == "real"},
            },
            "n_records": len(records),
        },
        "analytic_identity": ANALYTIC_IDENTITY,
        "summary": summary,
        "checks": checks,
        "negative_controls": negative,
        "records": records,
        "conclusion": (
            "[事實] 在 5 個合成 N(0,1) latent 與 5 個真實 E2.3 latent、legacy password 金鑰與 seed-only 金鑰下，"
            f"S0 的逐座標 |z_enc| 與 |z| 最大差為 {max(summary['synthetic_S0']['max_abs_elementwise_max_diff'], summary['real_S0']['max_abs_elementwise_max_diff'])}（逐位元相等）；"
            f"S1 的 sorted(|z_enc|) 與 sorted(|z|) 最大差為 {max(summary['synthetic_S1']['max_sorted_abs_max_diff'], summary['real_S1']['max_sorted_abs_max_diff'])}（逐位元相等），"
            f"但逐座標幅值在 {min(summary['synthetic_S1']['min_frac_indices_magnitude_changed'], summary['real_S1']['min_frac_indices_magnitude_changed']):.6f} 以上的座標被置換改變；"
            f"||z_enc||₂ = ||z||₂：float64 相對誤差最大 {max(s['max_norm_rel_diff_float64'] for s in summary.values()):.2e}（S0 為 0；S1 的殘差來自歸約順序），"
            f"float32 torch 範數相對誤差最大 {max(s['max_norm_rel_diff_float32_torch'] for s in summary.values()):.2e}，以 math.fsum 計的平方和精確相等。"
            "negative control（以不同 latent 的幅值比對）差值明顯非零，故檢驗可被否證。"
            "[推定] 這些不變量不依賴金鑰，使無金鑰的幅值比對成為可行的攻擊面（A3.4 以 gallery 實測，不在本任務內）。"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    for name, entry in summary.items():
        print(
            f"{name:<14} abs_diff={entry['max_abs_elementwise_max_diff']:.3g} sorted_diff={entry['max_sorted_abs_max_diff']:.3g} "
            f"norm_rel64={entry['max_norm_rel_diff_float64']:.3g} norm_rel32={entry['max_norm_rel_diff_float32_torch']:.3g} "
            f"moved_frac=[{entry['min_frac_indices_magnitude_changed']:.6f},{entry['max_frac_indices_magnitude_changed']:.6f}]"
        )


if __name__ == "__main__":
    main()
