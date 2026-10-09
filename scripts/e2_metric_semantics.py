#!/usr/bin/env python3
"""E2.4 metric semantics: what a pixel-space cosine of 0.999 does and does not mean.

Background (AUDIT.md AF-022): the predecessor reported "Cosine Sim 0.9989" and
described it as the cosine between the decrypted latent Z^rec and the original
latent Z.  The code (past/SourceCode/scripts/evaluation_metrics.py, lines
175-190) instead flattens each raw output image of the ``samples`` array
(shape (N, 1, 256, 256) -> 65536-vector per image, float32, no mean
subtraction, no uint8 conversion, no per-image min-max) and averages the
per-image cosine.  This script measures, on the 20 dev_v1.1 images and in the
same [0, 1] float scale, what such a pixel-space cosine looks like for

  a. different-patient pairs (190 pairs),
  b. the same image plus i.i.d. Gaussian noise at PSNR ~ 30.6 dB and ~ 26.2 dB,
  c. different-patient pairs after per-image mean subtraction,
  d. constructive counterexamples with cosine = 1 (or ~ 0.9995) but x != y,
  e. a cosine-vs-PSNR table under additive noise.

It is an independent implementation (it does not import or copy the auditor's
audit/e2_legacy_cosine_semantics.py); the auditor's JSON is only read in the
final ``cross_check`` block.

CPU only; no checkpoint is loaded.  Importing scripts.e1_ddim_runner runs the
environment guard, so always run through the wrapper from the repo root:

    scripts/run_cfg_ddim.sh python -m scripts.e2_metric_semantics
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e1_ddim_runner import load_split, preprocess  # noqa: E402  (runs env guard)

SEED = 1911
TASK_ID = "E2.4"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.e2_metric_semantics"
SPLIT = ROOT / "splits/dev_v1.1.csv"
OUTPUT = ROOT / "results/E2.4_metric_semantics.json"
AUDITOR_JSON = ROOT / "audit/out/e2_legacy_cosine_semantics.json"

# (b) PSNR targets that bracket the predecessor's reported 30.23 dB.
B_TARGETS = (30.6, 26.2)
# (e) cosine-vs-PSNR table.
E_TARGETS = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
# (d) pure rescaling counterexamples: y = a * x.
D_SCALES = (0.5, 0.9)
# (d2) localized corruption: invert (1 - x) a square patch in the image centre.
# The side (12) was chosen post hoc so that the mean cosine is ~0.9989, the
# predecessor's reported value; it illustrates, and does not model, their error.
PATCH_SIDE = 12
PATCH_ORIGIN = (120, 120)


# ---------------------------------------------------------------------------
# metrics (float64 unless stated)
# ---------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a, b) = a.b / (|a||b|) on flattened float64 vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def legacy_cosine_float32(a32: np.ndarray, b32: np.ndarray) -> float:
    """Mirror evaluation_metrics.py:181-186 on float32 vectors (np.dot/np.linalg.norm)."""
    dot = np.dot(a32, b32)
    return float(dot / (np.linalg.norm(a32) * np.linalg.norm(b32)))


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y - x) ** 2))


def psnr_db(x: np.ndarray, y: np.ndarray) -> float:
    """PSNR on the [0, 1] scale (peak 1), -10 log10(MSE); inf when identical."""
    err = mse(x, y)
    return float("inf") if err == 0.0 else float(-10.0 * math.log10(err))


def max_abs(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(y - x)))


def uint8_equality_rate(x: np.ndarray, y: np.ndarray) -> float:
    x8 = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
    y8 = np.rint(np.clip(y, 0, 1) * 255).astype(np.uint8)
    return float(np.mean(x8 == y8))


def float32_bit_exact_rate(x: np.ndarray, y: np.ndarray) -> float:
    x32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    y32 = np.ascontiguousarray(y, dtype=np.float32).view(np.uint32)
    return float(np.mean(x32 == y32))


def summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    p5, p95 = np.percentile(arr, [5, 95])  # linear interpolation
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p05": float(p5),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(p95),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# noise construction
# ---------------------------------------------------------------------------
def draw(index: int, target_db: float) -> np.ndarray:
    """Standard-normal vector, independent per (target PSNR, image index)."""
    rng = np.random.default_rng([SEED, int(round(target_db * 10)), index])
    return rng.standard_normal(65536)


def add_noise_exact(x: np.ndarray, target_db: float, index: int) -> np.ndarray:
    """x + n, with n rescaled so that mean(n^2) = 10^(-target/10) exactly (no clipping)."""
    n = draw(index, target_db)
    n *= math.sqrt(10.0 ** (-target_db / 10.0)) / math.sqrt(float(np.mean(n * n)))
    return x + n


def add_noise_clipped(x: np.ndarray, target_db: float, index: int) -> np.ndarray:
    """clip(x + s*n, 0, 1) with s found by bisection so that PSNR hits the target.

    MSE(s) is non-decreasing in s (per pixel |clip(x + s n) - x| is monotone),
    so bisection converges; 80 halvings reach float64 resolution.
    """
    n = draw(index, target_db)
    goal = 10.0 ** (-target_db / 10.0)
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mse(x, np.clip(x + mid * n, 0.0, 1.0)) < goal:
            lo = mid
        else:
            hi = mid
    return np.clip(x + 0.5 * (lo + hi) * n, 0.0, 1.0)


def noise_block(images: list[np.ndarray], target_db: float, clipped: bool) -> dict:
    make = add_noise_clipped if clipped else add_noise_exact
    cos_list, psnr_list, theory_list = [], [], []
    for i, x in enumerate(images):
        y = make(x, target_db, i)
        cos_list.append(cosine(x, y))
        psnr_list.append(psnr_db(x, y))
        # cos ~ 1/sqrt(1 + N/S) if the noise is orthogonal to x; N, S = mean squares.
        theory_list.append(1.0 / math.sqrt(1.0 + mse(x, y) / float(np.mean(x * x))))
    return {
        "target_psnr_db": target_db,
        "clipped_to_unit_range": clipped,
        "achieved_psnr_db_mean": float(np.mean(psnr_list)),
        "achieved_psnr_db_min": float(np.min(psnr_list)),
        "achieved_psnr_db_max": float(np.max(psnr_list)),
        "cosine_mean": float(np.mean(cos_list)),
        "cosine_min": float(np.min(cos_list)),
        "cosine_max": float(np.max(cos_list)),
        "cosine_image0": float(cos_list[0]),
        "psnr_db_image0": float(psnr_list[0]),
        "cosine_per_image": [float(v) for v in cos_list],
        "orthogonal_noise_theory_cosine_mean": float(np.mean(theory_list)),
        "orthogonal_noise_theory_max_abs_dev": float(
            np.max(np.abs(np.asarray(cos_list) - np.asarray(theory_list)))
        ),
    }


# ---------------------------------------------------------------------------
# counterexamples
# ---------------------------------------------------------------------------
def pair_record(x: np.ndarray, y: np.ndarray) -> dict:
    psnr = psnr_db(x, y)
    return {
        "cosine": cosine(x, y),
        "one_minus_cosine": 1.0 - cosine(x, y),
        "legacy_float32_cosine": legacy_cosine_float32(
            x.astype(np.float32), y.astype(np.float32)
        ),
        "psnr_db": None if math.isinf(psnr) else psnr,  # null = identical (MSE 0)
        "max_abs": max_abs(x, y),
        "mse": mse(x, y),
        "uint8_pixel_equality_rate": uint8_equality_rate(x, y),
        "float32_bit_exact_rate": float32_bit_exact_rate(x, y),
        "arrays_equal": bool(np.array_equal(x.astype(np.float32), y.astype(np.float32))),
    }


def aggregate(records: list[dict]) -> dict:
    keys = ("cosine", "max_abs", "uint8_pixel_equality_rate", "float32_bit_exact_rate")
    out = {f"{k}_mean": float(np.mean([r[k] for r in records])) for k in keys}
    finite = [r["psnr_db"] for r in records if r["psnr_db"] is not None]
    out["psnr_db_mean"] = float(np.mean(finite)) if finite else None  # None: all identical
    out["psnr_db_min"] = float(np.min(finite)) if finite else None
    out["psnr_db_max"] = float(np.max(finite)) if finite else None
    out["cosine_min"] = float(np.min([r["cosine"] for r in records]))
    out["cosine_max"] = float(np.max([r["cosine"] for r in records]))
    out["max_abs_max"] = float(np.max([r["max_abs"] for r in records]))
    out["any_pair_equal"] = bool(any(r["arrays_equal"] for r in records))
    return out


def patch_invert(x: np.ndarray) -> np.ndarray:
    """Replace a PATCH_SIDE^2 centre patch by its photographic negative (1 - x)."""
    img = x.reshape(256, 256).copy()
    r, c = PATCH_ORIGIN
    img[r : r + PATCH_SIDE, c : c + PATCH_SIDE] = 1.0 - img[r : r + PATCH_SIDE, c : c + PATCH_SIDE]
    return img.reshape(-1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def diff_entry(mine: float, auditor: float) -> dict:
    return {"this_script": mine, "auditor": auditor, "abs_diff": abs(mine - auditor)}


def cross_check(a: dict, b_exact: dict, b_clip: dict, c: dict) -> dict:
    """Compare with the auditor's numbers; never fails, only reports."""
    aud = json.loads(AUDITOR_JSON.read_text(encoding="utf-8"))
    out: dict = {
        "auditor_json": str(AUDITOR_JSON.relative_to(ROOT)),
        "auditor_json_sha256": sha256_file(AUDITOR_JSON),
        "protocol_note": (
            "(a) and (c) use the same 190 pairs and should agree to float precision. "
            "(b) differs by protocol: the auditor used image 0 only, Gaussian sigma 0.03/0.05 "
            "with clipping to [0,1] (achieved 30.58/26.22 dB); this script reports the mean over "
            "20 images with noise scaled to the PSNR target (exact, unclipped = primary; "
            "clipped variant and the image-0 clipped value are given for comparison)."
        ),
    }
    ad = aud["different_patients_pixel_cosine"]
    out["a_different_patients_pixel_cosine"] = {
        "median": diff_entry(a["median"], ad["median"]),
        "min": diff_entry(a["min"], ad["min"]),
        "max": diff_entry(a["max"], ad["max"]),
        "n_pairs": {"this_script": a["n"], "auditor": ad["n_pairs"]},
    }
    out["c_mean_removed_cosine_median"] = diff_entry(
        c["median"], aud["different_patients_mean_removed_cosine_median"]
    )
    b_rows = []
    for entry in aud["same_image_plus_noise"]:
        target = round(entry["psnr_db"], 1)  # 30.6 or 26.2
        exact = b_exact[str(target)]
        clip = b_clip[str(target)]
        b_rows.append(
            {
                "auditor_sigma": entry["sigma"],
                "auditor_psnr_db_image0": entry["psnr_db"],
                "auditor_cosine_image0": entry["cosine"],
                "psnr_db_diff_vs_this_script_clipped_mean20": abs(
                    clip["achieved_psnr_db_mean"] - entry["psnr_db"]
                ),
                "cosine_vs_this_script_unclipped_mean20": diff_entry(
                    exact["cosine_mean"], entry["cosine"]
                ),
                "cosine_vs_this_script_clipped_mean20": diff_entry(
                    clip["cosine_mean"], entry["cosine"]
                ),
                "cosine_vs_this_script_clipped_image0": diff_entry(
                    clip["cosine_image0"], entry["cosine"]
                ),
                "this_script_clipped_image0_psnr_db": clip["psnr_db_image0"],
            }
        )
    out["b_same_image_plus_noise"] = b_rows
    return out


def main() -> None:
    rows = load_split(SPLIT)
    sample_ids = [r["sample_id"] for r in rows]
    patients = [r["patient_id"] for r in rows]
    if len(set(patients)) != len(patients):
        raise ValueError("dev split must contain one image per patient for the pair analysis")

    images32 = [preprocess(ROOT / r["local_path"])[1] for r in rows]  # float32, [0, 1]
    images = [arr.astype(np.float64).ravel() for arr in images32]
    assert all(arr.size == 65536 for arr in images)
    value_range = {
        "min": float(min(a.min() for a in images)),
        "max": float(max(a.max() for a in images)),
        "mean_of_image_means": float(np.mean([a.mean() for a in images])),
        "mean_of_image_mean_squares": float(np.mean([np.mean(a * a) for a in images])),
    }

    # (a) different-patient pairs ------------------------------------------------
    pairs = list(itertools.combinations(range(len(images)), 2))
    pair_cos = [cosine(images[i], images[j]) for i, j in pairs]
    pair_cos32 = [legacy_cosine_float32(images32[i].ravel(), images32[j].ravel()) for i, j in pairs]
    a_block = summary(pair_cos)
    a_block["n_pairs"] = len(pairs)
    a_block["pair_psnr_db"] = summary([psnr_db(images[i], images[j]) for i, j in pairs])
    a_block["legacy_float32_vs_float64_max_abs_diff"] = float(
        np.max(np.abs(np.asarray(pair_cos) - np.asarray(pair_cos32)))
    )

    # (c) mean-subtracted pairs ----------------------------------------------------
    centered = [a - a.mean() for a in images]
    c_cos = [cosine(centered[i], centered[j]) for i, j in pairs]
    c_block = summary(c_cos)
    c_block["n_pairs"] = len(pairs)

    # (b) noise at the predecessor's PSNR level -----------------------------------
    b_exact = {str(t): noise_block(images, t, clipped=False) for t in B_TARGETS}
    b_clip = {str(t): noise_block(images, t, clipped=True) for t in B_TARGETS}

    # (d) cosine = 1 without equality -----------------------------------------------
    d_scale = {}
    for scale in D_SCALES:
        recs = [pair_record(x, scale * x) for x in images]
        d_scale[str(scale)] = {
            "scale_a": scale,
            "image0": recs[0],
            "over_20_images": aggregate(recs),
        }
    recs_patch = [pair_record(x, patch_invert(x)) for x in images]
    d_patch = {
        "description": (
            f"{PATCH_SIDE}x{PATCH_SIDE} patch at rows/cols {PATCH_ORIGIN[0]}.. replaced by 1 - x "
            f"({PATCH_SIDE * PATCH_SIDE / 65536:.4%} of the pixels)"
        ),
        "selection_note": (
            "patch side chosen post hoc so that the mean cosine is ~0.9989 (the predecessor's "
            "reported value); an illustration of cosine ~ 0.999 with MaxAbs ~ 0.7, not a model "
            "of the predecessor's actual error"
        ),
        "image0": recs_patch[0],
        "over_20_images": aggregate(recs_patch),
    }
    identity = aggregate([pair_record(x, x.copy()) for x in images])

    # (e) cosine vs PSNR under additive noise -------------------------------------
    e_table = []
    for t in E_TARGETS:
        blk = noise_block(images, t, clipped=False)
        e_table.append(
            {
                "target_psnr_db": t,
                "achieved_psnr_db_mean": blk["achieved_psnr_db_mean"],
                "cosine_mean": blk["cosine_mean"],
                "cosine_min": blk["cosine_min"],
                "cosine_max": blk["cosine_max"],
                "orthogonal_noise_theory_cosine_mean": blk["orthogonal_noise_theory_cosine_mean"],
            }
        )
    e_reference = {
        "different_patient_median_cosine": a_block["median"],
        "note": "rows with cosine above the different-patient median carry no PSNR information in the cosine itself",
    }

    cross = cross_check(a_block, b_exact, b_clip, c_block)

    result = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git("status", "--porcelain", "--untracked-files=no")),
        "split": str(SPLIT.relative_to(ROOT)),
        "split_sha256": rows[0]["split_sha256"],
        "sample_ids": sample_ids,
        "n_images": len(images),
        "n_distinct_patients": len(set(patients)),
        "seed": SEED,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "pixel_scale": {
            "description": (
                "preprocess() output: float32 in [0,1] (per-image min-max), 256x256 flattened to 65536; "
                "metrics in float64; no mean subtraction unless stated; peak=1 for PSNR"
            ),
            "legacy_equivalent": (
                "evaluation_metrics.py:175-190 flattens raw float32 samples[i] (1x256x256), "
                "no mean subtraction, mean over samples"
            ),
            **value_range,
        },
        "a_different_patient_pixel_cosine": a_block,
        "b_same_image_plus_gaussian_noise": {
            "primary_exact_scaled_unclipped": b_exact,
            "variant_clipped_to_unit_range": b_clip,
            "noise_model": "i.i.d. N(0,1) per pixel, rng seeds [1911, round(10*PSNR), image_index]",
        },
        "c_different_patient_mean_removed_cosine": c_block,
        "d_cosine_one_without_equality": {
            "identity_control": identity,
            "rescaling_y_equals_a_times_x": d_scale,
            "localized_corruption_negative_patch": d_patch,
        },
        "e_cosine_vs_psnr_table_unclipped_noise": {"rows": e_table, **e_reference},
        "cross_check": cross,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"(a) different-patient pairs n={a_block['n_pairs']}: median={a_block['median']:.6f} "
          f"min={a_block['min']:.6f} max={a_block['max']:.6f} p05={a_block['p05']:.6f} p95={a_block['p95']:.6f}")
    for t in B_TARGETS:
        e, k = b_exact[str(t)], b_clip[str(t)]
        print(f"(b) target {t} dB: unclipped cos_mean={e['cosine_mean']:.6f} (psnr {e['achieved_psnr_db_mean']:.3f}); "
              f"clipped cos_mean={k['cosine_mean']:.6f} (psnr {k['achieved_psnr_db_mean']:.3f})")
    print(f"(c) mean-removed different-patient median={c_block['median']:.6f}")
    for scale, blk in d_scale.items():
        agg = blk["over_20_images"]
        print(f"(d) y={scale}*x: cos_mean={agg['cosine_mean']:.12f} psnr_mean={agg['psnr_db_mean']:.2f} "
              f"max_abs_mean={agg['max_abs_mean']:.3f} any_equal={agg['any_pair_equal']}")
    agg = d_patch["over_20_images"]
    print(f"(d2) negative patch: cos_mean={agg['cosine_mean']:.6f} psnr_mean={agg['psnr_db_mean']:.2f} "
          f"max_abs_mean={agg['max_abs_mean']:.3f}")
    for row in e_table:
        print(f"(e) {row['target_psnr_db']:>4.0f} dB -> cosine mean {row['cosine_mean']:.6f}")


if __name__ == "__main__":
    main()
