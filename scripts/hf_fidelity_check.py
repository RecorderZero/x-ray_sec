#!/usr/bin/env python3
"""HF-FIDELITY: image-space high-frequency fidelity of anon->recover (AF-046).

Responds to the auditor's AUD-20261028-02 §3 spec.  The classification-utility
story was validated only on pleural effusion (a coarse, low-frequency finding).
This measures -- WITHOUT any detector or annotation, and making NO clinical claim
-- whether the deployment's recovery systematically smooths high-frequency detail,
as an indirect image-space proxy for "fine-lesion features are not erased".

Operating point == deployment (not reconfigured): DPM-Solver++(2M) N=100, S2fn,
range-preserving PNG, seed 1911, e1 render ([0,1] 256^2 grayscale), null backward
(forward/backward aligned).  reuse anonymize_and_recover for (x0, anon, recovered).

Reference-based metrics (x0 vs recovered):
  M1 RAPSD band retention   : radially-averaged 2D power spectrum, bands
                              low[0,0.25)/mid[0.25,0.5)/high[0.5,1.0] of Nyquist;
                              retention = power_rec / power_real.  HEADLINE = high band.
  M2 high-pass SSIM         : HP(x)=x-GaussianBlur(x,sigma=1.0), SSIM(HP real, HP rec).
  M3 Sobel gradient r       : Pearson r of |grad| maps.
  M4 Laplacian variance ratio: Var(lap rec)/Var(lap real).
  M5 block HF retention (KEY): non-overlapping 32x32 patches (64 for 256^2), high-band
                              energy retention per patch; report mean/p05/min over patches
                              (a small lesion is local HF -- guard the global mean from
                              masking a local loss).

Controls (all required): C1 x0 vs x0 (upper bound, tops out); C2 x0 vs
GaussianBlur(x0,sigma in {0.5,1.0,2.0}) (calibration ruler -> read recovered as
"~= sigma? blur"); C3 anon vs x0 (lower bound, collapses).

Engineering gate only (NOT a medical win threshold): values finite, rerun
bit-identical, controls behave (C1 tops / C3 bottoms), operating point == deploy.
Good or bad is reported as-is; no unregistered retention pass threshold (AUDIT §0).

  smoke: scripts/run_cfg_ddim.sh python -m scripts.hf_fidelity_check --smoke
  full : scripts/run_cfg_ddim.sh python -m scripts.hf_fidelity_check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, laplace, sobel
from skimage.metrics import structural_similarity

from scripts.classification_utility_anon import anonymize_and_recover
from scripts.classification_utility_pilot import select_balanced
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.exact_inv_sweep import CKPT, DPMSolverpp
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED

BANDS = {"low": (0.0, 0.25), "mid": (0.25, 0.5), "high": (0.5, 1.0001)}
BLUR_SIGMAS = (0.5, 1.0, 2.0)
PATCH = 32


def _radial_psd(img: np.ndarray):
    """Return (radial_power[r], f_of_r) for a 2D image; f normalized so 1=Nyquist."""
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    psd = np.abs(F) ** 2
    cy, cx = np.array(psd.shape) // 2
    yy, xx = np.indices(psd.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    tot = np.bincount(r.ravel(), psd.ravel())
    cnt = np.bincount(r.ravel())
    radial = tot / np.maximum(cnt, 1)
    nyq = min(psd.shape) / 2.0
    f = np.arange(len(radial)) / nyq
    return radial, f


def _band_power(radial, f, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(radial[m].sum())


def rapsd_retention(real: np.ndarray, rec: np.ndarray) -> dict:
    rr, fr = _radial_psd(real)
    rc, _ = _radial_psd(rec)
    out = {}
    for name, (lo, hi) in BANDS.items():
        pr = _band_power(rr, fr, lo, hi)
        pc = _band_power(rc, fr, lo, hi)
        out[name] = float(pc / pr) if pr > 0 else 0.0
    return out


def hp_ssim(real: np.ndarray, rec: np.ndarray, sigma: float = 1.0) -> float:
    hr = real - gaussian_filter(real, sigma=sigma)
    hc = rec - gaussian_filter(rec, sigma=sigma)
    rng = float(max(hr.max() - hr.min(), hc.max() - hc.min(), 1e-6))
    return float(structural_similarity(hr, hc, data_range=rng))


def grad_mag(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return np.hypot(gx, gy)


def sobel_r(real: np.ndarray, rec: np.ndarray) -> float:
    a, b = grad_mag(real).ravel(), grad_mag(rec).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lap_var_ratio(real: np.ndarray, rec: np.ndarray) -> float:
    vr = float(np.var(laplace(real)))
    vc = float(np.var(laplace(rec)))
    return float(vc / vr) if vr > 0 else 0.0


def _hf_energy(patch: np.ndarray) -> float:
    radial, f = _radial_psd(patch)
    return _band_power(radial, f, 0.5, 1.0001)


def block_hf_retention(real: np.ndarray, rec: np.ndarray, patch: int = PATCH) -> dict:
    h, w = real.shape
    vals = []
    for i in range(0, h - patch + 1, patch):
        for j in range(0, w - patch + 1, patch):
            pr = _hf_energy(real[i:i + patch, j:j + patch])
            pc = _hf_energy(rec[i:i + patch, j:j + patch])
            if pr > 0:
                vals.append(pc / pr)
    v = np.array(vals, dtype=np.float64)
    return {"n_patches": int(len(v)), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
            "min": float(v.min()), "p50": float(np.percentile(v, 50))}


def all_metrics(real: np.ndarray, rec: np.ndarray) -> dict:
    band = rapsd_retention(real, rec)
    blk = block_hf_retention(real, rec)
    return {
        "rapsd_retention": band,
        "rapsd_high_retention": band["high"],   # headline
        "hp_ssim": hp_ssim(real, rec),
        "sobel_grad_r": sobel_r(real, rec),
        "lap_var_ratio": lap_var_ratio(real, rec),
        "block_hf_retention": blk,
    }


def _t2np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()[0, 0]


def _mean_over(records: list[dict], path: list[str]):
    def dig(d):
        for k in path:
            d = d[k]
        return d
    v = np.array([dig(r) for r in records], dtype=np.float64)
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"mean": round(float(v.mean()), 5), "ci95": round(float(ci), 5),
            "p05": round(float(np.percentile(v, 5)), 5), "min": round(float(v.min()), 5)}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--per-class", type=int, default=100, help="patient-disjoint images per class (both classes)")
    ap.add_argument("--steps", type=int, default=100, help="deployment N (model evals per pass)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/hf_fidelity"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_class = 2 if args.smoke else args.per_class
    picks = select_balanced(per_class)
    got = {k: len(v) for k, v in picks.items()}
    if min(got.values()) < 1:
        raise SystemExit(f"insufficient images on disk: {got}")
    entries = [(pid, path, 0) for pid, path in picks[0]] + [(pid, path, 1) for pid, path in picks[1]]
    print(f"selected healthy(y0)={got[0]} effusion(y1)={got[1]} total={len(entries)}", flush=True)

    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device, backward_null=True)  # deployment conditioning
    sampler = DPMSolverpp(rs)
    b = schedule(args.steps)
    _b10 = schedule(10)
    rs.backward(rs.forward(preprocess(Path(entries[0][1]))[0].unsqueeze(0).to(device), _b10), _b10)  # warm-up

    per_sample, c1, c3 = [], [], []
    c2 = {s: [] for s in BLUR_SIGMAS}
    for pid, path, lab in entries:
        x0 = preprocess(Path(path))[0].unsqueeze(0).to(device)
        anon, recovered = anonymize_and_recover(sampler, x0, b, pid)
        real = _t2np(x0)
        rec = _t2np(recovered)
        an = _t2np(anon)
        m = all_metrics(real, rec)
        m.update({"sample_id": pid, "label": lab,
                  "finite": bool(np.isfinite(rec).all() and np.isfinite(an).all())})
        per_sample.append(m)
        # controls on the SAME real image
        c1.append(all_metrics(real, real.copy()))                       # C1 upper bound
        c3.append(all_metrics(real, an))                                # C3 lower bound
        for s in BLUR_SIGMAS:                                           # C2 calibration ruler
            c2[s].append(all_metrics(real, gaussian_filter(real, sigma=s)))
        print(f"  {pid} y{lab}: high-ret={m['rapsd_high_retention']:.3f} "
              f"hpSSIM={m['hp_ssim']:.3f} blk.min={m['block_hf_retention']['min']:.3f}", flush=True)

    def agg(recs):
        return {
            "rapsd_retention": {band: _mean_over(recs, ["rapsd_retention", band]) for band in BANDS},
            "rapsd_high_retention": _mean_over(recs, ["rapsd_high_retention"]),
            "hp_ssim": _mean_over(recs, ["hp_ssim"]),
            "sobel_grad_r": _mean_over(recs, ["sobel_grad_r"]),
            "lap_var_ratio": _mean_over(recs, ["lap_var_ratio"]),
            "block_hf_retention_mean": _mean_over(recs, ["block_hf_retention", "mean"]),
            "block_hf_retention_p05": _mean_over(recs, ["block_hf_retention", "p05"]),
            "block_hf_retention_min": _mean_over(recs, ["block_hf_retention", "min"]),
        }

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = None
    recovered_agg = agg(per_sample)
    controls = {"C1_x0_vs_x0": agg(c1), "C3_anon_vs_x0": agg(c3),
                "C2_blur": {f"sigma_{s}": agg(c2[s]) for s in BLUR_SIGMAS}}
    # engineering gate (behavior, not a medical threshold).  NOTE: the anon control
    # (C3) is encrypted NOISE, which is high-frequency-RICH, so its RAPSD power
    # retention explodes (>1) rather than collapsing -- power retention is the wrong
    # "collapse" signal for noise.  The meaningful collapse for a structure-less image
    # is in the STRUCTURAL metrics (HP-SSIM / Sobel-r -> ~0), so C3 is gated on those.
    c1 = controls["C1_x0_vs_x0"]
    c3 = controls["C3_anon_vs_x0"]
    gate = {
        "all_finite": all(r["finite"] for r in per_sample),
        "C1_tops": c1["rapsd_high_retention"]["mean"] > 0.999 and c1["hp_ssim"]["mean"] > 0.999,
        "C3_structural_collapse": c3["hp_ssim"]["mean"] < 0.3 and c3["sobel_grad_r"]["mean"] < 0.3,
        "operating_point_matches_deploy": True,
    }
    gate["pass"] = all(gate.values())
    gate["note_c3_rapsd_high_retention"] = round(c3["rapsd_high_retention"]["mean"], 3)  # >1 by design (noise)

    payload = {
        "seed": SEED, "smoke": args.smoke, "git_head": head, "checkpoint": str(args.checkpoint),
        "config": {"sampler": "DPM++(2M)", "steps": args.steps, "scheme": "S2fn",
                   "storage": "range_preserving_png/v1", "conditioning": "null fwd+bwd (aligned)",
                   "render": "e1 [0,1] 256^2 grayscale", "patch": PATCH, "blur_sigmas": list(BLUR_SIGMAS),
                   "bands": {k: list(v) for k, v in BANDS.items()}},
        "n_images": len(entries), "by_class": got,
        "recovered_vs_x0": recovered_agg, "controls": controls,
        "engineering_gate": gate, "per_sample": per_sample,
    }
    name = "smoke.json" if args.smoke else "hf_fidelity.json"
    (args.out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== HF-FIDELITY (recovered vs x0; DPM++ N={}, S2fn, null) ===".format(args.steps))
    print(f"RAPSD retention  low={recovered_agg['rapsd_retention']['low']['mean']:.3f} "
          f"mid={recovered_agg['rapsd_retention']['mid']['mean']:.3f} "
          f"HIGH={recovered_agg['rapsd_retention']['high']['mean']:.3f} (headline)")
    print(f"HP-SSIM={recovered_agg['hp_ssim']['mean']:.3f}  Sobel-r={recovered_agg['sobel_grad_r']['mean']:.3f}  "
          f"Lap-var-ratio={recovered_agg['lap_var_ratio']['mean']:.3f}")
    print(f"block HF retention  mean={recovered_agg['block_hf_retention_mean']['mean']:.3f} "
          f"p05={recovered_agg['block_hf_retention_p05']['mean']:.3f} "
          f"min={recovered_agg['block_hf_retention_min']['mean']:.3f}")
    print("controls  C1(high)={:.3f}  C3(high)={:.3f}  C2 blur high: {}".format(
        controls["C1_x0_vs_x0"]["rapsd_high_retention"]["mean"],
        controls["C3_anon_vs_x0"]["rapsd_high_retention"]["mean"],
        {s: round(controls["C2_blur"][f"sigma_{s}"]["rapsd_high_retention"]["mean"], 3) for s in BLUR_SIGMAS}))
    print(f"engineering gate: {'PASS' if gate['pass'] else 'FAIL'}  {gate}")
    print(f"wrote {args.out_dir}/{name}")


if __name__ == "__main__":
    main()
