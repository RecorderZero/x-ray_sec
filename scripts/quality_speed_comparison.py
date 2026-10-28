#!/usr/bin/env python3
"""Final-vs-predecessor reconstruction quality & speed comparison (user request 2026-10-28).

Two deployments on the SAME patient-disjoint image set, timed end-to-end:
  * OURS (final): DPM-Solver++(2M) null-backward + S2fn + range-preserving PNG,
    swept over N (model evals per pass).  Inversion AND generation are both
    unconditional (null), so forward/backward are fully aligned.
  * PREDECESSOR (original): legacy dense DDIM T=500 (~499 evals/pass) + S1
    SignedPermutationKey + legacy min-max PNG storage -- 學長's shipped pipeline.

Per image we report the recovery-vs-x0 image metrics the predecessor paper used
(PSNR / SSIM / MAE / MSE / cosine / Linf) and the encrypt->decrypt latent
round-trip cosine -- the lossless-crypto check cos(z, D(E(z))) with z = inversion
of x0 -- plus the full-pipeline wall-clock seconds.

Headline comparisons the table answers:
  (A) matched-N (both ~499 NFE): head-to-head quality + time.
  (B) matched-PSNR: the smallest ours-N whose mean PSNR >= the predecessor's
      -> speedup = predecessor_seconds / ours_seconds (our sampler needs far
      fewer steps for the same fidelity).

Encryption is lossless in latent space for BOTH schemes (S1 bit-exact; S2fn
~8e-7), so the end-to-end fidelity gap is sampler + conditioning + storage, not
the cipher.  To isolate the sampler/step effect alone (same storage, same scheme)
see scripts/rq4_redo_decode_sampler.py and deploy_dpm_sweep.py.

  smoke: scripts/run_cfg_ddim.sh python -m scripts.quality_speed_comparison --smoke
  full : scripts/run_cfg_ddim.sh python -m scripts.quality_speed_comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.classification_utility_anon import anonymize_and_recover
from scripts.classification_utility_legacy import legacy_anonymize_and_recover
from scripts.classification_utility_pilot import select_balanced
from scripts.deploy_s2_sweep import encrypt, decrypt
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.e2_legacy_wrapper import apply_legacy_key, legacy_invert
from scripts.exact_inv_sweep import CKPT, DPMSolverpp
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED, TOP

METRICS = ("psnr_db", "ssim", "mae", "mse", "cosine", "linf")
OURS_N = (499, 200, 100, 50, 25)
SMOKE_N = (100, 25)


def _img_metrics(x_rec: torch.Tensor, x0: torch.Tensor) -> dict:
    m = image_pair_metrics("", x_rec, x0)
    out = {k: round(float(m[f"_{k}"]), 5) for k in ("psnr_db", "ssim", "mae", "cosine", "linf")}
    out["mse"] = round(float(m["_rmse"]) ** 2, 8)
    return out


def _latent_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    fa, fb = a.double().flatten(), b.double().flatten()
    denom = float(fa.norm() * fb.norm())
    return round(float(torch.dot(fa, fb) / denom) if denom else 0.0, 9)


def _timed(fn, *a):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _agg(records: list[dict], keys) -> dict:
    out = {}
    for k in keys:
        v = np.array([r[k] for r in records], dtype=np.float64)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[k] = {"mean": round(float(v.mean()), 5), "ci95": round(float(ci), 5)}
    return out


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--per-class", type=int, default=50, help="patient-disjoint images per class")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/quality_speed"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ours_sweep = SMOKE_N if args.smoke else OURS_N

    per_class = 2 if args.smoke else args.per_class
    picks = select_balanced(per_class)
    got = {k: len(v) for k, v in picks.items()}
    if min(got.values()) < 1:
        raise SystemExit(f"insufficient images on disk: {got}")
    entries = [(pid, path) for pid, path in picks[0]] + [(pid, path) for pid, path in picks[1]]
    print(f"selected healthy(y0)={got[0]} effusion(y1)={got[1]} total={len(entries)}", flush=True)

    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device, backward_null=True)  # final deployment: null backward
    dpm = DPMSolverpp(rs)

    # warm-up (CUDA init / kernel compile) so the first timed pass is honest
    _b10 = schedule(10)
    rs.backward(rs.forward(preprocess(Path(entries[0][1]))[0].unsqueeze(0).to(device), _b10), _b10)

    boundaries = {n: schedule(n) for n in ours_sweep}
    pred_records, ours_records = [], {n: [] for n in ours_sweep}

    for pid, path in entries:
        x0 = preprocess(Path(path))[0].unsqueeze(0).to(device)

        # --- predecessor original: legacy dense DDIM + S1 + legacy storage ---
        (anon_p, rec_p), t_pred = _timed(
            legacy_anonymize_and_recover, diffusion, model_fn, x0,
            args.source_root, "S1", 500, "legacy", pid)
        zp = legacy_invert(diffusion, model_fn, x0, 500)
        zp_dec = apply_legacy_key(apply_legacy_key(zp, args.source_root, "S1", "anonymize"),
                                  args.source_root, "S1", "deanonymize")
        rec = _img_metrics(rec_p, x0)
        rec.update({"sample_id": pid, "pipeline_seconds": round(t_pred, 4),
                    "latent_crypto_cosine": _latent_cos(zp, zp_dec),
                    "evals_per_pass": 499})
        pred_records.append(rec)

        # --- ours final: DPM++ null + S2fn + range-preserving, per N ---
        for n in ours_sweep:
            b = boundaries[n]
            (anon_o, rec_o), t_o = _timed(anonymize_and_recover, dpm, x0, b, pid)
            zo = dpm.forward(x0, b)
            zc, orig = encrypt(zo, "S2fn", pid)
            zo_dec = decrypt(zc, "S2fn", pid, orig)
            rec = _img_metrics(rec_o, x0)
            rec.update({"sample_id": pid, "pipeline_seconds": round(t_o, 4),
                        "latent_crypto_cosine": _latent_cos(zo, zo_dec),
                        "evals_per_pass": int(len(b) - 1)})
            ours_records[n].append(rec)
        print(f"  {pid}: pred psnr={pred_records[-1]['psnr_db']:.2f} "
              f"ours@100 psnr={ours_records[ours_sweep[-1] if 100 not in ours_sweep else 100][-1]['psnr_db']:.2f}",
              flush=True)

    keys = (*METRICS, "pipeline_seconds", "latent_crypto_cosine")
    pred_agg = _agg(pred_records, keys)
    pred_agg["evals_per_pass"] = pred_records[0]["evals_per_pass"]
    ours_agg = {}
    for n in ours_sweep:
        a = _agg(ours_records[n], keys)
        a["evals_per_pass"] = ours_records[n][0]["evals_per_pass"]
        ours_agg[n] = a

    # (A) matched-N: ours at the largest N (closest to predecessor's ~499 NFE)
    matched_n = max(ours_sweep)
    matched = {
        "N": matched_n,
        "ours_psnr_db": ours_agg[matched_n]["psnr_db"]["mean"],
        "pred_psnr_db": pred_agg["psnr_db"]["mean"],
        "ours_seconds": ours_agg[matched_n]["pipeline_seconds"]["mean"],
        "pred_seconds": pred_agg["pipeline_seconds"]["mean"],
        "quality_delta_db": round(ours_agg[matched_n]["psnr_db"]["mean"] - pred_agg["psnr_db"]["mean"], 3),
        "time_ratio_pred_over_ours": round(
            pred_agg["pipeline_seconds"]["mean"] / ours_agg[matched_n]["pipeline_seconds"]["mean"], 2),
    }

    # (B) matched-PSNR: smallest ours N whose mean PSNR >= predecessor's mean PSNR
    target = pred_agg["psnr_db"]["mean"]
    reached = None
    for n in sorted(ours_sweep):  # ascending N (fewest steps first)
        if ours_agg[n]["psnr_db"]["mean"] >= target:
            reached = n
            break
    matched_psnr = {"predecessor_psnr_db": target}
    if reached is not None:
        matched_psnr.update({
            "ours_min_N_at_or_above": reached,
            "ours_psnr_db": ours_agg[reached]["psnr_db"]["mean"],
            "ours_seconds": ours_agg[reached]["pipeline_seconds"]["mean"],
            "pred_seconds": pred_agg["pipeline_seconds"]["mean"],
            "speedup_pred_over_ours": round(
                pred_agg["pipeline_seconds"]["mean"] / ours_agg[reached]["pipeline_seconds"]["mean"], 2),
        })
    else:
        matched_psnr["note"] = "no ours-N in the sweep reached the predecessor PSNR (unexpected)"

    payload = {
        "seed": SEED, "smoke": args.smoke, "n_images": len(entries),
        "ours": {"sampler": "DPM++(2M)", "conditioning": "null forward+backward (aligned)",
                 "scheme": "S2fn", "storage": "range_preserving_png/v1", "by_N": ours_agg},
        "predecessor": {"sampler": "legacy dense DDIM T=500", "conditioning": "null fwd / y=0 bwd",
                        "scheme": "S1", "storage": "legacy_png (min-max, range discarded)", **pred_agg},
        "matched_N": matched, "matched_PSNR": matched_psnr,
        "metrics_note": "MSE = RMSE^2; cosine/linf/latent_crypto_cosine from image_pair_metrics; "
                        "latent_crypto_cosine = cos(z, D(E(z))), z = inversion(x0) -> lossless-crypto check",
    }
    (args.out_dir / ("smoke.json" if args.smoke else "comparison.json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CSV: one row per (system, N)
    with (args.out_dir / ("smoke.csv" if args.smoke else "comparison.csv")).open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "N", "evals_per_pass", "psnr_db", "ssim", "mae", "mse",
                    "cosine", "linf", "latent_crypto_cosine", "pipeline_seconds"])
        w.writerow(["predecessor", 499, pred_agg["evals_per_pass"],
                    *[pred_agg[k]["mean"] for k in ("psnr_db", "ssim", "mae", "mse", "cosine", "linf",
                                                    "latent_crypto_cosine", "pipeline_seconds")]])
        for n in OURS_N if not args.smoke else SMOKE_N:
            if n not in ours_agg:
                continue
            a = ours_agg[n]
            w.writerow(["ours", n, a["evals_per_pass"],
                        *[a[k]["mean"] for k in ("psnr_db", "ssim", "mae", "mse", "cosine", "linf",
                                                 "latent_crypto_cosine", "pipeline_seconds")]])

    print("\n=== quality & speed: ours (DPM++ null) vs predecessor (legacy dense) ===")
    print(f"{'system':>12} {'N':>5} {'psnr':>7} {'ssim':>6} {'mae':>7} {'mse':>9} "
          f"{'cos':>8} {'lat-cos':>9} {'sec':>7}")
    pr = pred_agg
    print(f"{'predecessor':>12} {499:>5} {pr['psnr_db']['mean']:>7.2f} {pr['ssim']['mean']:>6.3f} "
          f"{pr['mae']['mean']:>7.4f} {pr['mse']['mean']:>9.2e} {pr['cosine']['mean']:>8.5f} "
          f"{pr['latent_crypto_cosine']['mean']:>9.6f} {pr['pipeline_seconds']['mean']:>7.2f}")
    for n in ours_sweep:
        a = ours_agg[n]
        print(f"{'ours':>12} {n:>5} {a['psnr_db']['mean']:>7.2f} {a['ssim']['mean']:>6.3f} "
              f"{a['mae']['mean']:>7.4f} {a['mse']['mean']:>9.2e} {a['cosine']['mean']:>8.5f} "
              f"{a['latent_crypto_cosine']['mean']:>9.6f} {a['pipeline_seconds']['mean']:>7.2f}")
    print(f"\n(A) matched-N={matched_n}: ours {matched['ours_psnr_db']:.2f} dB vs "
          f"predecessor {matched['pred_psnr_db']:.2f} dB (+{matched['quality_delta_db']} dB)")
    if reached is not None:
        print(f"(B) matched-PSNR {target:.2f} dB: ours reaches it at N={reached} "
              f"-> speedup {matched_psnr['speedup_pred_over_ours']}x "
              f"({matched_psnr['pred_seconds']:.2f}s -> {matched_psnr['ours_seconds']:.2f}s)")
    print(f"\nwrote {args.out_dir}/comparison.json + .csv")


if __name__ == "__main__":
    main()
