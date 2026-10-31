#!/usr/bin/env python3
"""RQ4 formal re-report at the deployment operating point (supersedes Q8_FORMAL fidelity).

The frozen ``paper_assets/q8_formal/table_fidelity.csv`` reports the reversibility
main result under the *superseded* pipeline: predecessor legacy dense DDIM T=500,
y=0 backward, legacy per-image min-max PNG (P marginal 37.46 dB).  The declared
deployment is now DPM-Solver++(2M) N=100, **null** backward (forward/backward
aligned; supersedes AF-018/021), range-preserving PNG.  This re-runs the fidelity
main table at that operating point on the *same* formal evaluation set and with the
*same* methodology as Q8 -- security_v1 N=200 patient-disjoint, patient-cluster
percentile bootstrap CI (seed 1911, 10000 resamples), PSNR/SSIM/cosine/L-inf/
RMSE/MAE + true LPIPS(alex) -- so the paper's headline fidelity matches the pipeline
it claims to deploy.  It does not touch the audited legacy q8_pilot_runner.

Schemes: P (identity-key reversibility ceiling / leak baseline) and S2fn (the
deployment cipher).  Full 4-pass pipeline (invert -> encrypt -> generate anon ->
range-preserving PNG -> invert -> decrypt -> generate recovery), recovery-vs-x0 as
the claim plus the single round-trip (anon vs x0).  Per-image pipeline wall-time is
recorded for the runtime line.

  smoke: scripts/run_cfg_ddim.sh python -m scripts.rq4_formal_redo --smoke
  formal: scripts/run_cfg_ddim.sh python -m scripts.rq4_formal_redo --evidence-role formal
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.m1_storage import range_preserving_png_handoff
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED
from scripts.exact_inv_sweep import DPMSolverpp, CKPT
from scripts.deploy_s2_sweep import encrypt, decrypt
from scripts import q8_metrics


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SCHEMES = ("P", "S2fn")
# image_pair_metrics returns these under a leading underscore
IMG_METRICS = ("psnr_db", "ssim", "cosine", "linf", "rmse", "mae")
STORAGE = "range_preserving_png/v1"


@torch.inference_mode()
def pipeline(sampler, x0, sid, scheme, b):
    """Deployment 4-pass: invert -> encrypt -> anon -> PNG -> invert -> decrypt -> recover."""
    z = sampler.forward(x0, b)
    zc, orig = encrypt(z, scheme, sid)
    anon = sampler.backward(zc, b)
    anon_png, _ = range_preserving_png_handoff(anon)
    zc2 = sampler.forward(anon_png, b)
    x_rec = sampler.backward(decrypt(zc2, scheme, sid, orig), b)
    return x_rec, anon


def _img_metrics(a, x0):
    m = image_pair_metrics("", a, x0)
    return {k: float(m[f"_{k}"]) for k in IMG_METRICS}


def _to_hw(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().reshape(256, 256)


def git_head() -> dict:
    def _run(args):
        try:
            return subprocess.check_output(args, cwd=Path(__file__).resolve().parents[1]).decode().strip()
        except Exception:
            return ""
    # dirty reflects TRACKED modifications only (AF-051): untracked artifacts
    # (audit/, refpaper/, out-of-tree outputs) are legitimately uncommitted and
    # must not flag a deterministic run as dirty.  `dirty_tracked_files` records
    # which tracked files were uncommitted at run time for provenance.
    tracked = _run(["git", "status", "--porcelain=v1", "--untracked-files=no"])
    return {"commit": _run(["git", "rev-parse", "HEAD"]),
            "dirty": bool(tracked),
            "dirty_tracked_files": tracked}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--evidence-role", choices=["smoke", "formal"], default="smoke")
    ap.add_argument("--smoke", action="store_true", help="alias for --evidence-role smoke --limit 2")
    ap.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    ap.add_argument("--limit", type=int, default=None, help="cap images (smoke); formal requires the full split")
    ap.add_argument("--steps", type=int, default=100, help="DPM++ N (deployment N=100)")
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--lpips", choices=["auto", "require", "off"], default="require")
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/rq4_formal"))
    args = ap.parse_args()
    if args.smoke:
        args.evidence_role = "smoke"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    if args.evidence_role == "smoke":
        rows = rows[: (args.limit or 2)]
    elif args.limit is not None:
        rows = rows[: args.limit]
    if args.evidence_role == "formal" and len(rows) != 200:
        raise SystemExit(f"formal requires the full 200-image security_v1 (got {len(rows)})")
    patients = [r["patient_id"] for r in rows]
    if len(set(patients)) != len(patients):
        raise SystemExit("patient IDs are not unique; the patient-cluster CI assumption is violated")

    # LPIPS gate (fail-closed on require); computed on CPU from clipped [0,1] images.
    lp_status = q8_metrics.lpips_status()
    if args.lpips == "require" and not lp_status["available"]:
        raise SystemExit(f"LPIPS required but unavailable: {lp_status['reason']}")
    lpips = q8_metrics.build_lpips_metric("cpu") if (args.lpips != "off" and lp_status["available"]) else None

    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device, backward_null=True)  # deployment: null backward
    dpm = DPMSolverpp(rs)
    b = schedule(args.steps)
    rs.backward(rs.forward(preprocess(Path(rows[0]["local_path"]))[0].unsqueeze(0).to(device), schedule(10)),
                schedule(10))  # warm-up (exclude from timing)

    records = []
    for i, r in enumerate(rows, 1):
        sid = r["sample_id"]
        x0 = preprocess(Path(r["local_path"]))[0].unsqueeze(0).to(device)
        x0_hw = _to_hw(x0)
        for scheme in args.schemes:
            torch.cuda.synchronize(); t0 = time.time()
            x_rec, anon = pipeline(dpm, x0, sid, scheme, b)
            torch.cuda.synchronize(); dt = time.time() - t0
            rec = _img_metrics(x_rec, x0)
            rec["single_roundtrip_psnr_db"] = float(image_pair_metrics("", anon, x0)["_psnr_db"])
            if lpips is not None:
                rec["lpips"] = lpips(x0_hw, _to_hw(x_rec))
            rec.update({"sample_id": sid, "patient_id": r["patient_id"], "label": r.get("label", ""),
                        "scheme": scheme, "pipeline_seconds": dt,
                        "finite": bool(torch.isfinite(x_rec).all())})
            records.append(rec)
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)

    # --- aggregate: one row per (scheme, metric), Q8 table_fidelity columns ---
    metric_keys = [*IMG_METRICS, "single_roundtrip_psnr_db"] + (["lpips"] if lpips is not None else [])
    fidelity_rows = []
    for scheme in args.schemes:
        sub = [r for r in records if r["scheme"] == scheme]
        clusters = [r["patient_id"] for r in sub]
        for key in metric_keys:
            s = q8_metrics.summarize([r[key] for r in sub], clusters)
            fidelity_rows.append({
                "method": scheme, "sampler": "DPM-Solver++(2M)", "steps": args.steps,
                "backward": "null", "storage": STORAGE,
                "comparison": "single_roundtrip" if key == "single_roundtrip_psnr_db" else "marginal",
                "space": "image", "reference": "x0",
                "metric": "psnr_db" if key == "single_roundtrip_psnr_db" else key,
                "n": s["n"], "mean": s["mean"], "ci95_low": s["ci95_low"], "ci95_high": s["ci95_high"],
                "ci_method": s["ci_method"], "median": s["median"], "p95": s["p95"],
                "min": s["min"], "max": s["max"],
            })
    # runtime line per scheme
    runtime_rows = [{"scheme": scheme, **q8_metrics.summarize(
        [r["pipeline_seconds"] for r in records if r["scheme"] == scheme],
        [r["patient_id"] for r in records if r["scheme"] == scheme]),
        "stage": "full_4pass_pipeline"} for scheme in args.schemes]

    def _write_csv(path, rows_):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows_)

    tag = "" if args.evidence_role == "formal" else f"_{args.evidence_role}_n{len(rows)}"
    _write_csv(args.out_dir / f"table_fidelity{tag}.csv", fidelity_rows)
    _write_csv(args.out_dir / f"table_runtime{tag}.csv", runtime_rows)
    _write_csv(args.out_dir / f"per_sample_long{tag}.csv", records)

    manifest = {
        "evidence_role": args.evidence_role, "seed": SEED, "n_images": len(rows),
        "supersedes": "paper_assets/q8_formal/table_fidelity.csv (legacy DDIM T=500, y=0, legacy PNG)",
        "operating_point": {"sampler": "DPM-Solver++(2M)", "steps": args.steps,
                            "backward": "null (forward/backward aligned; supersedes AF-018/021)",
                            "storage": STORAGE, "generation": "unconditional"},
        "schemes": args.schemes, "metrics": metric_keys,
        "ci_method": q8_metrics.CI_METHOD, "bootstrap_resamples": q8_metrics.BOOTSTRAP_RESAMPLES,
        "lpips": lp_status, "split": str(args.split), "split_sha256": sha256_file(args.split),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "git": git_head(),
    }
    (args.out_dir / f"analysis_manifest{tag}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== RQ4 formal re-report (DPM++ N={args.steps}, null, range-preserving; n={len(rows)}) ===")
    for row in fidelity_rows:
        if row["metric"] == "psnr_db" and row["comparison"] == "marginal":
            print(f"  {row['method']:5s} recon PSNR = {row['mean']:.3f} dB "
                  f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]  (n={row['n']})")
        if row["metric"] == "lpips":
            print(f"  {row['method']:5s} recon LPIPS = {row['mean']:.4f} "
                  f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]")
    print(f"wrote {args.out_dir}/table_fidelity{tag}.csv (+runtime, per_sample, manifest)")


if __name__ == "__main__":
    main()
