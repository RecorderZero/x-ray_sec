#!/usr/bin/env python3
"""AF-046 upgrade: multi-pathology detection-preservation via an INDEPENDENT referee.

Instead of (or alongside) our own frozen probe + the HF-fidelity proxy, score the
RECOVERED images with a third-party, pre-trained CXR classifier we never touched
(torchxrayvision DenseNet, published per-pathology AUC, zero retraining).  If an
independent clinical model gives ~the same per-pathology AUC on recovered as on
real -- over the SAME images and the SAME preprocessing -- the diagnostic signal
it relies on survived the anonymise->recover round-trip, for MANY pathologies
(not just the two CFG-conditioned classes).  This is an ML-referee proxy, not a
radiologist read; CheXpert labels carry uncertainty; whatever comes out is reported.

Operating point = deployment: DPM-Solver++(2M) N=100, S2fn, null backward,
range-preserving PNG, seed 1911, e1 render.  real = upper baseline, recovered =
the claim, anon = privacy lower bound -- all three through the identical referee
preprocessing so Delta = AUC(recovered) - AUC(real) isolates the recovery effect.

  scripts/run_cfg_ddim.sh python -m scripts.referee_pathology_check --per-sample 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
import torchvision
from sklearn.metrics import roc_auc_score

from scripts.classification_utility_pilot import CSV, patient_of, resolve_image
from scripts.classification_utility_anon import anonymize_and_recover
from scripts.deploy_s2_sweep import CKPT
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.exact_inv_sweep import DPMSolverpp
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED

# CheXpert label column -> torchxrayvision pathology name (only the overlap is scorable)
CHEX_TO_XRV = {
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Cardiomegaly": "Cardiomegaly",
    "Lung Opacity": "Lung Opacity",
    "Lung Lesion": "Lung Lesion",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",
    "Fracture": "Fracture",
}
WEIGHTS = ("densenet121-res224-all", "densenet121-res224-chex")


def select_frontal(per_sample: int, seed: int):
    """Patient-disjoint random frontal sample with full multi-label ground truth."""
    import csv as _csv
    rng = np.random.default_rng(seed)
    rows = []
    with CSV.open(newline="") as handle:
        for row in _csv.DictReader(handle):
            if row.get("Frontal/Lateral") == "Frontal":
                rows.append(row)
    rng.shuffle(rows)
    picks, used = [], set()
    for row in rows:
        path = resolve_image(row["Path"])
        if path is None or not Path(path).exists():
            continue
        pid = patient_of(Path(path))
        if pid in used:
            continue
        used.add(pid)
        picks.append((pid, path, row))
        if len(picks) >= per_sample:
            break
    return picks


def to_xrv_batch(arrays: list[np.ndarray]) -> torch.Tensor:
    """Each array is HxW float in [0,1] -> xrv input (N,1,224,224) in [-1024,1024]."""
    tfm = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                           xrv.datasets.XRayResizer(224)])
    out = []
    for a in arrays:
        # recovery can overshoot [0,1] by ~1e-3; the published PNG is clamped to the
        # display range anyway, so clip before xrv's strict [0,maxval] normalize.
        a = np.clip(a, 0.0, 1.0)
        img = xrv.datasets.normalize((a * 255.0).astype(np.float32), 255)  # -> [-1024,1024]
        out.append(torch.from_numpy(tfm(img[None, ...])))                  # (1,224,224)
    return torch.stack(out)


@torch.inference_mode()
def referee_scores(model, arrays: list[np.ndarray], device, batch: int = 32) -> np.ndarray:
    x = to_xrv_batch(arrays)
    outs = []
    for i in range(0, len(x), batch):
        outs.append(model(x[i:i + batch].to(device)).cpu().numpy())
    return np.concatenate(outs, 0)  # (N, n_pathologies), sigmoid probabilities


def gt_vector(rows, chex_col: str) -> np.ndarray:
    """1.0 -> positive; 0.0/blank -> negative; -1.0 (uncertain) -> NaN (masked)."""
    out = []
    for r in rows:
        v = r.get(chex_col, "")
        out.append(1.0 if v == "1.0" else np.nan if v == "-1.0" else 0.0)
    return np.array(out, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--per-sample", type=int, default=400)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--min-pos", type=int, default=10, help="min positives to report a pathology AUC")
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/referee_pathology"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (recovery pipeline)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")

    picks = select_frontal(args.per_sample, SEED)
    rows = [r for _p, _path, r in picks]
    print(f"selected {len(picks)} patient-disjoint frontal images", flush=True)

    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device, backward_null=True)  # deployment: null backward
    sampler = DPMSolverpp(rs)
    b = schedule(args.steps)

    real_imgs, rec_imgs, anon_imgs = [], [], []
    for i, (pid, path, _row) in enumerate(picks, 1):
        x0 = preprocess(Path(path))[0].unsqueeze(0).to(device)  # [1,1,256,256] in [0,1]
        anon, recovered = anonymize_and_recover(sampler, x0, b, pid)
        real_imgs.append(x0[0, 0].detach().float().cpu().numpy())
        rec_imgs.append(recovered[0, 0].detach().float().cpu().numpy())
        anon_imgs.append(anon[0, 0].detach().float().cpu().numpy())
        if i % 25 == 0 or i == len(picks):
            print(f"  recovered {i}/{len(picks)}", flush=True)

    report = {"config": {"operating_point": f"DPM++ N={args.steps} S2fn null range-preserving",
                         "per_sample": len(picks), "seed": SEED, "min_pos": args.min_pos},
              "models": {}}
    for w in WEIGHTS:
        model = xrv.models.DenseNet(weights=w).to(device).eval()
        path_idx = {p: j for j, p in enumerate(model.pathologies) if p}
        scores = {t: referee_scores(model, imgs, device)
                  for t, imgs in (("real", real_imgs), ("recovered", rec_imgs), ("anon", anon_imgs))}
        per_path = {}
        for chex_col, xrv_name in CHEX_TO_XRV.items():
            if xrv_name not in path_idx:
                continue
            j = path_idx[xrv_name]
            gt = gt_vector(rows, chex_col)
            mask = ~np.isnan(gt)
            y = gt[mask]
            n_pos = int(y.sum())
            if n_pos < args.min_pos or n_pos == len(y):
                per_path[xrv_name] = {"n_pos": n_pos, "n": int(mask.sum()), "note": "insufficient positives"}
                continue
            aucs = {t: round(float(roc_auc_score(y, scores[t][mask, j])), 4)
                    for t in ("real", "recovered", "anon")}
            per_path[xrv_name] = {"n_pos": n_pos, "n": int(mask.sum()),
                                  **{f"auc_{t}": aucs[t] for t in aucs},
                                  "delta_rec_minus_real": round(aucs["recovered"] - aucs["real"], 4)}
        scored = [p for p in per_path.values() if "auc_real" in p]
        macro = {t: round(float(np.mean([p[f"auc_{t}"] for p in scored])), 4)
                 for t in ("real", "recovered", "anon")} if scored else {}
        report["models"][w] = {"pathologies": model.pathologies, "per_pathology": per_path,
                               "macro_auc": macro,
                               "macro_delta_rec_minus_real": round(macro["recovered"] - macro["real"], 4) if macro else None}
        print(f"\n=== referee {w} (n={len(picks)}) ===", flush=True)
        for name, p in per_path.items():
            if "auc_real" in p:
                print(f"  {name:28s} n_pos={p['n_pos']:3d}  real={p['auc_real']:.3f} "
                      f"rec={p['auc_recovered']:.3f} anon={p['auc_anon']:.3f}  Δ={p['delta_rec_minus_real']:+.3f}")
        if macro:
            print(f"  {'MACRO':28s}            real={macro['real']:.3f} rec={macro['recovered']:.3f} "
                  f"anon={macro['anon']:.3f}  Δ={report['models'][w]['macro_delta_rec_minus_real']:+.3f}")

    out = args.out_dir / f"referee_pathology_n{len(picks)}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
