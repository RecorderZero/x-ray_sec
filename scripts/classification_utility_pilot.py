#!/usr/bin/env python3
"""Utility-via-classification pilot: is disease classification a better utility
axis than the (chance-level) lesion localization?

GMIA (refpaper/114) demonstrates utility by training a pathology classifier on
anonymized images and showing accuracy ~= real.  Localization on CheXlocalize was
chance for us AND the predecessor (see localization-chance-level), so we test the
classification axis instead.

First look (this script, --real-only): a linear probe (logistic regression on a
frozen CXR feature extractor) for the binary task the generator itself knows --
No Finding (healthy, y=0) vs Pleural Effusion (diseased, y=1) -- on REAL CheXpert
frontal images, patient-disjoint train/test.  If AUROC >> 0.5 the classification
signal is learnable here (unlike localization), which justifies extending the probe
to our anonymized / recovered images (the real utility-preservation experiment).

NOT a formal result: single split, fixed N, frozen ImageNet/CXR features (no
fine-tuning), no anonymization pipeline yet.

  scripts/run_cfg_ddim.sh python -m scripts.classification_utility_pilot --per-class 200
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.t2bb_m1_frozen_linkage import densenet_embeddings, rad_dino_embeddings, RAD_DINO_ID

SEED = 1911
DATASET = Path("dataset/chexpertchestxrays-u20210408")
CSV = DATASET / "train_visualCheXbert.csv"
TRAIN_BATCHES = ["CheXpert-v1.0 batch 2 (train 1)", "CheXpert-v1.0 batch 3 (train 2)",
                 "CheXpert-v1.0 batch 4 (train 3)"]


def resolve_image(csv_rel_path: str):
    marker = "train/"
    idx = csv_rel_path.find(marker)
    suffix = csv_rel_path[idx + len(marker):] if idx >= 0 else csv_rel_path
    for batch in TRAIN_BATCHES:
        cand = DATASET / batch / suffix
        if cand.exists():
            return cand
    return None


def patient_of(path: Path) -> str:
    for part in path.parts:
        if part.startswith("patient"):
            return part
    return str(path)


def select_balanced(per_class: int):
    """Patient-disjoint healthy (No Finding) vs diseased (Pleural Effusion) frontal."""
    picks = {0: [], 1: []}
    used_patients = set()
    with CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Frontal/Lateral") != "Frontal":
                continue
            nf = row.get("No Finding") == "1.0"
            pe = row.get("Pleural Effusion") == "1.0"
            label = 0 if (nf and not pe) else 1 if (pe and not nf) else None
            if label is None or len(picks[label]) >= per_class:
                continue
            path = resolve_image(row["Path"])
            if path is None:
                continue
            pid = patient_of(path)
            if pid in used_patients:            # one image per patient, disjoint classes
                continue
            picks[label].append((pid, path))
            used_patients.add(pid)
            if len(picks[0]) >= per_class and len(picks[1]) >= per_class:
                break
    return picks


def split_by_patient(items, test_frac, rng):
    pids = sorted({pid for pid, _ in items})
    rng.shuffle(pids)
    n_test = int(round(len(pids) * test_frac))
    test_pids = set(pids[:n_test])
    train = [(p, path) for p, path in items if p not in test_pids]
    test = [(p, path) for p, path in items if p in test_pids]
    return train, test


def load_images(entries):
    return [Image.open(path).convert("RGB") for _pid, path in entries]


def probe(feat_train, y_train, feat_test, y_test):
    scaler = StandardScaler().fit(feat_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(scaler.transform(feat_train), y_train)
    prob = clf.predict_proba(scaler.transform(feat_test))[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {"acc": round(float(accuracy_score(y_test, pred)), 4),
            "auroc": round(float(roc_auc_score(y_test, prob)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4)}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--encoders", nargs="+", default=["densenet121_imagenet", "rad_dino"])
    ap.add_argument("--real-only", action="store_true", default=True)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/classification_utility"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    picks = select_balanced(args.per_class)
    got = {k: len(v) for k, v in picks.items()}
    if min(got.values()) < 10:
        raise SystemExit(f"insufficient images resolved on disk: {got}")
    print(f"selected healthy(y0)={got[0]} effusion(y1)={got[1]}", flush=True)

    tr0, te0 = split_by_patient(picks[0], args.test_frac, rng)
    tr1, te1 = split_by_patient(picks[1], args.test_frac, rng)
    train = [(e, 0) for e in tr0] + [(e, 1) for e in tr1]
    test = [(e, 0) for e in te0] + [(e, 1) for e in te1]
    y_train = np.array([lab for _e, lab in train])
    y_test = np.array([lab for _e, lab in test])
    img_train = load_images([e for e, _ in train])
    img_test = load_images([e for e, _ in test])
    print(f"train n={len(train)} (y1={int(y_train.sum())}) | test n={len(test)} (y1={int(y_test.sum())})", flush=True)

    results = {}
    for enc in args.encoders:
        try:
            if enc == "densenet121_imagenet":
                ftr = densenet_embeddings(img_train, device, 16)
                fte = densenet_embeddings(img_test, device, 16)
            else:
                ftr = rad_dino_embeddings(img_train, device, 16, RAD_DINO_ID)
                fte = rad_dino_embeddings(img_test, device, 16, RAD_DINO_ID)
        except Exception as exc:  # e.g. rad-dino not cached
            results[enc] = {"error": str(exc)[:200]}
            print(f"{enc}: SKIPPED ({str(exc)[:80]})", flush=True)
            continue
        m = probe(ftr, y_train, fte, y_test)
        results[enc] = {"real": m, "feat_dim": int(ftr.shape[1])}
        print(f"{enc:>22}  REAL  acc={m['acc']:.3f}  auroc={m['auroc']:.3f}  f1={m['f1']:.3f}", flush=True)

    payload = {"seed": SEED, "task": "No Finding (y0) vs Pleural Effusion (y1), binary",
               "split": "patient-disjoint; one image per patient", "per_class": args.per_class,
               "train_n": len(train), "test_n": len(test), "probe": "StandardScaler + LogisticRegression",
               "note": "frozen features, no fine-tuning; real images only (no anonymization yet)",
               "results": results}
    (args.out_dir / "real_baseline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}/real_baseline.json")


if __name__ == "__main__":
    main()
