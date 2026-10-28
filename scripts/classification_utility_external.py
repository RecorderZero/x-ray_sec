#!/usr/bin/env python3
"""External-dataset generalization of the classification-utility probe.

Question (user, 2026-10-28): we trained the "train real" classifier on CheXpert;
does it STILL work on an external dataset? This script takes the SAME frozen-feature
probe (StandardScaler + LogisticRegression) trained on CheXpert real and evaluates it
on NIH ChestX-ray14 real images -- a genuine cross-dataset test (different hospital,
scanner, population), with NO retraining. No diffusion / GPU anonymization here: this
is purely "does the utility classifier transfer", the external-validity check the main
experiment (same-dataset train->recovered) cannot give.

Design A labels map cleanly across datasets: NIH "No Finding" = y0, NIH "Effusion"
(= pleural effusion) present = y1. Both datasets go through the IDENTICAL e1 preprocess
and [0,1] 256->grayscale-RGB render, so the only shift is the dataset itself.

Reported per encoder (DenseNet-121 ImageNet, rad-dino):
  CheXpert->CheXpert (held-out)   in-domain reference (= our main real->real)
  CheXpert->NIH                   cross-dataset generalization (the answer)
Expect some drop from domain shift; read the gap, not just the absolute.

  scripts/run_cfg_ddim.sh python -m scripts.classification_utility_external \
      --per-class 200 --nih-root dataset/nih-cxr14
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from scripts.classification_utility_anon import probe, to_pil
from scripts.classification_utility_pilot import select_balanced
from scripts.e1_ddim_runner import preprocess
from scripts.respaced_ddim_sweep import SEED


def nih_select(nih_root: Path, per_class: int, seed: int):
    """Patient-disjoint balanced NIH subset: y0 = 'No Finding', y1 = 'Effusion' present.

    Only images actually present on disk (the downloaded archive subset) are used.
    """
    csv_path = nih_root / "Data_Entry_2017_v2020.csv"
    on_disk = {p.name: p for p in nih_root.rglob("*.png")}
    rng = np.random.default_rng(seed)
    by_class_patient: dict[int, dict[str, Path]] = {0: {}, 1: {}}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            name = row["Image Index"]
            path = on_disk.get(name)
            if path is None:
                continue
            labels = set(row["Finding Labels"].split("|"))
            pid = row["Patient ID"]
            if row["Finding Labels"] == "No Finding":
                by_class_patient[0].setdefault(pid, path)      # one image per patient
            elif "Effusion" in labels:
                by_class_patient[1].setdefault(pid, path)
    picks = {}
    for c in (0, 1):
        pids = list(by_class_patient[c])
        rng.shuffle(pids)
        picks[c] = [(pid, by_class_patient[c][pid]) for pid in pids[:per_class]]
    return picks


def embed(image_sets, encoder, device):
    from scripts.classification_utility_anon import (RAD_DINO_ID, densenet_embeddings,
                                                     rad_dino_embeddings)
    if encoder == "densenet121_imagenet":
        return {k: densenet_embeddings(v, device, 16) for k, v in image_sets.items()}
    return {k: rad_dino_embeddings(v, device, 16, RAD_DINO_ID) for k, v in image_sets.items()}


def render_set(entries):
    imgs, labels = [], []
    for path, lab in entries:
        arr = preprocess(Path(path))[0][0].numpy()      # [256,256] in [0,1]
        imgs.append(to_pil(arr)); labels.append(lab)
    return imgs, np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.3, help="CheXpert held-out fraction for in-domain ref")
    ap.add_argument("--nih-root", type=Path, default=Path("dataset/nih-cxr14"))
    ap.add_argument("--encoders", nargs="+", default=["densenet121_imagenet", "rad_dino"])
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/classification_utility"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CheXpert real (our training domain)
    cx = select_balanced(args.per_class)
    cx_entries = [(path, 0) for _pid, path in cx[0]] + [(path, 1) for _pid, path in cx[1]]
    cx_imgs, cx_lab = render_set(cx_entries)

    # NIH real (external)
    nih = nih_select(args.nih_root, args.per_class, SEED)
    nih_got = {k: len(v) for k, v in nih.items()}
    if min(nih_got.values()) < 10:
        raise SystemExit(f"too few NIH images on disk: {nih_got} (download more archives?)")
    nih_entries = [(path, 0) for _pid, path in nih[0]] + [(path, 1) for _pid, path in nih[1]]
    nih_imgs, nih_lab = render_set(nih_entries)
    print(f"CheXpert n={len(cx_lab)} (y1={int(cx_lab.sum())}) | NIH n={len(nih_lab)} (y1={int(nih_lab.sum())})", flush=True)

    # CheXpert in-domain held-out split (patient-disjoint: one image per patient already)
    rng = np.random.default_rng(SEED)
    idx_by_c = {c: np.where(cx_lab == c)[0] for c in (0, 1)}
    te = []
    for c in (0, 1):
        perm = rng.permutation(idx_by_c[c])
        te.extend(int(i) for i in perm[:max(1, int(round(len(perm) * args.test_frac)))])
    te = np.array(sorted(set(te))); tr = np.array([i for i in range(len(cx_lab)) if i not in set(te.tolist())])

    results = {}
    for enc in args.encoders:
        try:
            feats = embed({"cx": cx_imgs, "nih": nih_imgs}, enc, device)
        except Exception as exc:
            results[enc] = {"error": str(exc)[:200]}
            print(f"{enc}: SKIPPED ({str(exc)[:80]})", flush=True)
            continue
        cx_feat, nih_feat = feats["cx"], feats["nih"]
        in_domain = probe(cx_feat[tr], cx_lab[tr], cx_feat[te], cx_lab[te])    # CheXpert->CheXpert held-out
        external = probe(cx_feat, cx_lab, nih_feat, nih_lab)                   # CheXpert(all)->NIH
        results[enc] = {"chexpert_to_chexpert_heldout": in_domain, "chexpert_to_nih": external}
        print(f"{enc:>22}  CheXpert->CheXpert {in_domain['auroc']:.3f}  |  CheXpert->NIH {external['auroc']:.3f} "
              f"(acc {external['acc']:.3f})", flush=True)

    payload = {"seed": SEED, "design": "A: No Finding (y0) vs Pleural Effusion/Effusion (y1), binary",
               "task": "external-dataset generalization of the train-real utility probe (no retraining)",
               "train_domain": "CheXpert (frontal, e1 preprocess)", "external_domain": "NIH ChestX-ray14",
               "per_class": args.per_class, "chexpert_n": len(cx_lab), "nih_n": len(nih_lab),
               "nih_available": nih_got, "probe": "StandardScaler + LogisticRegression, frozen features",
               "results": results}
    out = args.out_dir / f"external_utility_nih_n{args.per_class}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
