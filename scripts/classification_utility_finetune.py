#!/usr/bin/env python3
"""Item 3: trained-classifier utility (GMIA-style), not just a frozen linear probe.

GMIA (refpaper/114) demonstrates utility by TRAINING a DenseNet pathology
classifier.  Our frozen linear probe (classification_utility_anon.py) is a proxy;
this script fine-tunes a DenseNet-121 end-of-network (last dense block + head,
backbone frozen) on the binary No Finding vs Pleural Effusion task and evaluates
the SAME trained model on real / recovered / anon test images -- the trained-
classifier analogue of the preservation matrix.

Pipeline and render identical to classification_utility_anon (DPM++, S2fn).
K patient-disjoint splits -> mean & 95% CI.  GPU.

  smoke: scripts/run_cfg_ddim.sh python -m scripts.classification_utility_finetune \
             --per-class 6 --steps 50 --splits 2 --epochs 3
  full : scripts/run_cfg_ddim.sh python -m scripts.classification_utility_finetune \
             --per-class 200 --steps 100 --splits 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms

from scripts.classification_utility_anon import anonymize_and_recover, to_pil
from scripts.classification_utility_pilot import select_balanced
from scripts.deploy_s2_sweep import CKPT
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.exact_inv_sweep import DPMSolverpp
from scripts.respaced_ddim_sweep import Respaced, psnr, schedule, SEED

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), float(self.labels[i])


def build_model(device):
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.features.denseblock4.parameters():  # fine-tune last block + head only
        p.requires_grad = True
    for p in m.features.norm5.parameters():
        p.requires_grad = True
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    return m.to(device)


def evaluate(model, imgs, labels, device, bs=32):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            x = torch.stack([EVAL_TF(im) for im in imgs[i:i + bs]]).to(device)
            probs.append(torch.sigmoid(model(x).squeeze(1)).float().cpu().numpy())
    prob = np.concatenate(probs)
    if len(set(np.asarray(labels).tolist())) < 2:   # guard tiny one-class val sets
        return float("nan")
    return float(roc_auc_score(labels, prob))


def finetune_once(train_imgs, train_lab, val_imgs, val_lab, device, epochs, patience):
    model = build_model(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(ImgDS(train_imgs, train_lab, TRAIN_TF),
                                         batch_size=16, shuffle=True, num_workers=2, drop_last=False)
    best_auc, best_state, bad = -1.0, None, 0
    for _ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x).squeeze(1), y).backward()
            opt.step()
        va = evaluate(model, val_imgs, val_lab, device)
        if va > best_auc:
            best_auc, best_state, bad = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.inference_mode()
def generate(entries, sampler, b, device):
    real, anon, rec, rpsnr, apsnr = [], [], [], [], []
    t0 = time.perf_counter()
    for i, (pid, path, _lab) in enumerate(entries):
        x0 = preprocess(Path(path))[0].unsqueeze(0).to(device)
        a, r = anonymize_and_recover(sampler, x0, b, pid)
        real.append(to_pil(x0[0, 0].float().cpu().numpy()))
        anon.append(to_pil(a[0, 0].float().cpu().numpy()))
        rec.append(to_pil(r[0, 0].float().cpu().numpy()))
        rpsnr.append(psnr(r, x0)); apsnr.append(psnr(a, x0))
        if (i + 1) % 20 == 0 or i + 1 == len(entries):
            print(f"  generated {i + 1}/{len(entries)} recon~{np.mean(rpsnr):.1f}dB "
                  f"({(time.perf_counter() - t0) / (i + 1):.2f}s/img)", flush=True)
    return real, anon, rec, float(np.mean(rpsnr)), float(np.mean(apsnr))


def make_split(idx_by_class, seed, test_frac, val_frac=0.15):
    """Stratified per-class split so train/val/test each hold both classes."""
    g = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in (0, 1):
        perm = [int(i) for i in g.permutation(idx_by_class[c])]
        n_test = max(1, int(round(len(perm) * test_frac)))
        n_val = max(1, int(round((len(perm) - n_test) * val_frac)))
        test += perm[:n_test]
        val += perm[n_test:n_test + n_val]
        train += perm[n_test + n_val:]
    return np.array(train), np.array(val), np.array(sorted(test))


def summarize(vals):
    a = np.asarray(vals, float)
    mean = float(a.mean()); std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return round(mean, 4), round(std, 4), round(ci, 4)


@torch.inference_mode(False)
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/classification_utility"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")

    picks = select_balanced(args.per_class)
    if min(len(v) for v in picks.values()) < 4:
        raise SystemExit(f"insufficient images: {{k: len(v) for k,v in picks.items()}}")
    entries = [(pid, path, 0) for pid, path in picks[0]] + [(pid, path, 1) for pid, path in picks[1]]
    labels = np.array([lab for _p, _path, lab in entries])
    idx_by_class = {c: np.where(labels == c)[0] for c in (0, 1)}
    print(f"n_images={len(entries)} (y1={int(labels.sum())})", flush=True)

    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    sampler = DPMSolverpp(Respaced(diffusion, model_fn, device))
    b = schedule(args.steps)
    real, anon, rec, recon_db, anon_db = generate(entries, sampler, b, device)
    img_by_type = {"real": real, "anon": anon, "recovered": rec}

    conds = [("real", "real"), ("real", "recovered"), ("real", "anon"), ("anon", "anon")]
    collected = {f"train_{a}__test_{b_}": [] for a, b_ in conds}
    for k in range(args.splits):
        tr, va, te = make_split(idx_by_class, SEED + k, args.test_frac)
        y_tr, y_va, y_te = labels[tr], labels[va], labels[te]
        models_cache = {}
        for tr_type, te_type in conds:
            if tr_type not in models_cache:
                m = finetune_once([img_by_type[tr_type][i] for i in tr], y_tr,
                                  [img_by_type[tr_type][i] for i in va], y_va, device, args.epochs, args.patience)
                models_cache[tr_type] = m
            auc = evaluate(models_cache[tr_type], [img_by_type[te_type][i] for i in te], y_te, device)
            collected[f"train_{tr_type}__test_{te_type}"].append(auc)
            print(f"  split {k} train {tr_type:>9} -> test {te_type:>9}  auroc={auc:.3f}", flush=True)

    results = {}
    for key, vals in collected.items():
        m, s, ci = summarize(vals)
        results[key] = {"auroc": m, "auroc_std": s, "auroc_ci95": ci, "n_splits": args.splits}
        print(f"{key:<34} AUROC {m:.3f} (95%CI ±{ci:.3f})", flush=True)

    payload = {"seed": SEED, "design": "A: No Finding vs Pleural Effusion, binary",
               "classifier": "DenseNet-121 fine-tune (denseblock4 + head; backbone frozen)",
               "sampler": "dpm++", "steps": args.steps, "scheme": "S2fn", "per_class": args.per_class,
               "n_images": len(entries), "n_splits": args.splits, "epochs": args.epochs,
               "recon_psnr_vs_x0_db": round(recon_db, 3), "anon_psnr_vs_x0_db": round(anon_db, 3),
               "results": {"densenet121_finetune": {"matrix": results}}}
    out = args.out_dir / f"finetune_utility_dpm++_n{args.per_class}_s{args.steps}_k{args.splits}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
