#!/usr/bin/env python3
"""T2-BB inner-risk, Step 1: image-space positive control.

Does a same-patient biometric signal exist in the PLAINTEXT chest X-rays BEFORE
any inversion/encryption? If an off-the-shelf encoder cannot even link two
frontal CXRs of the same patient above chance, no downstream (encrypted) result
would be interpretable (reports/t2bb_inner_linkage_protocol.md 7.1).

Protocol: gallery = the 20 'gallery'-role images (one per patient), query = the
20 'query'-role images. For each query, rank the gallery by cosine similarity of
L2-normalised embeddings; Top-1 hit = same patient's gallery ranked first
(random baseline 1/20 = 0.05). Verification AUC over all 20x20 query x gallery
pairs (20 same-patient positives, 380 negatives). Negative control: query->patient
labels permuted (seed 1911).

Encoders (off-the-shelf, no CXR-specific training): DenseNet121 (ImageNet, the
CXR-standard backbone) and AlexNet (ImageNet), plus a raw-downsampled-pixel
cosine baseline for context. CPU/light-GPU, no model inversion.

  scripts/run_cfg_ddim.sh python -m scripts.t2bb_step1_image_control \
    --split splits/t2bb_inner_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

SEED = 1911
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_split(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preprocess(path: str, size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    with Image.open(path) as image:
        return tf(image.convert("L"))


@torch.no_grad()
def embed_densenet(paths: list[str], device: str) -> np.ndarray:
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).eval().to(device)
    out = []
    for p in paths:
        x = preprocess(p, 224).unsqueeze(0).to(device)
        f = F.relu(m.features(x))
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        out.append(f.cpu().numpy()[0])
    return np.asarray(out)


@torch.no_grad()
def embed_alexnet(paths: list[str], device: str) -> np.ndarray:
    m = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).eval().to(device)
    out = []
    for p in paths:
        x = preprocess(p, 224).unsqueeze(0).to(device)
        f = m.features(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        out.append(f.cpu().numpy()[0])
    return np.asarray(out)


def embed_rawpix(paths: list[str], device: str) -> np.ndarray:
    out = []
    for p in paths:
        x = preprocess(p, 32)  # downsampled grey (3 identical channels)
        out.append(x[0].flatten().numpy())
    return np.asarray(out)


def l2norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score(pos) > score(neg))."""
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def evaluate(emb: np.ndarray, pair_ids: list[str], roles: list[str], rng) -> dict:
    g_idx = [i for i, r in enumerate(roles) if r == "gallery"]
    q_idx = [i for i, r in enumerate(roles) if r == "query"]
    g_emb = l2norm(emb[g_idx]); q_emb = l2norm(emb[q_idx])
    g_pid = [pair_ids[i] for i in g_idx]; q_pid = [pair_ids[i] for i in q_idx]
    sim = q_emb @ g_emb.T  # (Q, G) cosine
    # retrieval Top-1 / Top-5
    top1 = top5 = 0
    for qi, pid in enumerate(q_pid):
        order = np.argsort(-sim[qi])
        true_g = g_pid.index(pid)
        rank = int(np.where(order == true_g)[0][0])
        top1 += rank == 0
        top5 += rank < 5
    # verification AUC
    pos, neg = [], []
    for qi, qp in enumerate(q_pid):
        for gi, gp in enumerate(g_pid):
            (pos if qp == gp else neg).append(sim[qi, gi])
    # negative control: permute query->pair labels
    perm = rng.permutation(len(q_pid))
    nc_top1 = 0
    for qi in range(len(q_pid)):
        order = np.argsort(-sim[qi])
        shuffled_true = g_pid.index(q_pid[perm[qi]])
        nc_top1 += int(order[0] == shuffled_true)
    n = len(q_pid)
    return {
        "n_query": n, "n_gallery": len(g_pid),
        "top1": top1 / n, "top5": top5 / n,
        "random_top1": 1.0 / len(g_pid),
        "auc": auc(np.asarray(pos), np.asarray(neg)),
        "neg_control_top1": nc_top1 / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--split", type=Path, default=ROOT / "splits/t2bb_inner_v1.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_t2bb_step1_image_control.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_force_flag(parser)
    args = parser.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    torch.manual_seed(SEED)
    rows = load_split(args.split)
    paths = [r["local_path"] for r in rows]
    pair_ids = [r["pair_id"] for r in rows]
    roles = [r["pair_role"] for r in rows]

    encoders = {
        "densenet121_imagenet": embed_densenet,
        "alexnet_imagenet": embed_alexnet,
        "rawpix32_cosine": embed_rawpix,
    }
    results = []
    for name, fn in encoders.items():
        emb = fn(paths, args.device)
        rng = np.random.default_rng(SEED)
        res = evaluate(emb, pair_ids, roles, rng)
        res["encoder"] = name
        res["embedding_dim"] = int(emb.shape[1])
        results.append(res)
        print(f"{name:24s} dim={emb.shape[1]:5d}  Top-1={res['top1']:.3f}  "
              f"Top-5={res['top5']:.3f}  AUC={res['auc']:.3f}  "
              f"neg_ctrl_Top-1={res['neg_control_top1']:.3f}  (random Top-1={res['random_top1']:.3f})")

    cols = ["encoder", "embedding_dim", "n_query", "n_gallery", "top1", "top5",
            "random_top1", "auc", "neg_control_top1"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow({c: r[c] for c in cols})
    summary = {
        "schema_version": 1, "task_id": "T2BB-Step1", "seed": SEED,
        "split": rel(args.split),
        "protocol": "reports/t2bb_inner_linkage_protocol.md 7.1",
        "note": ("Image-space positive control on plaintext CXRs (no inversion/encryption). "
                 "Top-1 >> random and high AUC => same-patient biometric signal exists at the "
                 "image level, so Step 2 (does it survive inversion+cipher?) is interpretable. "
                 "Encoders are off-the-shelf ImageNet, not CXR-trained, so this is a LOWER bound "
                 "on a black-box attacker; a dedicated re-id net (Packhauser 2022) would be stronger."),
        "results": results,
    }
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rel(args.out)} and {rel(json_out)}")


if __name__ == "__main__":
    main()
