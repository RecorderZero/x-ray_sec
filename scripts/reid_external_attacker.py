#!/usr/bin/env python3
"""External re-identification attacker: Packhäuser-style Siamese ResNet-50.

License-clean REIMPLEMENTATION (architecture only, from Packhäuser et al. 2022,
Sci Rep; refpaper 111) -- their repo is GPL-3.0 so no code is vendored. Trained
from an ImageNet backbone on OUR CheXpert shadow split (splits/t2bb_shadow_v1),
not their ChestX-ray14 checkpoint.

Purpose (reports/method_comparison.md 3): upgrade the T2-BB inner-risk attacker
from a generic ImageNet encoder (DenseNet121, Top-1 0.55 / AUC 0.81) to a
dedicated CXR patient-verification architecture, then measure same-patient
linkage on the disjoint victim split under:
  plaintext              -- strong-attacker ceiling on raw CXR
  S0/S1/S2 ciphertext    -- attacker_view of the anonymized latent (reuse T2-BB)

Arch per paper: Siamese, ResNet-50 (ImageNet), 256x256, 3ch, ImageNet norm,
embedding 128, verification head on |e_a - e_b| with BCEWithLogitsLoss.

  scripts/run_cfg_ddim.sh python -m scripts.reid_external_attacker \
    --shadow-split splits/t2bb_shadow_v1.csv --max-patients 1000 \
    --victim-split splits/t2bb_inner_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from scripts.t2bb_inner_linkage import encrypt, attacker_view, evaluate, IMG  # noqa: E402

SEED = 1911
SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_gray_u8(path: str) -> np.ndarray:
    with Image.open(path) as im:
        g = im.convert("L").resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(g, dtype=np.uint8)


def u8_to_input(a: np.ndarray) -> torch.Tensor:
    return _TF(Image.fromarray(a, mode="L"))


def arr_to_input(a: np.ndarray) -> torch.Tensor:
    lo, hi = float(a.min()), float(a.max())
    disp = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    return _TF(Image.fromarray(np.rint(disp * 255).astype(np.uint8), mode="L"))


class SiameseReID(nn.Module):
    def __init__(self, emb: int = 128):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        bb.fc = nn.Linear(bb.fc.in_features, emb)
        self.backbone = bb
        self.head = nn.Linear(emb, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def score(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return self.head((ea - eb).abs()).squeeze(-1)


def read_split(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def groups_by_patient(rows: list[dict]) -> dict[str, list[int]]:
    g: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        g.setdefault(r["pair_id"], []).append(i)
    return g


def sample_pairs(groups: dict[str, list[int]], rng: np.random.Generator):
    pats = sorted(groups)
    pos = [(groups[p][0], groups[p][1], 1.0) for p in pats if len(groups[p]) >= 2]
    neg = []
    for _ in range(len(pos)):
        a, b = rng.choice(len(pats), size=2, replace=False)
        ia = groups[pats[a]][rng.integers(len(groups[pats[a]]))]
        ib = groups[pats[b]][rng.integers(len(groups[pats[b]]))]
        neg.append((ia, ib, 0.0))
    pairs = pos + neg
    rng.shuffle(pairs)
    return pairs


def batched_embed(net, inputs: list[torch.Tensor], device: str, bs: int = 64) -> torch.Tensor:
    outs = []
    for i in range(0, len(inputs), bs):
        x = torch.stack(inputs[i:i + bs]).to(device)
        outs.append(net.embed(x))
    return torch.cat(outs, 0)


def pair_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    allv = np.concatenate([scores_pos, scores_neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    r = ranks[:len(scores_pos)].sum()
    return float((r - len(scores_pos) * (len(scores_pos) + 1) / 2) / (len(scores_pos) * len(scores_neg)))


def train(net, u8, groups, device, epochs, patience, rng):
    pats = sorted(groups)
    val_pats = set(pats[-40:]) if len(pats) > 80 else set()
    tr_groups = {p: v for p, v in groups.items() if p not in val_pats}
    va_groups = {p: v for p, v in groups.items() if p in val_pats}
    inputs = [u8_to_input(a) for a in u8]  # precompute 3ch tensors (held on CPU)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        pairs = sample_pairs(tr_groups, rng)
        tot = 0.0
        for i in range(0, len(pairs), 32):
            batch = pairs[i:i + 32]
            xa = torch.stack([inputs[p[0]] for p in batch]).to(device)
            xb = torch.stack([inputs[p[1]] for p in batch]).to(device)
            lab = torch.tensor([p[2] for p in batch], device=device)
            logit = net.score(net.embed(xa), net.embed(xb))
            loss = bce(logit, lab)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        # val AUC (same/diff pairs on held-out shadow patients)
        if va_groups:
            net.eval()
            with torch.no_grad():
                vp = sample_pairs(va_groups, np.random.default_rng(SEED))
                sc, lb = [], []
                for i in range(0, len(vp), 64):
                    batch = vp[i:i + 64]
                    ea = net.embed(torch.stack([inputs[p[0]] for p in batch]).to(device))
                    eb = net.embed(torch.stack([inputs[p[1]] for p in batch]).to(device))
                    sc.extend(torch.sigmoid(net.score(ea, eb)).cpu().numpy())
                    lb.extend([p[2] for p in batch])
            sc, lb = np.array(sc), np.array(lb)
            vauc = pair_auc(sc[lb == 1], sc[lb == 0])
        else:
            vauc = float("nan")
        print(f"  ep{ep:03d} loss={tot/max(1,len(pairs)//32):.4f} val_pair_auc={vauc:.3f}", flush=True)
        if va_groups:
            if vauc > best_auc:
                best_auc, best_state, bad = vauc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop @ep{ep} best_val_auc={best_auc:.3f}", flush=True)
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    return best_auc


def score_matrix(net, inputs: list[torch.Tensor], device: str) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        emb = batched_embed(net, inputs, device)  # (N, D)
        n = emb.shape[0]
        diff = (emb[:, None, :] - emb[None, :, :]).abs().reshape(n * n, -1)
        s = torch.sigmoid(net.head(diff)).reshape(n, n)
    return s.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--shadow-split", type=Path, default=ROOT / "splits/t2bb_shadow_v1.csv")
    ap.add_argument("--victim-split", type=Path, default=ROOT / "splits/t2bb_inner_v1.csv")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/t2bb_latent_cache")
    ap.add_argument("--max-patients", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--schemes", nargs="+", default=["S0", "S1", "S2"])
    ap.add_argument("--storage", default="range_preserving")
    ap.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_reid_external.csv")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_force_flag(ap)
    args = ap.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)
    torch.manual_seed(SEED)
    device = args.device

    # --- shadow training images (plaintext CheXpert) ---
    shadow = read_split(args.shadow_split)[: args.max_patients * 2]
    t0 = time.time()
    s_u8 = [load_gray_u8(r["local_path"]) for r in shadow]
    s_groups = groups_by_patient(shadow)
    print(f"shadow: {len(shadow)} imgs / {len(s_groups)} patients loaded in {time.time()-t0:.0f}s", flush=True)

    net = SiameseReID().to(device)
    best_val = train(net, s_u8, s_groups, device, args.epochs, args.patience, np.random.default_rng(SEED))

    # --- victim eval ---
    victim = read_split(args.victim_split)
    v_pid = [r["pair_id"] for r in victim]
    v_role = [r["pair_role"] for r in victim]
    rng = lambda: np.random.default_rng(SEED)
    results = []

    # (a) plaintext ceiling
    pt_inputs = [u8_to_input(load_gray_u8(r["local_path"])) for r in victim]
    res = evaluate(score_matrix(net, pt_inputs, device), v_pid, v_role, rng())
    res.update(condition="plaintext", scheme="-", storage="-")
    results.append(res)
    print(f"plaintext   Top-1={res['top1']:.3f} AUC={res['auc']:.3f}", flush=True)

    # (b) S0/S1/S2 ciphertext attacker_view (reuse cached latents + T2-BB cipher)
    latents = [torch.from_numpy(np.load(args.cache_dir / "latents" / f"{r['file_sha256']}.npy"))
               .reshape(1, 1, IMG, IMG) for r in victim]
    for scheme in args.schemes:
        views = [attacker_view(encrypt(z, scheme, r["sample_id"])[0], args.storage)
                 for z, r in zip(latents, victim)]
        inputs = [arr_to_input(v) for v in views]
        res = evaluate(score_matrix(net, inputs, device), v_pid, v_role, rng())
        res.update(condition=f"{scheme}_{args.storage}", scheme=scheme, storage=args.storage)
        results.append(res)
        print(f"{scheme}_{args.storage}  Top-1={res['top1']:.3f} AUC={res['auc']:.3f}", flush=True)

    cols = ["condition", "scheme", "storage", "top1", "top5", "auc", "neg_control_top1", "random_top1"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, lineterminator="\n"); w.writeheader()
        for r in results:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in cols})
    summary = {
        "schema_version": 1, "task_id": "REID-external", "seed": SEED,
        "architecture": "Siamese ResNet-50 (ImageNet), emb128, |ea-eb| head, BCE (Packhauser 2022 arch, reimpl)",
        "trained_on": f"{args.shadow_split.name} first {args.max_patients} patients (CheXpert)",
        "shadow_val_pair_auc": round(best_val, 4),
        "note": ("License-clean reimplementation of Packhauser et al. 2022 (refpaper 111) verification "
                 "architecture, trained on CheXpert shadow patients (NOT their ChestX-ray14 checkpoint). "
                 "Published ceilings for context: their verification AUC 0.994 (ChestX-ray14), 0.987 "
                 "(CheXpert cross-dataset). plaintext = strong-attacker ceiling here; ciphertext conditions "
                 "test whether the anonymized latent view defeats this stronger attacker. S2 expected ~random."),
        "results": results,
    }
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (shadow_val_pair_auc={best_val:.3f})", flush=True)


if __name__ == "__main__":
    main()
