#!/usr/bin/env python3
"""T2-BB inner-risk, Step 2: anon<->anon linkage on the stored anonymous images.

Pilot for reports/t2bb_inner_linkage_protocol.md. Attacker model T2-BB: only the
released anonymous PNGs, NO target model, NO key. Question: can two anonymous
images of the same patient be linked (inner risk / unlinkability)?

Per image (splits/t2bb_inner_v1.csv, 20 patients x {gallery, query}):
  x0 -> DDIM invert (NOISE_LEVEL=500) -> latent z            [GPU, ~8.5 s/img]
  encrypt z per scheme (CPU):
    S0  audited Rademacher sign key           (e2_legacy_wrapper.apply_legacy_key)
    S1  audited signed permutation            (e2_legacy_wrapper.apply_legacy_key)
    S2  fixed-norm + SOT, PER-IMAGE nonce      (normalize_to_fixed_norm + sot.transform)
  store (attacker view):
    minmax            per-image min-max 8-bit, range discarded (predecessor legacy
                      PNG; also S2's private-range attacker view)
    range_preserving  public range-preserving PNG, true lo/hi leak (storage axis)

Scorers on the attacker-visible array (no key, no model): abs, sorted_abs, norm,
densenet121 (ImageNet embedding of the stored image). Gallery = 20 gallery-role
anon images, query = 20 query-role; Top-1/Top-5 retrieval + verification AUC;
negative control = shuffled labels (seed 1911); random Top-1 = 1/20 = 0.05.

Positive control: same scorers on the PLAINTEXT latent z (does patient signal
survive inversion, before any cipher?). Decrypt sanity: encrypt->decrypt max_abs
per scheme (invertibility check).

  scripts/run_cfg_ddim.sh python -m scripts.t2bb_inner_linkage \
    --split splits/t2bb_inner_v1.csv --checkpoint <ckpt>
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
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
PAST = ROOT / "past" / "SourceCode"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAST))

from scripts.q11_localization_parity import build_model, make_model_fn, MODEL_PATH  # noqa: E402
from scripts.e2_legacy_wrapper import apply_legacy_key, legacy_invert  # noqa: E402
from scripts.m1_storage import (  # noqa: E402
    decode_private_png_attacker_view, decode_range_preserving_png,
    encode_private_range_png, encode_range_preserving_png,
)
from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from src.crypto.s2_fixed_norm import normalize_to_fixed_norm, restore_original_norm  # noqa: E402
from src.crypto.sot import inverse_transform, transform  # noqa: E402

SEED = 1911
IMG = 256
NOISE_LEVEL = 500
SCHEMES = ("S0", "S1", "S2")
STORAGES = ("minmax", "range_preserving")
SCORERS = ("cos", "abs", "sorted_abs", "norm", "densenet121")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- crypto
def encrypt(z: torch.Tensor, scheme: str, sample_id: str):
    """Return (z_ano, aux) for the deployed form of each scheme."""
    if scheme in ("S0", "S1"):
        return apply_legacy_key(z, PAST, scheme, "anonymize"), None
    if scheme == "S2":
        u, s = normalize_to_fixed_norm(z)
        nonce = nonce_for(f"t2bb-inner::{sample_id}")   # PER-IMAGE nonce
        return transform(u, FIXTURE_KEY, nonce, rounds=1, use_permutation=True), (s, nonce)
    raise ValueError(scheme)


def decrypt(z_ano: torch.Tensor, scheme: str, aux) -> torch.Tensor:
    if scheme in ("S0", "S1"):
        return apply_legacy_key(z_ano, PAST, scheme, "deanonymize")
    s, nonce = aux
    u = inverse_transform(z_ano, FIXTURE_KEY, nonce, rounds=1, use_permutation=True)
    return restore_original_norm(u, s)


# --------------------------------------------------------------------------- storage (attacker view)
def attacker_view(z_ano: torch.Tensor, storage: str) -> np.ndarray:
    img2d = z_ano.detach().reshape(IMG, IMG).cpu().numpy().astype(np.float32)
    if storage == "minmax":                       # range discarded (legacy / S2 private)
        payload, _ = encode_private_range_png(img2d)
        return decode_private_png_attacker_view(payload).astype(np.float32)
    if storage == "range_preserving":             # public PNG, true range leaks
        payload, _ = encode_range_preserving_png(img2d)
        return decode_range_preserving_png(payload).astype(np.float32)
    raise ValueError(storage)


# --------------------------------------------------------------------------- scorers -> similarity
def l2norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


_DENSENET = None


@torch.no_grad()
def densenet_embed(arrays: list[np.ndarray], device: str) -> np.ndarray:
    global _DENSENET
    if _DENSENET is None:
        _DENSENET = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).eval().to(device)
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    out = []
    for a in arrays:
        lo, hi = float(a.min()), float(a.max())
        disp = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        pil = Image.fromarray(np.rint(disp * 255).astype(np.uint8), mode="L")
        x = tf(pil).unsqueeze(0).to(device)
        f = F.relu(_DENSENET.features(x))
        out.append(F.adaptive_avg_pool2d(f, 1).flatten(1).cpu().numpy()[0])
    return np.asarray(out)


def similarity(arrays: list[np.ndarray], scorer: str, device: str) -> np.ndarray:
    flat = np.asarray([a.reshape(-1) for a in arrays], dtype=np.float64)
    if scorer == "cos":
        # raw SIGNED cosine. S0 (z_ano=D z) and S1 (z_ano=P D z) apply ONE
        # image-independent orthogonal map, so cos(z_ano_i,z_ano_j)=cos(z_i,z_j)
        # exactly -> this is the cipher's strongest linear transparency probe.
        return l2norm(flat) @ l2norm(flat).T
    if scorer == "abs":
        return l2norm(np.abs(flat)) @ l2norm(np.abs(flat)).T
    if scorer == "sorted_abs":
        s = np.sort(np.abs(flat), axis=1)
        return l2norm(s) @ l2norm(s).T
    if scorer == "norm":
        n = np.linalg.norm(flat, axis=1)
        return -np.abs(n[:, None] - n[None, :])        # closer norm -> higher score
    if scorer == "densenet121":
        e = l2norm(densenet_embed(arrays, device))
        return e @ e.T
    raise ValueError(scorer)


# --------------------------------------------------------------------------- linkage eval
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); avg = (csum - counts + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def evaluate(full_sim: np.ndarray, pid: list[str], roles: list[str], rng) -> dict:
    g = [i for i, r in enumerate(roles) if r == "gallery"]
    q = [i for i, r in enumerate(roles) if r == "query"]
    g_pid = [pid[i] for i in g]; q_pid = [pid[i] for i in q]
    sim = full_sim[np.ix_(q, g)]
    top1 = top5 = 0
    pos, neg = [], []
    for qi, qp in enumerate(q_pid):
        order = np.argsort(-sim[qi])
        true_g = g_pid.index(qp)
        rank = int(np.where(order == true_g)[0][0])
        top1 += rank == 0; top5 += rank < 5
        for gi, gp in enumerate(g_pid):
            (pos if qp == gp else neg).append(sim[qi, gi])
    perm = rng.permutation(len(q_pid))
    nc = sum(int(np.argsort(-sim[qi])[0] == g_pid.index(q_pid[perm[qi]])) for qi in range(len(q_pid)))
    n = len(q_pid)
    return {"top1": top1 / n, "top5": top5 / n, "auc": auc(np.asarray(pos), np.asarray(neg)),
            "neg_control_top1": nc / n, "random_top1": 1.0 / len(g_pid)}


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--split", type=Path, default=ROOT / "splits/t2bb_inner_v1.csv")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/t2bb_latent_cache",
                        help="reuse DDIM-inverted latents keyed by file_sha256 (t2bb_invert_cache)")
    parser.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_t2bb_inner_linkage.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_force_flag(parser)
    args = parser.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)

    torch.manual_seed(SEED)
    device = args.device
    with args.split.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pid = [r["pair_id"] for r in rows]
    roles = [r["pair_role"] for r in rows]

    from scripts.e1_ddim_runner import preprocess  # [0,1] security-pipeline loader

    # --- obtain latents: reuse cache (file_sha256) where present, invert the rest ---
    latents_dir = args.cache_dir / "latents"
    cached = {r["file_sha256"]: latents_dir / f"{r['file_sha256']}.npy" for r in rows}
    misses = [r for r in rows if not cached[r["file_sha256"]].exists()]
    t0 = time.time()
    if misses:
        model, diffusion, _ = build_model(device)
        model_fn = make_model_fn(model)
        latents_dir.mkdir(parents=True, exist_ok=True)
        for r in misses:
            x0 = preprocess(Path(r["local_path"]))[0].unsqueeze(0).to(device)
            z = legacy_invert(diffusion, model_fn, x0, NOISE_LEVEL)
            np.save(cached[r["file_sha256"]], z.reshape(IMG, IMG).cpu().numpy().astype(np.float32))
    latents = [torch.from_numpy(np.load(cached[r["file_sha256"]])).reshape(1, 1, IMG, IMG).to(device)
               for r in rows]
    invert_sec = time.time() - t0
    print(f"latents: {len(rows)} imgs, {len(rows) - len(misses)} from cache, "
          f"{len(misses)} inverted in {invert_sec:.1f}s")

    results = []
    rng = lambda: np.random.default_rng(SEED)

    # --- positive control: plaintext latent ---
    z_arrays = [z.reshape(IMG, IMG).cpu().numpy().astype(np.float32) for z in latents]
    for scorer in ("cos", "abs", "sorted_abs", "norm"):
        res = evaluate(similarity(z_arrays, scorer, device), pid, roles, rng())
        res.update(scheme="plaintext_latent", storage="none", scorer=scorer)
        results.append(res)

    # --- per scheme ---
    decrypt_maxabs = {}
    for scheme in SCHEMES:
        anon, aux_list = [], []
        for r, z in zip(rows, latents):
            za, aux = encrypt(z, scheme, r["sample_id"])
            anon.append(za); aux_list.append(aux)
        # decrypt sanity
        m = max(float((decrypt(za, scheme, aux) - z).abs().max())
                for za, aux, z in zip(anon, aux_list, latents))
        decrypt_maxabs[scheme] = m
        for storage in STORAGES:
            views = [attacker_view(za, storage) for za in anon]
            for scorer in SCORERS:
                res = evaluate(similarity(views, scorer, device), pid, roles, rng())
                res.update(scheme=scheme, storage=storage, scorer=scorer)
                results.append(res)
        print(f"{scheme}: decrypt max_abs={m:.2e}")

    cols = ["scheme", "storage", "scorer", "top1", "top5", "auc", "neg_control_top1", "random_top1"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in results:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in cols})

    summary = {
        "schema_version": 1, "task_id": "T2BB-Step2-pilot", "seed": SEED,
        "split": rel(args.split), "n_patients": len(rows) // 2, "n_images": len(rows),
        "checkpoint_sha256_prefix": None, "noise_level": NOISE_LEVEL,
        "inversion_seconds": round(invert_sec, 1),
        "inversion_sec_per_image": round(invert_sec / len(rows), 3),
        "decrypt_maxabs": {k: float(v) for k, v in decrypt_maxabs.items()},
        "random_top1": 1.0 / (len(rows) // 2),
        "protocol": "reports/t2bb_inner_linkage_protocol.md",
        "note": ("Inner-risk anon<->anon linkage pilot. Interpret only if the plaintext_latent "
                 "positive control is above random. S2 uses a PER-IMAGE nonce (its defense), so "
                 "near-random is expected by construction; the predecessor S0/S1 are the real target. "
                 "Off-the-shelf densenet121 is a black-box lower bound, not a trained re-id net."),
        "results": results,
    }
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rel(args.out)} and {rel(json_out)}")


if __name__ == "__main__":
    main()
