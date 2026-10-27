#!/usr/bin/env python3
"""Item 3 (AF-045): image-space linkage parity under DPM-Solver++ N=100.

Question: does switching the DEFENDER's sampler from legacy DDIM (T=500) to the
deployed DPM-Solver++(2M) N=100 make the released anonymous images MORE linkable?
Higher recon fidelity could in principle leak more residual identity; the frozen
linkage must be re-measured, not assumed (the degeneracy check's anon-vs-x0 5.91 dB
is only a coarse n=2 proxy, not a linkage attack).

Defender side uses DPM++ N=100 end-to-end: invert x0 -> z, encrypt per scheme,
generate anon = backward(E(z)); the published image uses the SAME representation
as the shipped pipeline (legacy PNG for P/S0/S1, private-range public-q PNG for S2),
so this is apples-to-apples with the committed legacy table (table_t2bb_m1_frozen).
Attacker side is a frozen pretrained encoder (sampler-agnostic): densenet121-imagenet
(+ rad-dino on the full run).  We score anon<->anon paired linkage (same patient,
two studies) and report Top-1 / AUC with label-permutation null.

A PASS parity result: P keeps signal, S0/S1/S2 stay ~random -- i.e. DPM++ does not
raise linkability vs the frozen legacy table.

  smoke: scripts/run_cfg_ddim.sh python -m scripts.t2bb_dpm_parity --smoke
  full : scripts/run_cfg_ddim.sh python -m scripts.t2bb_dpm_parity
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import secrets
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.respaced_ddim_sweep import Respaced, schedule, SEED
from scripts.exact_inv_sweep import DPMSolverpp, CKPT
from scripts.e2_legacy_wrapper import apply_legacy_key
from scripts.q8_pilot_runner import encode_legacy
from scripts.m1_storage import encode_private_range_png
from src.crypto.s2_fixed_norm import normalize_to_fixed_norm
from src.crypto.sot import transform
from scripts.t2bb_m1_frozen_linkage import (
    densenet_embeddings, rad_dino_embeddings, l2_normalize, evaluate_similarity,
)

PAST = Path("past/SourceCode")
N = 100
SMOKE_PAIRS = 6
SCHEMES_SMOKE = ("P", "S2")
SCHEMES_FULL = ("P", "S0", "S1", "S2")
S2_ROUNDS = 1
S2_PERM = False


def encrypt(z, scheme, key, nonce):
    if scheme == "P":
        return z
    if scheme in ("S0", "S1"):
        return apply_legacy_key(z, PAST, scheme, "anonymize")
    normalized, _ = normalize_to_fixed_norm(z)
    return transform(normalized, key, nonce, rounds=S2_ROUNDS, use_permutation=S2_PERM)


def published_image(anon: torch.Tensor, scheme: str) -> Image.Image:
    """Render anon exactly as the shipped pipeline publishes it, then read back."""
    if scheme in ("P", "S0", "S1"):
        payload, _ = encode_legacy(anon)
    else:
        payload, _ = encode_private_range_png(anon[0, 0].detach().float().cpu().numpy())
    with Image.open(io.BytesIO(payload)) as h:
        return h.convert("RGB").copy()


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/t2bb_dpm_parity"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    schemes = SCHEMES_SMOKE if args.smoke else SCHEMES_FULL
    encoders = args.encoders or (["densenet121_imagenet"] if args.smoke else ["densenet121_imagenet", "rad_dino"])

    _, diffusion, model_fn = create_runtime(PAST, CKPT, device)
    rs = Respaced(diffusion, model_fn, device)
    dpm = DPMSolverpp(rs)
    b = schedule(N)

    rows = list(csv.DictReader(Path("splits/t2bb_inner_v1.csv").open(encoding="utf-8")))
    by_pair: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["pair_role"]] = r
    pairs = sorted(by_pair)
    if args.smoke:
        pairs = pairs[:SMOKE_PAIRS]

    # Deterministic per-run S2 key (not persisted beyond fingerprint); per-image nonce.
    key = secrets.token_bytes(32)

    # Generate DPM++ N=100 anon for every (pair, role, scheme); collect published images.
    anon_imgs: dict[str, dict[str, list[Image.Image]]] = {s: {"gallery": [], "query": []} for s in schemes}
    for pid in pairs:
        for role in ("gallery", "query"):
            row = by_pair[pid][role]
            x0 = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
            z = dpm.forward(x0, b)
            for s in schemes:
                nonce = secrets.token_bytes(16)
                anon = dpm.backward(encrypt(z, s, key, nonce), b)
                anon_imgs[s][role].append(published_image(anon, s))
        print(f"generated pair {pid}", flush=True)

    results = []
    for enc in encoders:
        embed = (lambda ims: densenet_embeddings(ims, str(device), 8)) if enc == "densenet121_imagenet" \
            else (lambda ims: rad_dino_embeddings(ims, str(device), 8, "microsoft/rad-dino"))
        for s in schemes:
            g = l2_normalize(embed(anon_imgs[s]["gallery"]))
            q = l2_normalize(embed(anon_imgs[s]["query"]))
            sim = q @ g.T
            m = evaluate_similarity(sim, repeats=5000)
            results.append({"sampler": "DPM++", "N": N, "scheme": s, "encoder": enc,
                            "pairs": len(pairs), "top1": round(m["top1"], 4), "auc": round(m["auc"], 4),
                            "random_top1": round(m["random_top1"], 4), "null_auc_mean": round(m["null_auc_mean"], 4),
                            "auc_permutation_p": round(m["auc_permutation_p"], 4)})
            print(f"{enc:<22} {s:>3}  top1={m['top1']:.3f} auc={m['auc']:.3f} "
                  f"(random {m['random_top1']:.3f}, null_auc {m['null_auc_mean']:.3f})", flush=True)

    name = "smoke.json" if args.smoke else "parity.json"
    (args.out_dir / name).write_text(json.dumps(
        {"seed": SEED, "smoke": args.smoke, "sampler": "DPM++", "N": N, "pairs": len(pairs),
         "s2_key_sha256": __import__("hashlib").sha256(key).hexdigest(),
         "s2_rounds": S2_ROUNDS, "s2_use_permutation": S2_PERM, "results": results},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out_dir}/{name}")


if __name__ == "__main__":
    main()
