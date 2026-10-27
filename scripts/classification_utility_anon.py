#!/usr/bin/env python3
"""Utility-preservation via classification: does disease signal survive our
S2 anonymization, and does authorized recovery restore it? (GMIA-style, our
reversible pipeline.)

Design A (user-approved 2026-10-27): binary No Finding (healthy, y=0) vs
Pleural Effusion (diseased, y=1) -- exactly the two CFG classes the generator
was trained on; selection drops every image that is neither (see
classification_utility_pilot for the selection rationale and the real baseline).

For each selected (patient-disjoint) image we render THREE versions through the
IDENTICAL [0,1] 256x256 -> grayscale-RGB path, so differences are purely the
anonymization, not the rendering:
  real       = x0 (the pipeline input itself, e1-preprocessed)
  anon       = B(E(F(x0)))            -- the published, non-identifiable image
  recovered  = B(D(F(PNG(anon))))     -- after authorized key-based recovery
using the deployment-realistic aligned respaced DDIM + range-preserving PNG
hand-off, scheme S2fn (fixed-norm fidelity S2), identical to deploy_s2_sweep.

A frozen-feature linear probe (StandardScaler + LogisticRegression) is then run
across the utility matrix, per encoder:
  real -> real        learnable-task control on rendered reals
  real -> anon        does a REAL-trained classifier read disease off anon?
                      (low = anon hides the pathology from an external reader)
  real -> recovered   recovery fidelity for utility (should ~= real->real)
  anon -> anon        GMIA claim: can a usable classifier be TRAINED on anon?

NOT a formal result: single split, fixed N/steps, frozen (no fine-tuning),
one operating point.  Read jointly with recon/anon PSNR printed per run.

  # smoke (4 imgs, fast) then scale:
  scripts/run_cfg_ddim.sh python -m scripts.classification_utility_anon --per-class 2 --steps 100
  scripts/run_cfg_ddim.sh python -m scripts.classification_utility_anon --per-class 100 --steps 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.classification_utility_pilot import select_balanced, split_by_patient
from scripts.deploy_s2_sweep import CKPT, decrypt, encrypt
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.exact_inv_sweep import DPMSolverpp
from scripts.m1_storage import range_preserving_png_handoff
from scripts.respaced_ddim_sweep import Respaced, psnr, schedule, SEED
from scripts.t2bb_m1_frozen_linkage import RAD_DINO_ID, densenet_embeddings, rad_dino_embeddings


def to_pil(arr01: np.ndarray) -> Image.Image:
    """[0,1] HxW array -> 8-bit grayscale -> RGB, the shared render for all 3 types."""
    u8 = np.rint(np.clip(arr01, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(u8, mode="L").convert("RGB")


@torch.inference_mode()
def anonymize_and_recover(sampler, x0: torch.Tensor, boundaries, sid: str):
    """S2fn deployment pipeline: returns (anon_tensor, recovered_tensor).

    `sampler` exposes the aligned .forward/.backward interface shared by the
    respaced DDIM (`Respaced`) and DPM-Solver++ 2M (`DPMSolverpp`) samplers.
    """
    z = sampler.forward(x0, boundaries)
    zc, orig = encrypt(z, "S2fn", sid)
    anon = sampler.backward(zc, boundaries)
    anon_png, _ = range_preserving_png_handoff(anon)
    zc2 = sampler.forward(anon_png, boundaries)
    recovered = sampler.backward(decrypt(zc2, "S2fn", sid, orig), boundaries)
    for name, t in (("anon", anon), ("recovered", recovered)):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"non-finite {name} for {sid}")
    return anon, recovered


def probe(feat_train, y_train, feat_test, y_test) -> dict:
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
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100, help="boundaries (model evals = steps)")
    ap.add_argument("--sampler", choices=["dpm++", "ddim"], default="dpm++",
                    help="deployment sampler; dpm++ (DPM-Solver++ 2M) matches the current pipeline")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--encoders", nargs="+", default=["densenet121_imagenet", "rad_dino"])
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/classification_utility"))
    ap.add_argument("--save-examples", type=int, default=6, help="rows in the real|anon|recovered contact sheet")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (anonymization pipeline)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda:0")

    picks = select_balanced(args.per_class)
    got = {k: len(v) for k, v in picks.items()}
    if min(got.values()) < 2:
        raise SystemExit(f"insufficient images resolved on disk: {got}")
    print(f"selected healthy(y0)={got[0]} effusion(y1)={got[1]}", flush=True)

    tr0, te0 = split_by_patient(picks[0], args.test_frac, rng)
    tr1, te1 = split_by_patient(picks[1], args.test_frac, rng)
    train = [(pid, path, 0) for pid, path in tr0] + [(pid, path, 1) for pid, path in tr1]
    test = [(pid, path, 0) for pid, path in te0] + [(pid, path, 1) for pid, path in te1]
    entries = train + test
    train_n = len(train)
    labels = np.array([lab for _p, _path, lab in entries])
    y_train, y_test = labels[:train_n], labels[train_n:]
    print(f"train n={train_n} (y1={int(y_train.sum())}) | test n={len(test)} (y1={int(y_test.sum())})", flush=True)

    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device)
    sampler = DPMSolverpp(rs) if args.sampler == "dpm++" else rs
    b = schedule(args.steps)
    print(f"sampler={args.sampler} steps={args.steps}", flush=True)

    real_imgs, anon_imgs, rec_imgs = [], [], []
    recon_psnr, anon_psnr, examples = [], [], []
    t0 = time.perf_counter()
    for i, (pid, path, _lab) in enumerate(entries):
        x0 = preprocess(Path(path))[0].unsqueeze(0).to(device)       # [1,1,256,256] in [0,1]
        anon, recovered = anonymize_and_recover(sampler, x0, b, pid)
        real_a = x0[0, 0].detach().float().cpu().numpy()
        anon_a = anon[0, 0].detach().float().cpu().numpy()
        rec_a = recovered[0, 0].detach().float().cpu().numpy()
        real_imgs.append(to_pil(real_a)); anon_imgs.append(to_pil(anon_a)); rec_imgs.append(to_pil(rec_a))
        recon_psnr.append(psnr(recovered, x0)); anon_psnr.append(psnr(anon, x0))
        if len(examples) < args.save_examples:
            examples.append((np.clip(real_a, 0, 1), np.clip(anon_a, 0, 1), np.clip(rec_a, 0, 1)))
        if (i + 1) % 10 == 0 or i + 1 == len(entries):
            print(f"  generated {i + 1}/{len(entries)}  "
                  f"recon_psnr~{np.mean(recon_psnr):.1f}dB anon_psnr~{np.mean(anon_psnr):.1f}dB "
                  f"({(time.perf_counter() - t0) / (i + 1):.2f}s/img)", flush=True)
    gen_seconds = time.perf_counter() - t0

    image_sets = {"real": real_imgs, "anon": anon_imgs, "recovered": rec_imgs}
    matrix = [("real", "real"), ("real", "anon"), ("real", "recovered"), ("anon", "anon")]
    results = {}
    for enc in args.encoders:
        try:
            if enc == "densenet121_imagenet":
                feats = {k: densenet_embeddings(v, device, 16) for k, v in image_sets.items()}
            else:
                feats = {k: rad_dino_embeddings(v, device, 16, RAD_DINO_ID) for k, v in image_sets.items()}
        except Exception as exc:  # e.g. rad-dino not cached locally
            results[enc] = {"error": str(exc)[:200]}
            print(f"{enc}: SKIPPED ({str(exc)[:80]})", flush=True)
            continue
        enc_res = {}
        for tr_type, te_type in matrix:
            m = probe(feats[tr_type][:train_n], y_train, feats[te_type][train_n:], y_test)
            enc_res[f"train_{tr_type}__test_{te_type}"] = m
            print(f"{enc:>22}  train {tr_type:>9} -> test {te_type:>9}  "
                  f"acc={m['acc']:.3f} auroc={m['auroc']:.3f} f1={m['f1']:.3f}", flush=True)
        results[enc] = {"feat_dim": int(feats["real"].shape[1]), "matrix": enc_res}

    payload = {
        "seed": SEED, "design": "A: No Finding (y0) vs Pleural Effusion (y1), binary",
        "split": "patient-disjoint; one image per patient", "per_class": args.per_class,
        "train_n": train_n, "test_n": len(test), "steps": args.steps, "scheme": "S2fn",
        "sampler": args.sampler,
        "render": "shared [0,1] 256x256 -> grayscale RGB for real/anon/recovered",
        "pipeline": f"aligned {args.sampler}; B(E(F(x0))) anon; B(D(F(PNG(anon)))) recovered",
        "probe": "StandardScaler + LogisticRegression, frozen features",
        "recon_psnr_vs_x0_db": {"mean": round(float(np.mean(recon_psnr)), 3), "min": round(float(np.min(recon_psnr)), 3)},
        "anon_psnr_vs_x0_db": {"mean": round(float(np.mean(anon_psnr)), 3)},
        "generation_seconds": round(gen_seconds, 1), "results": results,
    }
    out = args.out_dir / f"anon_utility_{args.sampler}_n{args.per_class}_s{args.steps}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")

    if examples:
        cols = ("real (x0)", "anon (S2fn)", "recovered")
        pad = 20
        canvas = Image.new("RGB", (256 * 3, len(examples) * (256 + pad)), "white")
        for r, triple in enumerate(examples):
            for c, arr in enumerate(triple):
                canvas.paste(to_pil(arr), (c * 256, r * (256 + pad) + pad))
        sheet = args.out_dir / f"anon_utility_examples_{args.sampler}_n{args.per_class}_s{args.steps}.png"
        canvas.save(sheet)
        print(f"wrote {sheet}  (columns: {', '.join(cols)})")


if __name__ == "__main__":
    main()
