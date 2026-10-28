#!/usr/bin/env python3
"""Predecessor-architecture baseline for the binary classification-utility task.

Companion to classification_utility_anon.py, which measures our pipeline (DPM++
N=100 + S2fn, range-preserving PNG).  This script runs the SAME binary task
(Design A: No Finding y=0 vs Pleural Effusion y=1), the SAME selection, the SAME
shared render, and the SAME frozen-feature probes -- but through the PREDECESSOR'S
ORIGINAL anonymization architecture, so the two AUROC matrices sit on one ruler.

Predecessor architecture (faithful to e2_legacy_wrapper / ddim_sample_loop_anonymization):
  * legacy DDIM, dense T=500 (NOISE_LEVEL), the legacy decode convention;
  * inversion null=True; generation y=0 healthy-conditional, guidance_scale=-1
    (single model call, NO CFG mixing);
  * encryption S1 = SignedPermutationKey (password "Ki@13579", seed 42), the
    predecessor's deployed scheme.
The predecessor's pseudo-healthy CFG branch (guidance_scale=4, |x0 - healthy|) is
a LOCALIZATION tool and contributes nothing to binary classification, so it is
deliberately omitted -- the architecture reduces to invert -> encrypt -> generate
(anon), invert -> decrypt -> generate (recover), exactly as the user framed it.

The ONLY differences from classification_utility_anon are therefore the two
architecture axes we claim to improve: the SAMPLER (legacy dense DDIM T=500 vs
our DPM++ N=100) and the ENCRYPTION scheme (S1 vs S2fn).  The storage hand-off is
held fixed at our range-preserving PNG for BOTH, so any utility gap is attributable
to sampler + encryption, not to storage (this is conservative -- it lends the
predecessor our storage contribution).

NOT our full protocol: one operating point, frozen probe only (no fine-tune),
sized smaller than the main experiment (user: a single parameter setting to read
the AUROC).  Read jointly with the recon/anon PSNR printed per run.

  # smoke (2 imgs) then scale:
  scripts/run_cfg_ddim.sh python -m scripts.classification_utility_legacy --per-class 2 --noise-level 500 --splits 1
  scripts/run_cfg_ddim.sh python -m scripts.classification_utility_legacy --per-class 50 --noise-level 500 --splits 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.classification_utility_anon import probe, to_pil
from scripts.classification_utility_pilot import select_balanced
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import legacy_png_handoff
from scripts.e2_legacy_wrapper import apply_legacy_key, legacy_generate, legacy_invert
from scripts.exact_inv_sweep import CKPT
from scripts.m1_storage import range_preserving_png_handoff
from scripts.respaced_ddim_sweep import psnr, SEED


def _storage_handoff(anon: torch.Tensor, storage: str) -> torch.Tensor:
    """PNG hand-off for the published anon image before re-inversion.

    range-preserving = our contribution (lo/hi kept, round-half-to-even).
    legacy = predecessor's actual storage (per-image min-max, uint8 truncation,
    value range DISCARDED) -- scripts.e2_anonymization_runner.legacy_png_handoff.
    """
    if storage == "range-preserving":
        return range_preserving_png_handoff(anon)[0]
    return legacy_png_handoff(anon)


@torch.inference_mode()
def legacy_anonymize_and_recover(diffusion, model_fn, x0, source_root, scheme, noise_level, storage, sid):
    """Predecessor pipeline: returns (anon_tensor, recovered_tensor).

    invert -> encrypt(key) -> generate = anon; PNG hand-off; invert -> decrypt ->
    generate = recovered.  Composition of invert/apply_legacy_key/generate is the
    legacy sampler bit-exactly (tests/test_legacy_repro.py).  sid is unused by the
    legacy key (fixed full-mask key, no per-image nonce -- itself a reason S1 is
    weaker than S2fn), kept in the signature to mirror the S2 pipeline call site.
    """
    x_t = legacy_invert(diffusion, model_fn, x0, noise_level)
    x_t_enc = apply_legacy_key(x_t, source_root, scheme, "anonymize")
    anon = legacy_generate(diffusion, model_fn, x_t_enc, noise_level)
    anon_png = _storage_handoff(anon, storage)
    x_t2 = legacy_invert(diffusion, model_fn, anon_png, noise_level)
    x_t2_dec = apply_legacy_key(x_t2, source_root, scheme, "deanonymize")
    recovered = legacy_generate(diffusion, model_fn, x_t2_dec, noise_level)
    for name, t in (("anon", anon), ("recovered", recovered)):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"non-finite {name} for {sid}")
    return anon, recovered


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--noise-level", type=int, default=500, help="legacy dense DDIM T (predecessor NOISE_LEVEL)")
    ap.add_argument("--scheme", choices=["S1", "S0", "P"], default="S1", help="legacy encryption; S1 = predecessor deployed")
    ap.add_argument("--storage", choices=["range-preserving", "legacy"], default="range-preserving",
                    help="anon PNG hand-off; legacy = predecessor native (range discarded, ~30dB)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--splits", type=int, default=10,
                    help="patient-disjoint resamples; >1 reports mean & 95%% CI (images generated once)")
    ap.add_argument("--encoders", nargs="+", default=["densenet121_imagenet", "rad_dino"])
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/classification_utility"))
    ap.add_argument("--save-examples", type=int, default=6)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (legacy anonymization pipeline)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    from scripts.classification_utility_anon import densenet_embeddings, rad_dino_embeddings, RAD_DINO_ID

    picks = select_balanced(args.per_class)
    got = {k: len(v) for k, v in picks.items()}
    if min(got.values()) < 2:
        raise SystemExit(f"insufficient images resolved on disk: {got}")
    print(f"selected healthy(y0)={got[0]} effusion(y1)={got[1]}", flush=True)

    entries = [(pid, path, 0) for pid, path in picks[0]] + [(pid, path, 1) for pid, path in picks[1]]
    labels = np.array([lab for _p, _path, lab in entries])
    idx_by_class = {c: np.where(labels == c)[0] for c in (0, 1)}

    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    print(f"legacy DDIM T={args.noise_level} scheme={args.scheme} storage={args.storage} "
          f"splits={args.splits} n_images={len(entries)}", flush=True)

    real_imgs, anon_imgs, rec_imgs = [], [], []
    recon_psnr, anon_psnr, examples = [], [], []
    t0 = time.perf_counter()
    for i, (pid, path, _lab) in enumerate(entries):
        x0 = preprocess(Path(path))[0].unsqueeze(0).to(device)       # [1,1,256,256] in [0,1]
        anon, recovered = legacy_anonymize_and_recover(
            diffusion, model_fn, x0, args.source_root, args.scheme, args.noise_level, args.storage, pid)
        real_a = x0[0, 0].detach().float().cpu().numpy()
        anon_a = anon[0, 0].detach().float().cpu().numpy()
        rec_a = recovered[0, 0].detach().float().cpu().numpy()
        real_imgs.append(to_pil(real_a)); anon_imgs.append(to_pil(anon_a)); rec_imgs.append(to_pil(rec_a))
        recon_psnr.append(psnr(recovered, x0)); anon_psnr.append(psnr(anon, x0))
        if len(examples) < args.save_examples:
            examples.append((np.clip(real_a, 0, 1), np.clip(anon_a, 0, 1), np.clip(rec_a, 0, 1)))
        if (i + 1) % 5 == 0 or i + 1 == len(entries):
            print(f"  generated {i + 1}/{len(entries)}  "
                  f"recon_psnr~{np.mean(recon_psnr):.1f}dB anon_psnr~{np.mean(anon_psnr):.1f}dB "
                  f"({(time.perf_counter() - t0) / (i + 1):.1f}s/img)", flush=True)
    gen_seconds = time.perf_counter() - t0

    def make_split(seed: int):
        g = np.random.default_rng(seed)
        test = []
        for c in (0, 1):
            perm = g.permutation(idx_by_class[c])
            n_test = max(1, int(round(len(perm) * args.test_frac)))
            test.extend(int(i) for i in perm[:n_test])
        test_set = set(test)
        train_idx = np.array([i for i in range(len(entries)) if i not in test_set])
        return train_idx, np.array(sorted(test_set))
    splits = [make_split(SEED + k) for k in range(args.splits)]
    print(f"splits={args.splits}  per-split train={len(splits[0][0])} test={len(splits[0][1])}", flush=True)

    def summarize(vals):
        a = np.asarray(vals, dtype=float)
        mean = float(a.mean()); std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        ci = float(1.96 * std / np.sqrt(len(a))) if len(a) > 1 else 0.0
        return round(mean, 4), round(std, 4), round(ci, 4)

    image_sets = {"real": real_imgs, "anon": anon_imgs, "recovered": rec_imgs}
    matrix = [("real", "real"), ("real", "anon"), ("real", "recovered"), ("anon", "anon")]
    results = {}
    for enc in args.encoders:
        try:
            if enc == "densenet121_imagenet":
                feats = {k: densenet_embeddings(v, device, 16) for k, v in image_sets.items()}
            else:
                feats = {k: rad_dino_embeddings(v, device, 16, RAD_DINO_ID) for k, v in image_sets.items()}
        except Exception as exc:
            results[enc] = {"error": str(exc)[:200]}
            print(f"{enc}: SKIPPED ({str(exc)[:80]})", flush=True)
            continue
        enc_res = {}
        for tr_type, te_type in matrix:
            aurocs, accs, f1s = [], [], []
            for tr_idx, te_idx in splits:
                m = probe(feats[tr_type][tr_idx], labels[tr_idx], feats[te_type][te_idx], labels[te_idx])
                aurocs.append(m["auroc"]); accs.append(m["acc"]); f1s.append(m["f1"])
            au_m, au_s, au_ci = summarize(aurocs)
            ac_m, _, _ = summarize(accs); f1_m, _, _ = summarize(f1s)
            enc_res[f"train_{tr_type}__test_{te_type}"] = {
                "auroc": au_m, "auroc_std": au_s, "auroc_ci95": au_ci,
                "acc": ac_m, "f1": f1_m, "n_splits": args.splits}
            pm = f" (95%CI ±{au_ci:.3f})" if args.splits > 1 else ""
            print(f"{enc:>22}  train {tr_type:>9} -> test {te_type:>9}  "
                  f"auroc={au_m:.3f}{pm} acc={ac_m:.3f}", flush=True)
        results[enc] = {"feat_dim": int(feats["real"].shape[1]), "matrix": enc_res}

    payload = {
        "seed": SEED, "design": "A: No Finding (y0) vs Pleural Effusion (y1), binary",
        "architecture": "predecessor (legacy)", "split": "patient-disjoint; one image per patient",
        "per_class": args.per_class, "n_images": len(entries), "n_splits": args.splits,
        "test_frac": args.test_frac, "train_n": int(len(splits[0][0])), "test_n": int(len(splits[0][1])),
        "noise_level": args.noise_level, "scheme": args.scheme, "sampler": "legacy-DDIM",
        "storage": args.storage,
        "render": "shared [0,1] 256x256 -> grayscale RGB for real/anon/recovered",
        "pipeline": f"legacy DDIM T={args.noise_level}; invert->{args.scheme}->generate anon; "
                    f"invert->decrypt->generate recovered; {args.storage} PNG hand-off",
        "probe": f"StandardScaler + LogisticRegression, frozen features; {args.splits} patient-disjoint split(s)",
        "recon_psnr_vs_x0_db": {"mean": round(float(np.mean(recon_psnr)), 3), "min": round(float(np.min(recon_psnr)), 3)},
        "anon_psnr_vs_x0_db": {"mean": round(float(np.mean(anon_psnr)), 3)},
        "generation_seconds": round(gen_seconds, 1), "results": results,
    }
    suffix = f"_k{args.splits}" if args.splits > 1 else ""
    stag = "" if args.storage == "range-preserving" else "_legacy"
    out = args.out_dir / f"legacy_utility_{args.scheme}{stag}_n{args.per_class}_t{args.noise_level}{suffix}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")

    if examples:
        pad = 20
        canvas = Image.new("RGB", (256 * 3, len(examples) * (256 + pad)), "white")
        for r, triple in enumerate(examples):
            for c, arr in enumerate(triple):
                canvas.paste(to_pil(arr), (c * 256, r * (256 + pad) + pad))
        sheet = args.out_dir / f"legacy_utility_examples_{args.scheme}{stag}_n{args.per_class}_t{args.noise_level}.png"
        canvas.save(sheet)
        print(f"wrote {sheet}  (columns: real, anon, recovered)")


if __name__ == "__main__":
    main()
