#!/usr/bin/env python3
"""Layer-B1 probe: how few DDIM steps can the inversion round-trip tolerate?

The predecessor's inversion/generation loops step one integer timestep at a time
(`gaussian_diffusion.py:1462,1525`, ~499 model evals/pass) because they rely on
`alphas_cumprod_prev/next`.  To take *fewer, larger* steps we spell out a
respaced DDIM forward/backward that reuses the SAME model and the exact
`p_mean_variance` / `_predict_eps_from_xstart` math, only jumping `alpha_bar` to
the next *scheduled* timestep instead of the adjacent one (eta=0, guidance -1).
We do NOT touch `past/SourceCode/`.

The encryption transform is bit-exact and orthogonal to step count, so the
step-count effect is isolated by measuring the **key-free** round-trip
`x0 -> forward_N -> z -> backward_N -> x_rec` (this is exactly the "P output vs
x0" quantity).  Validation gate: at full density (N≈499) PSNR(x_rec,x0) must
reproduce the frozen baseline P ≈ 42 dB; otherwise the respaced math is wrong
and the sweep is not trustworthy.

  scripts/run_cfg_ddim.sh python -m scripts.respaced_ddim_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import legacy_invert, legacy_generate

SEED = 1911
TOP = 499  # legacy noise_level-1 top timestep
INDICES = (0, 1, 10, 11)
SWEEP_N = (499, 200, 100, 50, 25, 10)
GRID_N = (499, 100, 50, 25, 10)


def schedule(n: int, top: int = TOP) -> np.ndarray:
    """n+1 ascending boundary timesteps in [0, top]; model evals = len-1."""
    b = np.rint(np.linspace(0, top, n + 1)).astype(int)
    return np.unique(b)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().clamp(0, 1)
    b = b.double().clamp(0, 1)
    mse = float((a - b).pow(2).mean())
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def disp(t: torch.Tensor) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo + 1e-8)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


class Respaced:
    """Respaced DDIM forward/backward sharing the legacy model + math."""

    def __init__(self, diffusion, model_fn, device):
        self.d = diffusion
        self.m = model_fn
        self.dev = device
        self.abar = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float64, device=device)

    def _base_kwargs(self, b: int) -> dict:
        return {
            "y": torch.zeros(b, device=self.dev, dtype=torch.int64),
            "p_uncond": -1,
            "clf_free": True,
        }

    @torch.inference_mode()
    def _step(self, x, t_cur, t_dst, model_kwargs):
        t = torch.full((x.shape[0],), int(t_cur), device=self.dev, dtype=torch.long)
        out = self.d.p_mean_variance(self.m, x, t, clip_denoised=True, model_kwargs=model_kwargs)
        px = out["pred_xstart"]
        eps = self.d._predict_eps_from_xstart(x, t, px)
        a = self.abar[int(t_dst)].to(x.dtype)
        return px * torch.sqrt(a) + torch.sqrt(1 - a) * eps

    @torch.inference_mode()
    def forward(self, x0, boundaries):
        mk = self._base_kwargs(x0.shape[0]); mk["null"] = True
        x = x0
        for k in range(len(boundaries) - 1):
            x = self._step(x, boundaries[k], boundaries[k + 1], mk)
        return x

    @torch.inference_mode()
    def backward(self, z, boundaries):
        mk = self._base_kwargs(z.shape[0])  # guidance -1 => no diffusion-level clf_free
        x = z
        for k in range(len(boundaries) - 1, 0, -1):
            x = self._step(x, boundaries[k], boundaries[k - 1], mk)
        return x


def _timed(fn, *a):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    ap.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"))
    ap.add_argument("--indices", type=int, nargs="+", default=list(INDICES))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/respaced_ddim_sweep"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device)

    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    images = [preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device) for i in args.indices]
    ids = [rows[i]["sample_id"] for i in args.indices]

    # warm-up (CUDA init / kernel compile) so the first timed pass is honest
    rs.backward(rs.forward(images[0], schedule(10)), schedule(10))

    # Legacy dense round-trip anchor on the SAME images (no crypto): the honest
    # reference the respaced scheme must at least match.  The legacy P pipeline
    # reports ~42 dB (anon_vs_x0); a correct respaced loop should be >= this.
    legacy_rt = [psnr(legacy_generate(diffusion, model_fn, legacy_invert(diffusion, model_fn, x0, 500), 500), x0)
                 for x0 in images]
    legacy_anchor = float(np.mean(legacy_rt))
    print(f"[anchor] legacy dense round-trip PSNR vs x0 = {legacy_anchor:.2f} dB", flush=True)

    records = []  # per (N, image)
    recon = {n: [] for n in SWEEP_N}
    for n in SWEEP_N:
        b = schedule(n)
        evals = len(b) - 1
        for x0, sid in zip(images, ids):
            z, t_f = _timed(rs.forward, x0, b)
            x_rec, t_b = _timed(rs.backward, z, b)
            records.append({
                "N": n, "evals_per_pass": evals, "sample_id": sid,
                "roundtrip_psnr_vs_x0_db": psnr(x_rec, x0),
                "fwd_seconds": round(t_f, 4), "bwd_seconds": round(t_b, 4),
            })
            recon[n].append(x_rec.detach().cpu())
        print(json.dumps(records[-len(images)], ensure_ascii=False), flush=True)

    # summary per N
    summary = {}
    for n in SWEEP_N:
        rr = [r for r in records if r["N"] == n]
        evals = rr[0]["evals_per_pass"]
        mean_psnr = float(np.mean([r["roundtrip_psnr_vs_x0_db"] for r in rr]))
        mean_pass = float(np.mean([r["fwd_seconds"] for r in rr] + [r["bwd_seconds"] for r in rr]))
        summary[n] = {"evals_per_pass": evals, "mean_roundtrip_psnr_db": mean_psnr,
                      "mean_seconds_per_pass": round(mean_pass, 4),
                      "est_4pass_seconds": round(4 * mean_pass, 2)}

    full = summary[SWEEP_N[0]]
    psnr_by_n = [summary[n]["mean_roundtrip_psnr_db"] for n in SWEEP_N]  # N descending
    monotonic = all(psnr_by_n[i] >= psnr_by_n[i + 1] - 0.05 for i in range(len(psnr_by_n) - 1))
    # Correct gate: full-density respaced is at least as good as the legacy dense
    # round-trip (within 1 dB), AND fidelity decreases monotonically with N.
    gate_ok = (full["mean_roundtrip_psnr_db"] >= legacy_anchor - 1.0) and monotonic
    payload = {"seed": SEED, "top_timestep": TOP, "indices": args.indices,
               "legacy_dense_roundtrip_anchor_psnr": round(legacy_anchor, 3),
               "validation_gate_full_density_psnr": full["mean_roundtrip_psnr_db"],
               "validation_gate_monotonic": monotonic,
               "validation_gate_pass": gate_ok, "summary": summary, "per_sample": records}
    (args.out_dir / "sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "evals_per_pass", "mean_roundtrip_psnr_db",
                    "mean_seconds_per_pass", "est_4pass_seconds", "speedup_vs_full"])
        base = summary[SWEEP_N[0]]["mean_seconds_per_pass"]
        for n in SWEEP_N:
            s = summary[n]
            w.writerow([n, s["evals_per_pass"], round(s["mean_roundtrip_psnr_db"], 2),
                        s["mean_seconds_per_pass"], s["est_4pass_seconds"],
                        round(base / s["mean_seconds_per_pass"], 2)])

    # grid: rows = images, cols = x0 | reconstructions at GRID_N
    cols = ["x0"] + [f"N={n}" for n in GRID_N]
    tile, pad, top = 196, 4, 30
    H = top + len(images) * (tile + pad) + pad
    W = pad + len(cols) * (tile + pad)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for ci, c in enumerate(cols):
        x = pad + ci * (tile + pad)
        draw.text((x + 2, 4), c, fill="black")
        for ri in range(len(images)):
            y = top + ri * (tile + pad)
            if c == "x0":
                arr = disp(images[ri][0, 0])
            else:
                n = GRID_N[ci - 1]
                arr = disp(recon[n][ri][0, 0])
                p = [r for r in records if r["N"] == n and r["sample_id"] == ids[ri]][0]
                draw.text((x + 3, y + 3), f"{p['roundtrip_psnr_vs_x0_db']:.1f} dB", fill="yellow")
            canvas.paste(Image.fromarray(arr, mode="L").convert("RGB").resize((tile, tile)), (x, y))
    canvas.save(args.out_dir / "grid.png")

    print(f"\n=== respaced DDIM sweep (mean over {len(images)} images) ===")
    print(f"{'N':>5} {'evals':>6} {'psnr_dB':>9} {'s/pass':>8} {'~4pass_s':>9} {'speedup':>8}")
    base = summary[SWEEP_N[0]]["mean_seconds_per_pass"]
    for n in SWEEP_N:
        s = summary[n]
        print(f"{n:>5} {s['evals_per_pass']:>6} {s['mean_roundtrip_psnr_db']:>9.2f} "
              f"{s['mean_seconds_per_pass']:>8.3f} {s['est_4pass_seconds']:>9.2f} "
              f"{base / s['mean_seconds_per_pass']:>7.2f}x")
    print(f"\nlegacy dense round-trip anchor: {legacy_anchor:.2f} dB")
    print(f"validation gate (respaced full-density >= legacy-1dB AND monotonic): "
          f"full={full['mean_roundtrip_psnr_db']:.2f} dB, monotonic={monotonic} "
          f"-> {'PASS' if gate_ok else 'FAIL'}")
    print(f"wrote {args.out_dir}/sweep.json + .csv + grid.png")


if __name__ == "__main__":
    main()
