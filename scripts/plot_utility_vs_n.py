#!/usr/bin/env python3
"""Item 6: recovered-image classification utility vs DDIM steps N.

Reads several anon_utility_*.json (same per-class, different --steps) and plots
the recovered-image AUROC as a function of N for each encoder, with the Real->Real
reference as a dashed line and the recon PSNR annotated per point (single y-axis;
PSNR is text, never a second axis).  Shows that utility tracks reconstruction
quality -- tying the classification story to the DPM++/step-count contribution.

  scripts/run_cfg_ddim.sh python -m scripts.plot_utility_vs_n \
      --jsons paper_assets/classification_utility/anon_utility_dpm++_n100_s*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = {"densenet121_imagenet": ("#2a78d6", "DenseNet-121"), "rad_dino": ("#eb6834", "rad-dino")}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--jsons", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("paper_assets/classification_utility/utility_vs_n.png"))
    args = ap.parse_args()
    runs = sorted((json.loads(p.read_text()) for p in args.jsons), key=lambda d: d["steps"])
    ns = [d["steps"] for d in runs]
    recon = [d.get("recon_psnr_vs_x0_db", {}).get("mean") for d in runs]
    encoders = [e for e in SERIES if all(e in d["results"] and "matrix" in d["results"][e] for d in runs)]

    fig, axis = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE); axis.set_facecolor(SURFACE)
    for enc in encoders:
        color, label = SERIES[enc]
        rec = [d["results"][enc]["matrix"]["train_real__test_recovered"]["auroc"] for d in runs]
        ref = [d["results"][enc]["matrix"]["train_real__test_real"]["auroc"] for d in runs]
        ci = [d["results"][enc]["matrix"]["train_real__test_recovered"].get("auroc_ci95", 0) for d in runs]
        axis.errorbar(ns, rec, yerr=ci, color=color, lw=2, marker="o", ms=7, capsize=3,
                      zorder=3, label=f"{label}: recovered")
        axis.plot(ns, ref, color=color, lw=1.3, ls=(0, (4, 3)), marker="", zorder=2,
                  label=f"{label}: real (ref)")

    for n, p in zip(ns, recon):
        if p is not None:
            axis.annotate(f"{p:.0f} dB", (n, axis.get_ylim()[0]), xytext=(0, 4),
                          textcoords="offset points", ha="center", va="bottom",
                          fontsize=8.5, color=INK2, style="italic")
    axis.axhline(0.5, ls=":", lw=1.0, color=INK2, zorder=1)
    axis.text(ns[-1], 0.505, "chance", ha="right", va="bottom", fontsize=8.5, color=INK2, style="italic")

    axis.set_xticks(ns); axis.set_xlabel("DDIM steps N  (recon PSNR annotated)", fontsize=10.5, color=INK)
    axis.set_ylabel("AUROC", fontsize=10.5, color=INK)
    axis.set_ylim(0.45, 1.0)
    axis.tick_params(colors=INK2); axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, lw=0.8)
    for sp in ("top", "right"):
        axis.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axis.spines[sp].set_color(GRID)
    axis.set_title("Recovered-image classification utility vs reconstruction steps (DPM++, S2fn)",
                   fontsize=10.5, color=INK, pad=10)
    axis.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
