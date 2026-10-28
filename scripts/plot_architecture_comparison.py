#!/usr/bin/env python3
"""Paper figure: recovered-image utility is architecture-insensitive.

Three architectures at the SAME operating point (per-class 100, k20, recovered
AUROC): ours (DPM++ N=100 + S2fn + range-preserving) vs predecessor (legacy DDIM
T=500 + S1) with range-preserving and with native storage.  Grouped bars (one group
per architecture, one bar per encoder) on a single AUROC axis with a chance line;
each group annotated with its recon PSNR and speed -- the message being that the
recovered utility sits flat ~0.90 across all three, so our contribution lives in the
annotated axes (speed, reconstruction fidelity), not in whether classification survives.

  scripts/run_cfg_ddim.sh python -m scripts.plot_architecture_comparison
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
CU = Path("paper_assets/classification_utility")
# (label, json) in intended left->right order
ARCHES = [
    ("Ours\nDPM++ N=100 · S2fn", CU / "anon_utility_dpm++_n100_s100_k20.json"),
    ("Predecessor\nDDIM T=500 · S1\n+ range-preserving", CU / "legacy_utility_S1_n100_t500_k20.json"),
    ("Predecessor\nDDIM T=500 · S1\n+ native storage", CU / "legacy_utility_S1_legacy_n100_t500_k20.json"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=CU / "fig_architecture_comparison.png")
    args = ap.parse_args()

    runs = [(lbl, json.loads(p.read_text())) for lbl, p in ARCHES]
    encoders = [e for e in SERIES if all(e in d["results"] and "matrix" in d["results"][e] for _, d in runs)]
    x = np.arange(len(runs)); n = len(encoders); width = 0.62 / n

    fig, axis = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE); axis.set_facecolor(SURFACE)
    for i, enc in enumerate(encoders):
        color, label = SERIES[enc]
        vals = [d["results"][enc]["matrix"]["train_real__test_recovered"]["auroc"] for _, d in runs]
        cis = [d["results"][enc]["matrix"]["train_real__test_recovered"].get("auroc_ci95", 0) or 0 for _, d in runs]
        off = (i - (n - 1) / 2) * width
        bars = axis.bar(x + off, vals, width * 0.92, color=color, label=label, zorder=3,
                        yerr=cis, capsize=3, error_kw={"ecolor": INK2, "elinewidth": 1.1, "zorder": 4})
        for rect, v, c in zip(bars, vals, cis):
            axis.text(rect.get_x() + rect.get_width() / 2, v + c + 0.012, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=4)

    axis.axhline(0.5, ls=(0, (4, 3)), lw=1.4, color=INK2, zorder=2)
    axis.text(len(runs) - 0.5, 0.515, "chance", ha="right", va="bottom", fontsize=8.5, color=INK2, style="italic")

    # recon PSNR + speed annotated per architecture (the axes where we actually differ)
    for xi, (_lbl, d) in enumerate(runs):
        recon = d["recon_psnr_vs_x0_db"]["mean"]
        spi = d["generation_seconds"] / d["n_images"]
        axis.text(xi, 0.455, f"recon {recon:.1f} dB\n{spi:.1f} s/img", ha="center", va="top",
                  fontsize=8.5, color=INK, transform=axis.get_xaxis_transform())

    axis.set_xticks(x); axis.set_xticklabels([lbl for lbl, _ in runs], fontsize=9, color=INK)
    axis.set_ylim(0.4, 1.0); axis.set_yticks(np.arange(0.4, 1.01, 0.1))
    axis.set_ylabel("Recovered-image AUROC", fontsize=10.5, color=INK)
    axis.tick_params(colors=INK2); axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        axis.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axis.spines[sp].set_color(GRID)
    axis.set_title("Recovered-image utility is architecture-insensitive\n"
                   "(flat ~0.90; our gains are speed & reconstruction fidelity, annotated)",
                   fontsize=10.5, color=INK, pad=10)
    axis.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
