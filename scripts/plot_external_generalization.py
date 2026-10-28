#!/usr/bin/env python3
"""Paper figure: the utility classifier generalizes to an external dataset.

Reads external_utility_nih_n150.json (CheXpert-trained frozen probe, zero retraining,
evaluated in-domain on CheXpert held-out and cross-dataset on NIH ChestX-ray14).
Grouped bars (one group per encoder, in-domain vs external) on a single AUROC axis
with a chance line; message: ~0.07 drop under domain shift = the classifier learns
real pathology, not CheXpert-specific artifact.

  scripts/run_cfg_ddim.sh python -m scripts.plot_external_generalization
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
COND = [("chexpert_to_chexpert_heldout", "#2a78d6", "In-domain (CheXpert→CheXpert)"),
        ("chexpert_to_nih", "#eb6834", "External (CheXpert→NIH)")]
ENC_LABEL = {"densenet121_imagenet": "DenseNet-121", "rad_dino": "rad-dino"}
CU = Path("paper_assets/classification_utility")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--json", type=Path, default=CU / "external_utility_nih_n150.json")
    ap.add_argument("--out", type=Path, default=CU / "fig_external_generalization.png")
    args = ap.parse_args()
    d = json.loads(args.json.read_text())
    encoders = [e for e in ENC_LABEL if e in d["results"] and "error" not in d["results"][e]]
    x = np.arange(len(encoders)); n = len(COND); width = 0.62 / n

    fig, axis = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); axis.set_facecolor(SURFACE)
    for i, (key, color, label) in enumerate(COND):
        vals = [d["results"][e][key]["auroc"] for e in encoders]
        off = (i - (n - 1) / 2) * width
        bars = axis.bar(x + off, vals, width * 0.92, color=color, label=label, zorder=3)
        for rect, v in zip(bars, vals):
            axis.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=9, color=INK2, zorder=4)

    axis.axhline(0.5, ls=(0, (4, 3)), lw=1.4, color=INK2, zorder=2)
    axis.text(len(encoders) - 0.5, 0.515, "chance", ha="right", va="bottom", fontsize=8.5, color=INK2, style="italic")

    axis.set_xticks(x); axis.set_xticklabels([ENC_LABEL[e] for e in encoders], fontsize=10, color=INK)
    axis.set_ylim(0.4, 1.0); axis.set_yticks(np.arange(0.4, 1.01, 0.1))
    axis.set_ylabel("AUROC", fontsize=10.5, color=INK)
    axis.tick_params(colors=INK2); axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        axis.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axis.spines[sp].set_color(GRID)
    axis.set_title("Utility classifier transfers to an external dataset\n"
                   "(zero retraining; CheXpert→NIH ChestX-ray14, ~0.07 AUROC drop)",
                   fontsize=10.5, color=INK, pad=10)
    axis.legend(frameon=False, fontsize=9.5, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.2), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
