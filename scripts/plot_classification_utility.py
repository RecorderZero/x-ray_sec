#!/usr/bin/env python3
"""Paper figure: classification utility-preservation matrix (grouped bars).

Reads an anon_utility_*.json produced by classification_utility_anon.py and
renders the four-probe AUROC matrix as grouped bars (one group per probe, one
bar per encoder) with a chance=0.5 reference line.  Message: Real->Real and
Real->Recovered sit high together (recovery preserves utility); Real->Anon and
Anon->Anon fall to chance (the cipher hides the pathology).

  scripts/run_cfg_ddim.sh python -m scripts.plot_classification_utility \
      --json paper_assets/classification_utility/anon_utility_dpm++_n200_s100.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# dataviz default categorical theme, slots 1-2 (validated CVD-safe pair)
SERIES = {"densenet121_imagenet": "#2a78d6", "rad_dino": "#eb6834"}
SERIES_LABEL = {"densenet121_imagenet": "DenseNet-121 (ImageNet)", "rad_dino": "rad-dino (CXR)"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
PROBES = [("train_real__test_real", "Real→Real"),
          ("train_real__test_recovered", "Real→Recovered"),
          ("train_real__test_anon", "Real→Anon"),
          ("train_anon__test_anon", "Anon→Anon")]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--metric", default="auroc", choices=["auroc", "acc", "f1"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    payload = json.loads(args.json.read_text())
    results = payload["results"]
    encoders = [e for e in SERIES if e in results and "matrix" in results[e]]
    if not encoders:
        raise SystemExit("no usable encoder results in json")

    x = np.arange(len(PROBES))
    n = len(encoders)
    width = 0.72 / n
    fig, axis = plt.subplots(figsize=(7.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE); axis.set_facecolor(SURFACE)

    for i, enc in enumerate(encoders):
        vals = [results[enc]["matrix"][key][args.metric] for key, _ in PROBES]
        cis = [results[enc]["matrix"][key].get(args.metric + "_ci95", 0) or 0 for key, _ in PROBES]
        offset = (i - (n - 1) / 2) * width
        bars = axis.bar(x + offset, vals, width * 0.92, color=SERIES[enc],
                        label=SERIES_LABEL[enc], zorder=3,
                        yerr=cis if any(cis) else None, capsize=3,
                        error_kw={"ecolor": INK2, "elinewidth": 1.1, "zorder": 4})
        for rect, v, c in zip(bars, vals, cis):
            axis.text(rect.get_x() + rect.get_width() / 2, v + c + 0.012, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=4)

    axis.axhline(0.5, ls=(0, (4, 3)), lw=1.4, color=INK2, zorder=2)
    axis.text(len(PROBES) - 0.52, 0.515, "chance", ha="right", va="bottom",
              fontsize=8.5, color=INK2, style="italic")

    # recovery (utility) vs cipher (privacy) regions, labelled above the bars
    axis.text(0.5, 1.045, "utility preserved", ha="center", va="center", fontsize=9,
              color=SERIES["densenet121_imagenet"], fontweight="bold", transform=axis.get_xaxis_transform())
    axis.text(2.5, 1.045, "pathology hidden in cipher", ha="center", va="center", fontsize=9,
              color=SERIES["rad_dino"], fontweight="bold", transform=axis.get_xaxis_transform())
    axis.axvline(1.5, ls=":", lw=1.0, color=GRID, zorder=1)

    axis.set_xticks(x); axis.set_xticklabels([lbl for _, lbl in PROBES], fontsize=10, color=INK)
    axis.set_ylim(0, 1.0); axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.set_ylabel(args.metric.upper(), fontsize=10.5, color=INK)
    axis.tick_params(colors=INK2)
    axis.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)

    sampler = payload.get("sampler", "?"); recon = payload.get("recon_psnr_vs_x0_db", {}).get("mean", "?")
    axis.set_title(f"Classification utility under S2 anonymization  "
                   f"(DPM++, N={payload.get('steps')}, recon {recon} dB, test n={payload.get('test_n')})",
                   fontsize=10.5, color=INK, pad=26)
    axis.legend(frameon=False, fontsize=9.5, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.22), labelcolor=INK)
    fig.tight_layout()

    out = args.out or args.json.with_name(args.json.stem + f"_{args.metric}.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
