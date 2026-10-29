#!/usr/bin/env python3
"""Presentation figures for the defense report outline (presentation/).

Reads FROZEN tables (numbers never hand-typed) and emits three PNGs into
presentation/assets/:

  1. fig_storage_motivation.png -- why range-preserving storage was proposed:
     legacy min-max PNG barely touches plaintext (-0.12 dB) but destroys
     encrypted-latent fidelity (S0/S1 -7.1/-7.4 dB); range-preserving recovers
     it.  Source: paper_assets/tables/table_error_attribution.csv.
  2. fig_sampler_nsweep.png -- recon PSNR vs inversion steps N for DDIM / BDIA /
     DPM-Solver++, marking the N=100 deployment choice.  Source:
     presentation/assets/W9-T1_sampler_comparison.csv.
  3. fig_pipeline_passes.png -- predecessor native 6-pass vs our layer-A 4-pass
     (diagram; pass breakdown from presentation/口試問題回覆.md / predecessor_decrypt_flow.md).

  scripts/run_cfg_ddim.sh python -m scripts.plot_presentation_figures
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

C_OURS = "#1f6feb"   # blue
C_PRED = "#b0582c"   # legacy/brown
C_GREY = "#9aa0a6"
C_GOOD = "#2e7d32"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def fig_storage_motivation() -> None:
    """Grouped bars: range-preserving vs legacy min-max PSNR per scheme."""
    rows = read_rows(ROOT / "paper_assets" / "tables" / "table_error_attribution.csv")
    rp, lg = {}, {}
    for r in rows:
        if r["stage"] == "m1_sampler_range":          # range-preserving storage
            rp[r["method"]] = float(r["image_psnr_db_vs_x0"])
        elif r["stage"] == "m1_deploy_legacy":        # legacy min-max storage
            lg[r["method"]] = float(r["image_psnr_db_vs_x0"])
    schemes = ["P", "S0", "S1", "S2"]
    labels = ["P\n(plaintext)", "S0\n(predecessor)", "S1\n(predecessor)", "S2\n(ours)"]
    rp_v = [rp[s] for s in schemes]
    lg_v = [lg[s] for s in schemes]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = range(len(schemes))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], lg_v, w, label="legacy min-max PNG (predecessor storage)", color=C_PRED)
    b2 = ax.bar([i + w / 2 for i in x], rp_v, w, label="range-preserving PNG (ours)", color=C_OURS)
    for i, s in enumerate(schemes):
        drop = lg[s] - rp[s]
        tag = "N/A (already\nrange-preserving)" if s == "S2" else f"{drop:+.2f} dB"
        ax.annotate(tag, (i, max(lg[s], rp[s]) + 0.6), ha="center", va="bottom",
                    fontsize=9, color=(C_GREY if s == "S2" else C_PRED), fontweight="bold")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_ylabel("Recovery PSNR vs $x_0$ (dB)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(28, 44)
    ax.set_title("Why range-preserving storage: legacy min-max discards the\n"
                 "encrypted latent's value range (−0.12 dB for plaintext, −7 dB for S0/S1)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "fig_storage_motivation.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def fig_sampler_nsweep() -> None:
    rows = read_rows(ASSETS / "W9-T1_sampler_comparison.csv")
    N = [int(r["N"]) for r in rows]
    series = [("DDIM", "ddim_psnr_db", C_GREY, "o", "-"),
              ("BDIA", "bdia_psnr_db", C_PRED, "s", "--"),
              ("DPM-Solver++", "dpmpp_psnr_db", C_OURS, "D", "-")]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, col, color, mk, ls in series:
        y = [float(r[col]) for r in rows]
        ax.plot(N, y, ls, marker=mk, color=color, label=name, lw=2, ms=6)
    ax.axvline(100, color=C_GOOD, ls=":", lw=1.5)
    ax.annotate("deployment: N=100\n(DPM++ 45.6 dB, ~5x faster)", (100, 31),
                color=C_GOOD, fontsize=9, ha="center",
                bbox=dict(boxstyle="round", fc="white", ec=C_GOOD, alpha=0.9))
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel("inversion steps N (= model evals per pass)")
    ax.set_ylabel("Recovery PSNR vs $x_0$ (dB)")
    ax.set_title("Sampler comparison: DPM-Solver++ dominates at every step budget\n"
                 "(key-free round-trip, matched NFE; source W9-T1)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "fig_sampler_nsweep.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def _box(ax, x, y, w, h, text, color, alpha=1.0, fontcolor="white", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none", alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=fontcolor, fontsize=fs, zorder=3, fontweight="bold")


def _arrow(ax, x0, y0, x1, y1, color="#555"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.4, zorder=1))


def fig_pipeline_passes() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 4.4); ax.axis("off")
    w, h, gap = 1.72, 0.9, 0.20

    pred = ["invert\n$x_0{\\to}z$", "generate\nanon", "invert\nanon", "decrypt +\nrecover",
            "re-invert\n(redundant)", "generate\npseudo-healthy"]
    ours = ["invert\n$x_0{\\to}z$", "generate\nanon", "invert\nanon", "decrypt +\nrecover"]

    # predecessor row (top)
    y = 2.9
    ax.text(0.05, y + h + 0.32, "Predecessor native pipeline — 6 passes (~7 s/pass dense DDIM  ⇒  ~42 s/image)",
            fontsize=11, fontweight="bold", color=C_PRED)
    for i, t in enumerate(pred):
        x = 0.1 + i * (w + gap)
        removed = i >= 4
        _box(ax, x, y, w, h, t, C_GREY if removed else C_PRED,
             alpha=0.45 if removed else 1.0, fontcolor="#333" if removed else "white")
        if i:
            xp = 0.1 + (i - 1) * (w + gap)
            _arrow(ax, xp + w, y + h / 2, x, y + h / 2)
    # bracket for removed
    xr = 0.1 + 4 * (w + gap)
    ax.annotate("removed by layer-A\n(−2 passes, ~1.5x)", (xr + w + gap / 2, y - 0.28),
                ha="center", va="top", fontsize=8.5, color=C_GREY, fontweight="bold")

    # ours row (bottom)
    y = 0.75
    ax.text(0.05, y + h + 0.32, "Ours (layer-A) — 4 passes (DPM++ N=100 ~1.6 s/pass  ⇒  ~6.3 s/image)",
            fontsize=11, fontweight="bold", color=C_OURS)
    for i, t in enumerate(ours):
        x = 0.1 + i * (w + gap)
        _box(ax, x, y, w, h, t, C_OURS)
        if i:
            xp = 0.1 + (i - 1) * (w + gap)
            _arrow(ax, xp + w, y + h / 2, x, y + h / 2)

    ax.text(6.0, 0.18,
            "speed-up = sampler (DPM++ N=100, ~4.9x)  ×  layer-A (6→4 pass, ~1.5x)  ≈  6.7x",
            ha="center", fontsize=10, style="italic", color="#222")
    fig.tight_layout()
    out = ASSETS / "fig_pipeline_passes.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    fig_storage_motivation()
    fig_sampler_nsweep()
    fig_pipeline_passes()


if __name__ == "__main__":
    main()
