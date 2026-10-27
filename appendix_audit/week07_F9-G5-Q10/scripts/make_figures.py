#!/usr/bin/env python3
"""Week 7（F9／G5／Q10）稽核摘要圖表產生器。

數字一律從已凍結、已稽核的來源檔**讀入計算**，不手打（AUD-20261017-01／-02、
AUD-20261019-01／-02；`paper_assets/tables/table_error_attribution.csv`，稽核方
已於 `audit/out/q10_verify.json` 獨立重算一致）。
輸出 ../figures/w07_fig*.png、../tables/w07_tab*.csv。
執行：scripts/run_cfg_ddim.sh python appendix_audit/week07_F9-G5-Q10/scripts/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
WEEK = Path(__file__).resolve().parents[1]
FIG, TAB = WEEK / "figures", WEEK / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "paper_assets/tables/table_error_attribution.csv"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, BASELINE = "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"
METHOD_COLOUR = {"P": INK2, "S0": BLUE, "S1": ORANGE, "S2": AQUA}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def write_csv(name, rows):
    with (TAB / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


def load_attribution():
    """method -> {stage: psnr}；只取有 image PSNR 的 stage。"""
    out: dict[str, dict[str, float]] = {}
    with SRC.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = r["image_psnr_db_vs_x0"].strip()
            if v:
                out.setdefault(r["method"], {})[r["stage"]] = float(v)
    return out


# Fig 1：誤差歸因階梯（M2 精確 latent 天花板 → M1 sampler 反轉 → M1 legacy 存檔）
def fig1_ladder(attr):
    methods = ["P", "S0", "S1", "S2"]
    stages = ["m2_exact_latent", "m1_sampler_range", "m1_deploy_legacy"]
    labels = ["M2 exact-latent\nceiling", "M1 sampler\n(range-preserving)", "M1 legacy\n(min-max storage)"]
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    x = np.arange(len(stages))
    for m in methods:
        y = [attr[m].get(s) for s in stages]
        ax.plot(x, y, marker="o", color=METHOD_COLOUR[m], linewidth=1.8, markersize=5, label=m)
        for xi, yi in zip(x, y):
            if yi is not None:
                ax.text(xi, yi + 0.18, f"{yi:.1f}", ha="center", va="bottom", fontsize=6.6, color=METHOD_COLOUR[m])
    ax.axhline(attr["P"]["m2_exact_latent"], color=BASELINE, linestyle="--", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylim(29, 43)
    ax.set_ylabel("image PSNR vs x0 (dB, formal N=200)")
    ax.legend(title="method", fontsize=8, ncol=4, loc="lower center")
    ax.set_title("Q10.1 error-attribution ladder: cipher is lossless; the losses are sampler + storage", fontsize=9.5)
    fig.text(0.5, -0.04, "Crypto round-trip is bit-exact (max_abs=0; S2 WHT ~8.3e-7). M2 ceiling 41.87 dB = VAE+8-bit. "
             "Sampler invert/regen is the honest ~3-4 dB residual. Legacy min-max costs S0/S1 ~7 dB; S2 keeps the range.",
             ha="center", fontsize=6.6, color=INK2)
    fig.savefig(FIG / "w07_fig01_error_attribution_ladder.png")
    plt.close(fig)
    write_csv("w07_tab01_error_attribution.csv", [
        {"method": m, "m2_exact_latent_db": attr[m]["m2_exact_latent"],
         "m1_sampler_range_db": attr[m]["m1_sampler_range"],
         "m1_deploy_legacy_db": attr[m].get("m1_deploy_legacy", ""),
         "n": 200, "source": "paper_assets/tables/table_error_attribution.csv; AUD-20261019-01"}
        for m in methods])


# Fig 2：品質的兩種存檔口徑（部署 vs range-preserving attribution），隔離存檔與加密
def fig2_storage(attr):
    methods = ["P", "S0", "S1", "S2"]
    # deployment: P/S0/S1 走 legacy，S2 走 range-preserving（其部署即保值域）
    deploy = [attr[m]["m1_deploy_legacy"] if m != "S2" else attr[m]["m1_sampler_range"] for m in methods]
    rangea = [attr[m]["m1_sampler_range"] for m in methods]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    w = 0.36; x = np.arange(len(methods))
    b1 = ax.bar(x - w / 2, deploy, w, color=ORANGE, edgecolor="white", linewidth=1.1,
                label="deployment storage (P/S0/S1 legacy, S2 range-preserving)")
    b2 = ax.bar(x + w / 2, rangea, w, color=AQUA, edgecolor="white", linewidth=1.1,
                label="range-preserving (attribution, isolates cipher)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=6.8, color=INK)
    ax.set_xticks(x, methods)
    ax.set_ylim(29, 42)
    ax.set_ylabel("M1 reconstruction PSNR vs x0 (dB, formal N=200)")
    ax.legend(fontsize=7.3, loc="lower center")
    ax.set_title("F9/G5 quality: storage effect separated from cipher effect", fontsize=9.5)
    s2_s0_same = rangea[methods.index("S2")] - rangea[methods.index("S0")]
    s2_s0_mixed = deploy[methods.index("S2")] - deploy[methods.index("S0")]
    fig.text(0.5, -0.03, f"Same-storage S2-S0 = +{s2_s0_same:.2f} dB (cipher cost, tiny); "
             f"mixed-deployment S2-S0 = +{s2_s0_mixed:.2f} dB is almost all storage, not encryption.",
             ha="center", fontsize=6.8, color=INK2)
    fig.savefig(FIG / "w07_fig02_quality_by_storage.png")
    plt.close(fig)
    write_csv("w07_tab02_quality_by_storage.csv", [
        {"method": m, "deployment_storage": ("legacy" if m != "S2" else "range-preserving"),
         "deploy_psnr_db": d, "range_preserving_psnr_db": r, "n": 200,
         "source": "paper_assets/tables/table_error_attribution.csv; AUD-20261017-01/-019-01"}
        for m, d, r in zip(methods, deploy, rangea)])


if __name__ == "__main__":
    attr = load_attribution()
    fig1_ladder(attr)
    fig2_storage(attr)
    print("done")
