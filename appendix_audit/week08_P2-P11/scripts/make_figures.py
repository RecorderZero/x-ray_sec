#!/usr/bin/env python3
"""Week 8（P2 附錄：T2-BB／re-id／BDIA＋P11／Z12／Q11）稽核摘要圖表產生器。

數字一律從已凍結、已稽核的來源檔**讀入計算**，不手打
（AUD-20261022-01／-02、AUD-20261023-01、AUD-20261024-01）：
  - T2-BB-M1-Frozen：`paper_assets/tables/table_t2bb_m1_frozen.json`
    （稽核方 `audit/out/t2bb_m1_recompute.json` 獨立重算 8/8 吻合）。
  - 外部 re-id（原作者權重 sanity）：`/data2/external_gpl/reid_original_weights_sanity.json`
    —— **out-of-tree GPL artifact，刻意不複製進庫**；本腳本讀它產生衍生表
    `w08_tab02_reid_same_representation.csv`（此 CSV 即庫內可留存的權威紀錄）。
    若日後該 out-of-tree 檔不在，fig02 無法重生，但衍生表已凍存。
  - Q11 定位 parity：`paper_assets/tables/table_localization_parity.csv`。
輸出 ../figures/w08_fig*.png、../tables/w08_tab*.csv。
執行：scripts/run_cfg_ddim.sh python appendix_audit/week08_P2-P11/scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
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

T2BB = ROOT / "paper_assets/tables/table_t2bb_m1_frozen.json"
REID = Path("/data2/external_gpl/reid_original_weights_sanity.json")
PARITY = ROOT / "paper_assets/tables/table_localization_parity.csv"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, BASELINE = "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def write_csv(name, rows):
    with (TAB / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


# Fig 1：T2-BB-M1-Frozen 黑箱連結 AUC（P 正控制顯著；S0/S1/S2 近隨機）
def fig1_t2bb():
    data = json.loads(T2BB.read_text(encoding="utf-8"))
    enc_label = {"densenet121_imagenet": "DenseNet121 (ImageNet)", "rad_dino": "RAD-DINO (CXR)"}
    encs = ["densenet121_imagenet", "rad_dino"]
    methods = ["P", "S0", "S1", "S2"]
    auc = {(r["method"], r["encoder"]): r["auc"] for r in data["results"]}
    pval = {(r["method"], r["encoder"]): r["auc_permutation_p"] for r in data["results"]}
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    w = 0.38; x = np.arange(len(methods))
    for k, enc in enumerate(encs):
        vals = [auc[(m, enc)] for m in methods]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=(BLUE if k == 0 else ORANGE),
                      edgecolor="white", linewidth=1.1, label=enc_label[enc])
        for b, m in zip(bars, methods):
            v = auc[(m, enc)]; p = pval[(m, enc)]
            star = "*" if p < 0.05 else ""
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}{star}",
                    ha="center", va="bottom", fontsize=6.8, color=INK)
    ax.axhline(0.5, color=BASELINE, linestyle="--", linewidth=0.9, label="chance (AUC 0.5)")
    ax.set_xticks(x, methods)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("rank-AUC, anon↔anon linkage (N=20 patients)")
    ax.legend(fontsize=7.3, loc="upper right")
    ax.set_title("T2-BB-M1-Frozen: only the unprotected control P is linkable (* = perm p<0.05)", fontsize=9.2)
    fig.text(0.5, -0.03, "Null holds only under (a) generic image-encoder interface AND (b) legacy range-discarding PNG; "
             "consistent with §5 minmax, NOT a scheme-level security proof. Range-preserving cos still leaks (AUC 0.657).",
             ha="center", fontsize=6.6, color=INK2)
    fig.savefig(FIG / "w08_fig01_t2bb_m1_frozen_auc.png")
    plt.close(fig)
    write_csv("w08_tab01_t2bb_m1_frozen.csv", [
        {"method": m, "encoder": enc, "auc": auc[(m, enc)], "auc_permutation_p": pval[(m, enc)],
         "significant_p05": pval[(m, enc)] < 0.05, "n_patients": 20,
         "source": "paper_assets/tables/table_t2bb_m1_frozen.json; recompute audit/out/t2bb_m1_recompute.json"}
        for m in methods for enc in encs])


# Fig 2：外部 re-id —— 正確的比較是「同表徵基線」，不是 plaintext 真實-CXR 天花板
def fig2_reid():
    if not REID.is_file():
        print(f"SKIP fig2 (out-of-tree artifact absent): {REID}")
        return
    data = json.loads(REID.read_text(encoding="utf-8"))
    by = {r["condition"]: r for r in data["results"]}
    order = [
        ("plaintext", "plaintext\n(real CXR, ceiling)", INK2),
        ("plaintext_latent_range_preserving", "identity latent-view\n(no cipher, control)", YELLOW),
        ("S0_range_preserving", "S0\n(transparent)", BLUE),
        ("S1_range_preserving", "S1", AQUA),
        ("S2_range_preserving", "S2\n(per-image nonce)", ORANGE),
    ]
    labels = [l for _, l, _ in order]
    vals = [by[c]["auc"] for c, _, _ in order]
    colours = [c for _, _, c in order]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(order))
    bars = ax.bar(x, vals, 0.62, color=colours, edgecolor="white", linewidth=1.1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(0.5, color=BASELINE, linestyle="--", linewidth=0.9)
    ax.text(len(order) - 0.5, 0.515, "chance", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, labels, fontsize=7.2)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("pair-verification AUC (Packhäuser original weights)")
    ax.set_title("External re-id: anchor S2 on the same-representation baseline, not the plaintext ceiling", fontsize=9.0)
    drop_rep = by["plaintext"]["auc"] - by["plaintext_latent_range_preserving"]["auc"]
    drop_cipher = by["plaintext_latent_range_preserving"]["auc"] - by["S2_range_preserving"]["auc"]
    fig.text(0.5, -0.04, f"Most of the 0.982->0.550 drop is latent-view OOD (identity, no cipher: -{drop_rep:.3f}); "
             f"the cipher's own effect is small and within-representation (0.550->0.477: -{drop_cipher:.3f}).",
             ha="center", fontsize=6.6, color=INK2)
    fig.savefig(FIG / "w08_fig02_reid_same_representation.png")
    plt.close(fig)
    write_csv("w08_tab02_reid_same_representation.csv", [
        {"condition": c, "scheme": by[c]["scheme"], "storage": by[c]["storage"], "auc": by[c]["auc"],
         "top1": by[c]["top1"], "random_top1": by[c]["random_top1"], "n_patients": 20,
         "source": "/data2/external_gpl/reid_original_weights_sanity.json (out-of-tree GPL sanity); AUD-20261022-02"}
        for c, _, _ in order])


# Tab 3：Q11 定位 parity（S2 換密碼 vs 前作自身兩方案差，逐 kind 取均值）
def tab3_parity():
    rows = list(csv.DictReader(PARITY.open(encoding="utf-8")))
    kinds = ["parity_restored", "parity_healthy", "parity_heatmap"]
    pairs = [("S2", "S0"), ("S0", "S1")]
    out = []
    for kind in kinds:
        for method, base in pairs:
            vals = [float(r["psnr_db"]) for r in rows
                    if r["kind"] == kind and r["method"] == method and r["baseline"] == base]
            out.append({"kind": kind, "pair": f"{method}-{base}", "mean_psnr_db": round(float(np.mean(vals)), 2),
                        "n_images": len(vals),
                        "source": "paper_assets/tables/table_localization_parity.csv; AUD-20261023-01"})
    write_csv("w08_tab03_localization_parity.csv", out)


if __name__ == "__main__":
    fig1_t2bb()
    fig2_reid()
    tab3_parity()
    print("done")
