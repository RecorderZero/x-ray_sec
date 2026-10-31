#!/usr/bin/env python3
"""Week 9（W9 加速取樣器＋null 部署＋utility/security 收尾）稽核摘要圖表產生器。

數字一律從已凍結、已稽核的來源檔**讀入計算**，不手打
（AUD-20261027-01／-20261028-01／-02／-20261029-01；見 AUDIT_summary.md § Week 9）：
  - AF-044 decode×sampler 分解：`paper_assets/rq4_redo/decompose.json`。
  - 層 B 加速（PSNR vs NFE）：`paper_assets/exact_inv_sweep/sweep.json`。
  - classification utility（null 部署）：
    `paper_assets/classification_utility/anon_utility_dpm++_n200_s100_k50.json`。
  - AF-045 部署忠實學習式 re-id：
    `paper_assets/tables/table_t2bb_learned_attack_dpm_deploy.json`。
  - AF-046 多病理裁判：`paper_assets/referee_pathology/referee_pathology_n400.json`。
  - AF-046 HF-FIDELITY：`paper_assets/hf_fidelity/hf_fidelity.json`。
輸出 ../figures/w09_fig*.png、../tables/w09_tab*.csv。
執行：scripts/run_cfg_ddim.sh python appendix_audit/week09_W9-accel-deploy/scripts/make_figures.py
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

RQ4 = ROOT / "paper_assets/rq4_redo/decompose.json"
ACCEL = ROOT / "paper_assets/exact_inv_sweep/sweep.json"
UTIL = ROOT / "paper_assets/classification_utility/anon_utility_dpm++_n200_s100_k50.json"
REID = ROOT / "paper_assets/tables/table_t2bb_learned_attack_dpm_deploy.json"
REFEREE = ROOT / "paper_assets/referee_pathology/referee_pathology_n400.json"
HF = ROOT / "paper_assets/hf_fidelity/hf_fidelity.json"

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


# Fig 1：AF-044 decode×sampler 二維分解（decode 增益隨 scheme 變；DPM++ N=100 ≈ aligned-DDIM N=499）
def fig1_decode():
    d = json.loads(RQ4.read_text(encoding="utf-8"))["decomposition"]
    schemes = ["P", "S2fn"]
    cols = [("legacy_ddim_N499", "legacy-DDIM\nN=499"), ("aligned_ddim_N499", "aligned-DDIM\nN=499"),
            ("dpmpp_N100", "aligned-DPM++\nN=100")]
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    w = 0.38; x = np.arange(len(cols))
    for k, sch in enumerate(schemes):
        vals = [d[sch][c] for c, _ in cols]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=(BLUE if k == 0 else ORANGE),
                      edgecolor="white", linewidth=1.1, label=sch)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [lbl for _, lbl in cols])
    ax.set_ylim(36, 48)
    ax.set_ylabel("recovery PSNR vs x0 (dB), n=4 dev")
    ax.legend(title="scheme", fontsize=8, loc="upper left")
    ax.set_title("AF-044: decode-convention gain is scheme-dependent; DPM++ N=100 ≈ aligned-DDIM N=499", fontsize=8.6)
    gp = d["P"]["decode_convention_gain_db"]; gs = d["S2fn"]["decode_convention_gain_db"]
    fig.text(0.5, -0.04, f"decode gain (legacy→aligned DDIM, same N=499): P +{gp:.2f} dB vs S2fn +{gs:.2f} dB. "
             "The S2fn headline is mostly decode-fix, NOT sampler; formal RQ4 re-report target = aligned-null, N=200.",
             ha="center", fontsize=6.6, color=INK2)
    fig.savefig(FIG / "w09_fig01_decode_sampler_decompose.png")
    plt.close(fig)
    write_csv("w09_tab01_decode_sampler_decompose.csv", [
        {"scheme": sch, "legacy_ddim_N499": d[sch]["legacy_ddim_N499"], "aligned_ddim_N499": d[sch]["aligned_ddim_N499"],
         "decode_convention_gain_db": d[sch]["decode_convention_gain_db"], "dpmpp_N100": d[sch]["dpmpp_N100"],
         "dpmpp_N200": d[sch]["dpmpp_N200"], "sampler_gain_aligned_N100_db": d[sch]["sampler_gain_aligned_N100_db"],
         "n_images": 4, "source": "paper_assets/rq4_redo/decompose.json; AUD-20261028-01"}
        for sch in schemes])


# Fig 2：層 B 加速 —— PSNR vs NFE（DPM++(2M) 同保真省步數；BDIA、DDIM 對照）
def fig2_accel():
    d = json.loads(ACCEL.read_text(encoding="utf-8"))
    anchor = d["legacy_dense_anchor_db"]
    nfes = sorted((int(k) for k in d["summary"]), reverse=False)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    series = [("dpm_psnr_db", "DPM-Solver++(2M)", ORANGE, "o"),
              ("bdia_psnr_db", "BDIA (γ=1)", AQUA, "s"),
              ("ddim_psnr_db", "respaced DDIM (η=0)", BLUE, "^")]
    for key, lbl, col, mk in series:
        ys = [d["summary"][str(n)][key] for n in nfes]
        ax.plot(nfes, ys, marker=mk, color=col, linewidth=1.6, markersize=5, label=lbl)
    ax.axhline(anchor, color=BASELINE, linestyle="--", linewidth=1.0)
    ax.text(nfes[-1], anchor + 0.3, f"legacy dense anchor {anchor:.2f} dB", ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(nfes); ax.set_xticklabels([str(n) for n in nfes])
    ax.set_xlabel("NFE per inversion pass (log scale)")
    ax.set_ylabel("S2fn recovery PSNR vs x0 (dB), n=4 dev")
    ax.legend(fontsize=7.6, loc="lower right")
    ax.set_title("Layer-B samplers: DPM-Solver++(2M) holds fidelity at ~5× fewer steps", fontsize=8.8)
    d100 = d["summary"]["100"]["dpm_psnr_db"]; d499 = d["summary"]["499"]["ddim_psnr_db"]
    fig.text(0.5, -0.04, f"DPM++ N=100 ({d100:.1f} dB) ≈ DDIM N=499 ({d499:.1f} dB); approximate inversion is NOT bit-exact "
             "(disclosed). Timings are a finite-sync prototype, not an algorithmic speed ranking.",
             ha="center", fontsize=6.6, color=INK2)
    fig.savefig(FIG / "w09_fig02_accel_psnr_vs_nfe.png")
    plt.close(fig)
    write_csv("w09_tab02_accel_psnr_vs_nfe.csv", [
        {"nfe": n, "ddim_psnr_db": d["summary"][str(n)]["ddim_psnr_db"],
         "bdia_psnr_db": d["summary"][str(n)]["bdia_psnr_db"], "dpm_psnr_db": d["summary"][str(n)]["dpm_psnr_db"],
         "ddim_pass_s": d["summary"][str(n)]["ddim_pass_s"], "dpm_pass_s": d["summary"][str(n)]["dpm_pass_s"],
         "legacy_dense_anchor_db": anchor, "n_images": 4,
         "source": "paper_assets/exact_inv_sweep/sweep.json; AUD-20261027-01"}
        for n in nfes])


# Fig 3：classification utility 矩陣（null 部署）—— recovered≈real、cipher=chance
def fig3_utility():
    d = json.loads(UTIL.read_text(encoding="utf-8"))["results"]
    enc_label = {"densenet121_imagenet": "DenseNet121 (ImageNet)", "rad_dino": "RAD-DINO (CXR)"}
    encs = ["densenet121_imagenet", "rad_dino"]
    conds = [("train_real__test_real", "real→real", INK2),
             ("train_real__test_recovered", "real→recovered", BLUE),
             ("train_real__test_anon", "real→anon", ORANGE),
             ("train_anon__test_anon", "anon→anon", YELLOW)]
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    w = 0.2; x = np.arange(len(encs))
    for j, (key, lbl, col) in enumerate(conds):
        vals = [d[enc]["matrix"][key]["auroc"] for enc in encs]
        errs = [d[enc]["matrix"][key]["auroc_ci95"] for enc in encs]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, yerr=errs, capsize=2.5, color=col,
                      edgecolor="white", linewidth=1.0, label=lbl,
                      error_kw={"elinewidth": 0.9, "ecolor": INK2})
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.axhline(0.5, color=BASELINE, linestyle="--", linewidth=0.9)
    ax.text(len(encs) - 0.5, 0.515, "chance", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [enc_label[e] for e in encs])
    ax.set_ylim(0.3, 1.18)
    ax.set_ylabel("AUROC (No Finding vs Effusion), 50 patient-disjoint splits")
    ax.legend(fontsize=7.6, loc="upper center", ncol=4, columnspacing=1.4, handletextpad=0.5)
    ax.set_title("Null-deploy utility: recovery preserves classification; cipher stays at chance", fontsize=8.6)
    recon = d.get("densenet121_imagenet") and json.loads(UTIL.read_text(encoding="utf-8"))["recon_psnr_vs_x0_db"]["mean"]
    anon = json.loads(UTIL.read_text(encoding="utf-8"))["anon_psnr_vs_x0_db"]["mean"]
    fig.text(0.5, -0.04, f"recon {recon:.1f} dB, anon {anon:.1f} dB (null backward). real & recovered share the same render; "
             "both frozen-probe and fine-tune agree (see protocol).", ha="center", fontsize=6.6, color=INK2)
    fig.savefig(FIG / "w09_fig03_utility_matrix_null.png")
    plt.close(fig)
    write_csv("w09_tab03_utility_matrix_null.csv", [
        {"encoder": enc, "condition": key, "auroc": d[enc]["matrix"][key]["auroc"],
         "auroc_ci95": d[enc]["matrix"][key]["auroc_ci95"], "acc": d[enc]["matrix"][key]["acc"],
         "n_splits": d[enc]["matrix"][key]["n_splits"], "n_images": 400,
         "source": "paper_assets/classification_utility/anon_utility_dpm++_n200_s100_k50.json; AUD-20261029-01"}
        for enc in encs for key, _, _ in conds])


# Fig 4：AF-045 部署忠實學習式 re-id（正控制學得起來；加密通道回隨機）
def fig4_reid():
    d = json.loads(REID.read_text(encoding="utf-8"))
    victim = {r["channel"]: r for r in d["results"] if r["set"] == "victim_learned"}
    shadow = {r["channel"]: r for r in d["results"] if r["set"] == "shadow_val_learned"}
    order = [("p_deploy", "p_deploy\n(unencrypted, +control)", BLUE),
             ("s2_anon_deploy", "s2_anon_deploy\n(released PNG, headline)", ORANGE),
             ("s2_anon", "s2_anon\n(raw cipher latent, bound)", YELLOW)]
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    x = np.arange(len(order))
    vals = [victim[c]["auc"] for c, _, _ in order]
    colours = [col for _, _, col in order]
    bars = ax.bar(x, vals, 0.55, color=colours, edgecolor="white", linewidth=1.1)
    for b, (c, _, _) in zip(bars, order):
        v = victim[c]["auc"]; t1 = victim[c]["top1"]; rnd = victim[c]["random_top1"]
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"AUC {v:.3f}\ntop1 {t1:.2f} (rnd {rnd:.2f})",
                ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.axhline(0.5, color=BASELINE, linestyle="--", linewidth=0.9)
    ax.text(len(order) - 0.5, 0.515, "chance", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [lbl for _, lbl, _ in order], fontsize=7.3)
    ax.set_ylim(0.3, 0.85)
    ax.set_ylabel("victim pair-verification AUC\n(DPM++ N=100 null deploy)")
    ax.set_title("AF-045: deployment-faithful learned re-id — positive control learns, cipher channels random", fontsize=8.2)
    fig.text(0.5, -0.05, "Structurally-correct negative result: the attack HAS capacity (p_deploy shadow-val AUC "
             f"{shadow['p_deploy']['auc']:.3f}); S2 released-PNG & raw-cipher return to chance. Security = per-image nonce, "
             "sampler-agnostic. victim N=20, single key/checkpoint.", ha="center", fontsize=6.5, color=INK2)
    fig.savefig(FIG / "w09_fig04_learned_reid_dpm_deploy.png")
    plt.close(fig)
    write_csv("w09_tab04_learned_reid_dpm_deploy.csv", [
        {"channel": c, "victim_auc": victim[c]["auc"], "victim_top1": victim[c]["top1"],
         "random_top1": victim[c]["random_top1"], "shadow_val_auc": shadow[c]["auc"],
         "shadow_val_top1": shadow[c]["top1"], "train_patients": d["train_patients"], "victim_pairs": 20,
         "source": "paper_assets/tables/table_t2bb_learned_attack_dpm_deploy.json; AUD-20261029-01"}
        for c, _, _ in order])


# Fig 5：AF-046 多病理裁判（torchxrayvision 零重訓；recovered≈real 逐病理，anon~chance）
def fig5_referee():
    d = json.loads(REFEREE.read_text(encoding="utf-8"))
    m = d["models"]["densenet121-res224-all"]
    pp = m["per_pathology"]
    paths = sorted(pp, key=lambda p: pp[p]["auc_real"])
    real = [pp[p]["auc_real"] for p in paths]
    rec = [pp[p]["auc_recovered"] for p in paths]
    anon = [pp[p]["auc_anon"] for p in paths]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(len(paths))
    h = 0.38
    ax.barh(y + h / 2, real, h, color=INK2, edgecolor="white", linewidth=0.8, label="real")
    ax.barh(y - h / 2, rec, h, color=BLUE, edgecolor="white", linewidth=0.8, label="recovered")
    ax.plot(anon, y, linestyle="none", marker="D", markersize=4.5, color=ORANGE, label="anon")
    ax.axvline(0.5, color=BASELINE, linestyle="--", linewidth=0.9)
    ax.set_yticks(y, paths, fontsize=7.4)
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("per-pathology AUC (torchxrayvision densenet121-res224-all, n=400)")
    ax.legend(fontsize=7.6, loc="lower right")
    md = m["macro_delta_rec_minus_real"]
    ax.set_title(f"AF-046: recovery preserves multi-pathology detectability (macro Δ rec−real = {md:+.4f})", fontsize=8.4)
    fig.text(0.5, -0.03, "Independent zero-retrain referee refutes 'recovery erases other pathology'. anon ~chance. "
             "Fracture is chance even on real (256² referee blind spot, honestly reported); ML-proxy, not radiologist read.",
             ha="center", fontsize=6.5, color=INK2)
    fig.savefig(FIG / "w09_fig05_referee_pathology.png")
    plt.close(fig)
    rows = []
    for mdl in ("densenet121-res224-all", "densenet121-res224-chex"):
        mm = d["models"][mdl]
        for p, v in mm["per_pathology"].items():
            rows.append({"model": mdl, "pathology": p, "n_pos": v["n_pos"], "auc_real": v["auc_real"],
                         "auc_recovered": v["auc_recovered"], "auc_anon": v["auc_anon"],
                         "delta_rec_minus_real": v["delta_rec_minus_real"],
                         "source": "paper_assets/referee_pathology/referee_pathology_n400.json; AUD-20261029-01"})
        rows.append({"model": mdl, "pathology": "MACRO", "n_pos": "", "auc_real": mm["macro_auc"]["real"],
                     "auc_recovered": mm["macro_auc"]["recovered"], "auc_anon": mm["macro_auc"]["anon"],
                     "delta_rec_minus_real": mm["macro_delta_rec_minus_real"],
                     "source": "paper_assets/referee_pathology/referee_pathology_n400.json; AUD-20261029-01"})
    write_csv("w09_tab05_referee_pathology.csv", rows)


# Tab 6：AF-046 HF-FIDELITY（headline 須與分塊 min 並列；控制組 C1 到頂、C3 結構崩塌）
def tab6_hf():
    d = json.loads(HF.read_text(encoding="utf-8"))
    rec = d["recovered_vs_x0"]; c1 = d["controls"]["C1_x0_vs_x0"]; c3 = d["controls"]["C3_anon_vs_x0"]
    rows = [
        {"metric": "RAPSD retention (high band 0.5-1.0)", "recovered_mean": rec["rapsd_retention"]["high"]["mean"],
         "recovered_p05": rec["rapsd_retention"]["high"]["p05"], "recovered_min": rec["rapsd_retention"]["high"]["min"],
         "C1_x0_vs_x0": c1["rapsd_retention"]["high"]["mean"], "C3_anon_vs_x0": c3["rapsd_retention"]["high"]["mean"]},
        {"metric": "RAPSD retention (mid band)", "recovered_mean": rec["rapsd_retention"]["mid"]["mean"],
         "recovered_p05": rec["rapsd_retention"]["mid"]["p05"], "recovered_min": rec["rapsd_retention"]["mid"]["min"],
         "C1_x0_vs_x0": c1["rapsd_retention"]["mid"]["mean"], "C3_anon_vs_x0": c3["rapsd_retention"]["mid"]["mean"]},
        {"metric": "RAPSD retention (low band)", "recovered_mean": rec["rapsd_retention"]["low"]["mean"],
         "recovered_p05": rec["rapsd_retention"]["low"]["p05"], "recovered_min": rec["rapsd_retention"]["low"]["min"],
         "C1_x0_vs_x0": c1["rapsd_retention"]["low"]["mean"], "C3_anon_vs_x0": c3["rapsd_retention"]["low"]["mean"]},
        {"metric": "HP-SSIM", "recovered_mean": rec["hp_ssim"]["mean"], "recovered_p05": rec["hp_ssim"]["p05"],
         "recovered_min": rec["hp_ssim"]["min"], "C1_x0_vs_x0": c1["hp_ssim"]["mean"], "C3_anon_vs_x0": c3["hp_ssim"]["mean"]},
        {"metric": "Sobel gradient r", "recovered_mean": rec["sobel_grad_r"]["mean"], "recovered_p05": rec["sobel_grad_r"]["p05"],
         "recovered_min": rec["sobel_grad_r"]["min"], "C1_x0_vs_x0": c1["sobel_grad_r"]["mean"], "C3_anon_vs_x0": c3["sobel_grad_r"]["mean"]},
        {"metric": "Laplacian variance ratio", "recovered_mean": rec["lap_var_ratio"]["mean"], "recovered_p05": rec["lap_var_ratio"]["p05"],
         "recovered_min": rec["lap_var_ratio"]["min"], "C1_x0_vs_x0": c1["lap_var_ratio"]["mean"], "C3_anon_vs_x0": c3["lap_var_ratio"]["mean"]},
        {"metric": "32x32 block HF retention (mean)", "recovered_mean": rec["block_hf_retention_mean"]["mean"],
         "recovered_p05": rec["block_hf_retention_mean"]["p05"], "recovered_min": rec["block_hf_retention_mean"]["min"],
         "C1_x0_vs_x0": c1["block_hf_retention_mean"]["mean"], "C3_anon_vs_x0": c3["block_hf_retention_mean"]["mean"]},
        {"metric": "32x32 block HF retention (p05)", "recovered_mean": rec["block_hf_retention_p05"]["mean"],
         "recovered_p05": rec["block_hf_retention_p05"]["p05"], "recovered_min": rec["block_hf_retention_p05"]["min"],
         "C1_x0_vs_x0": c1["block_hf_retention_p05"]["mean"], "C3_anon_vs_x0": c3["block_hf_retention_p05"]["mean"]},
        {"metric": "32x32 block HF retention (min; worst patch)", "recovered_mean": rec["block_hf_retention_min"]["mean"],
         "recovered_p05": rec["block_hf_retention_min"]["p05"], "recovered_min": rec["block_hf_retention_min"]["min"],
         "C1_x0_vs_x0": c1["block_hf_retention_min"]["mean"], "C3_anon_vs_x0": c3["block_hf_retention_min"]["mean"]},
    ]
    for r in rows:
        r["source"] = "paper_assets/hf_fidelity/hf_fidelity.json; n=200; AUD-20261029-01"
    write_csv("w09_tab06_hf_fidelity.csv", rows)


if __name__ == "__main__":
    fig1_decode()
    fig2_accel()
    fig3_utility()
    fig4_reid()
    fig5_referee()
    tab6_hf()
    print("done")
