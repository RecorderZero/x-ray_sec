#!/usr/bin/env python3
"""Build the weekly paper appendix (figures, tables, MANIFEST) from committed results.

Every figure and table in ``appendix/week<N>/`` is produced here from files that
are tracked in Git; numbers are never typed by hand.  Each output is listed in
``appendix/week<N>/MANIFEST.csv`` with its SHA-256, the SHA-256 of every source
it was built from, the run IDs behind it, a draft caption and its intended use
in the paper.  See ``appendix/README.md`` for the conventions.

  scripts/run_cfg_ddim.sh python -m scripts.build_weekly_appendix --week 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scripts.output_guard import add_force_flag, refuse_overwrite

ROOT = Path(__file__).resolve().parents[1]

# Chart tokens (light mode, print); categorical slots 1-3 validated all-pairs
# (dataviz validate_palette.js: CVD dE 9.2, normal-vision dE 24.0).
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Item:
    item_id: str
    kind: str  # figure | table
    filename: str
    paper_use: str  # 正文主表 / 正文 / 附錄 / 僅內部
    section: str
    caption: str
    supports: str
    sources: list[str]
    run_ids: list[str] = field(default_factory=list)
    build: Callable[[Path, list[Path]], None] | None = None  # None -> copy sources[0]


# ---------------------------------------------------------------- styling
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- week 1 figures
def fig_storage_reversibility(out: Path, src: list[Path]) -> None:
    plt = _style()
    rows = [r for r in _read_csv(src[0]) if r["metric"] == "vs_x0_psnr_db"]
    schemes = ("P", "S0", "S1")
    storages = (("legacy_png", "legacy PNG (predecessor)"),
                ("range_preserving_png", "range-preserving PNG (ours)"),
                ("float", "float (upper bound)"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width, gap = 0.26, 0.02
    for s_index, (storage, label) in enumerate(storages):
        xs, means, lows, highs = [], [], [], []
        for g_index, scheme in enumerate(schemes):
            row = next(r for r in rows if r["scheme"] == scheme and r["storage"] == storage)
            mean = float(row["mean"])
            xs.append(g_index + (s_index - 1) * (width + gap))
            means.append(mean)
            lows.append(mean - float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]) - mean)
        ax.bar(xs, means, width=width, color=SLOT[s_index], label=label, zorder=2)
        ax.errorbar(xs, means, yerr=[lows, highs], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
    ax.set_xticks(range(len(schemes)), ["P (identity key)", "S0 Rademacher", "S1 Signed Perm."])
    ax.set_ylabel("M1 recovery PSNR vs x0 (dB)")
    ax.set_ylim(0, 45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8, handlelength=1.2, columnspacing=1.2)
    ax.set_title("Storage protocol, not the cipher, drives M1 reversibility (dev, n=20, 95% CI)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_pixel_cosine_vs_psnr(out: Path, src: list[Path]) -> None:
    plt = _style()
    data = json.loads(src[0].read_text(encoding="utf-8"))
    table = data["e_cosine_vs_psnr_table_unclipped_noise"]["rows"]
    psnr = np.array([row["achieved_psnr_db_mean"] for row in table])
    gap = 1 - np.array([row["cosine_mean"] for row in table])
    patients = data["a_different_patient_pixel_cosine"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhspan(1 - patients["p95"], 1 - patients["p05"], color=GRID, alpha=0.7, zorder=1, linewidth=0)
    ax.axhline(1 - patients["median"], color=INK_MUTED, linewidth=1, zorder=2)
    ax.text(10.4, (1 - patients["median"]) * 1.12,
            f"two different patients: median cosine {patients['median']:.3f} (p05–p95 band)",
            fontsize=8, color=INK_2)
    ax.plot(psnr, gap, color=SLOT[0], linewidth=2, marker="o", markersize=5, zorder=3,
            label="same image + Gaussian noise")
    ax.scatter([30.23], [1 - 0.998978], s=60, color=SLOT[1], marker="s", zorder=4,
               edgecolors="white", linewidths=1.5)
    ax.annotate("predecessor's reported\n'Cosine Sim 0.9989' (30.23 dB)", xy=(30.23, 1 - 0.998978),
                xytext=(31.5, 6e-3), fontsize=8, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8})
    ax.set_yscale("log")
    ax.set_xlabel("PSNR of the compared image pair (dB)")
    ax.set_ylabel("1 − pixel-space cosine (log)")
    ax.set_title("Pixel cosine saturates: 0.9989 only means about 30 dB (dev_v1.1, n=20)",
                 fontsize=10, color=INK)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_latent_vs_image(out: Path, src: list[Path]) -> None:
    plt = _style()
    legacy = _read_csv(src[0])
    ranged = _read_csv(src[1])
    panels = (("legacy PNG (predecessor)", legacy, "m1png_decrypted_latent_cosine", "m1png_vs_x0_psnr_db"),
              ("range-preserving PNG (ours)", ranged, "m1range_decrypted_latent_cosine", "m1range_vs_x0_psnr_db"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (title, rows, x_key, y_key) in zip(axes, panels):
        for index, scheme in enumerate(("P", "S0", "S1")):
            selected = [r for r in rows if r["scheme"] == scheme]
            ax.scatter([float(r[x_key]) for r in selected], [float(r[y_key]) for r in selected],
                       s=34, color=SLOT[index], marker=MARKERS[index], edgecolors="white",
                       linewidths=1.2, zorder=3, label=scheme)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("decrypted-latent cosine to z")
        ax.set_xlim(-0.15, 1.05)
    axes[0].set_ylabel("M1 recovery PSNR vs x0 (dB)")
    axes[0].legend(title="scheme", fontsize=8.5, title_fontsize=8.5, loc="lower left")
    fig.suptitle("Latent cosine does not predict image reversibility (dev, n=20 per scheme)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------- week 1 tables
def tab_environment(out: Path, src: list[Path]) -> None:
    env = json.loads(src[0].read_text(encoding="utf-8"))
    inventory = _read_csv(src[1])[0]
    rows = [
        ("python", env["python"].split(" ")[0]), ("pytorch", env["torch"]),
        ("cuda_build", env["torch_cuda_build"]), ("cudnn", env["cudnn"]),
        ("gpu", env["gpu_name"]), ("numpy", env["numpy"]), ("scipy", env["scipy"]),
        ("pillow", env["pillow"]), ("opencv", env["opencv"]),
        ("checkpoint_sha256", inventory["checkpoint_sha256"]),
        ("checkpoint_strict_load", inventory["strict_load"]),
        ("missing_unexpected_keys", f"{inventory['missing_key_count']}/{inventory['unexpected_key_count']}"),
        ("model_parameters", inventory["parameter_count"]),
        ("diffusion_timesteps", inventory["diffusion_timesteps"]),
    ]
    rows += [(f"model_{k}", v) for k, v in json.loads(inventory["model_config_json"]).items()]
    rows += [
        ("noise_level_T", "500 (DDIM, ddim1000 respacing)"),
        ("inversion_conditioning", "null=True (unconditional)"),
        ("generation_conditioning", "y=0 healthy class, single call, no CFG mixing (guidance_scale=-1)"),
        ("pipeline", "legacy_anonymization:ddim_sample_loop_anonymization"),
        ("batch_size_formal", "1 (bit-exact recomputation; see W1-T9)"),
        ("preprocessing", "grayscale; histogram equalization; INTER_AREA 256x256; JPEG q100; per-image min-max [0,1]"),
    ]
    _write_csv(out, [{"item": k, "value": v} for k, v in rows])


def tab_splits(out: Path, src: list[Path]) -> None:
    dev, formal, manifest = _read_csv(src[0]), _read_csv(src[1]), json.loads(src[2].read_text(encoding="utf-8"))

    def describe(name, rows, selection, seed, notes):
        labels = [r["label"] for r in rows]
        return {
            "split": name, "images": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "healthy_label0": labels.count("0"), "effusion_label1": labels.count("1"),
            "selection": selection, "seed": seed, "split_sha256": rows[0]["split_sha256"], "notes": notes,
        }

    _write_csv(out, [
        describe("dev_v1.1", dev, "CheXpert valid; first 10 patients per class by patient_id", "deterministic",
                 "dev/smoke only; hash excludes local_path"),
        describe("security_v1", formal, f"AF-012(a): legacy sample_n={manifest['legacy_sample_n_per_class']}/class "
                 f"random_state={manifest['legacy_random_state']} reproduced; excluded patients",
                 manifest["seed"], f"{manifest['excluded_patient_count']} training-sample patients excluded; "
                 "patient-disjoint; formal N=200 (user decision)"),
    ])


def tab_legacy_regression(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    rows = []
    for scheme in data["schemes"]:
        for key, value in scheme.items():
            if key.endswith("_max_abs"):
                rows.append({"scheme": scheme["scheme"], "check": key, "max_abs": value})
    _write_csv(out, rows)


def tab_latent_cache(out: Path, src: list[Path]) -> None:
    rows = []
    for path in src:
        data = json.loads(path.read_text(encoding="utf-8"))
        checks = data["checks"]
        rows.append({
            "split": Path(data["split"]).stem, "run_id": data["run_id"], "images": data["sample_count"],
            "latent_shape": "x".join(map(str, data["latent_shape"])), "batch_size": data["batch_size"],
            "recomputed": len(checks["recompute_sample_ids"]), "recompute_max_abs": checks["recompute_max_abs"],
            "recompute_bit_exact": checks["recompute_bit_exact_count"], "tolerance": checks["recompute_tolerance"],
            "finite": checks["finite_ok"], "status": data["status"], "cache_sha256": data["cache_file_sha256"],
        })
    _write_csv(out, rows)


def tab_pixel_cosine(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    a, c = data["a_different_patient_pixel_cosine"], data["c_different_patient_mean_removed_cosine"]
    b = data["b_same_image_plus_gaussian_noise"]["primary_exact_scaled_unclipped"]
    scale = data["d_cosine_one_without_equality"]["rescaling_y_equals_a_times_x"]["0.5"]["over_20_images"]
    rows = [
        {"comparison": "two different patients (190 pairs)", "psnr_db": a["pair_psnr_db"]["median"], "cosine": a["median"], "statistic": "median"},
        {"comparison": "two different patients, mean removed", "psnr_db": "", "cosine": c["median"], "statistic": "median"},
        {"comparison": "same image + noise", "psnr_db": 30.6, "cosine": b["30.6"]["cosine_mean"], "statistic": "mean of 20"},
        {"comparison": "same image + noise", "psnr_db": 26.2, "cosine": b["26.2"]["cosine_mean"], "statistic": "mean of 20"},
        {"comparison": "y = 0.5 x (not equal)", "psnr_db": scale["psnr_db_mean"], "cosine": scale["cosine_mean"], "statistic": "mean of 20"},
        {"comparison": "predecessor reported (base vs decrypt output)", "psnr_db": 30.23, "cosine": 0.998978, "statistic": "thesis 05_experiments.tex:275,308 (literature value)"},
    ]
    _write_csv(out, rows)


def tab_compute(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    rows = []
    for r in data["results"]:
        f, g, g0 = (r["estimated_forward_half_cycle_seconds_per_image"],
                    r["estimated_generation_half_cycle_seconds_per_image"],
                    r["estimated_generation_guidance0_half_cycle_seconds_per_image"])
        rows.append({"batch_size": r["batch_size"], "forward_half_cycle_s": round(f, 3),
                     "generation_half_cycle_s_guidance_m1": round(g, 3), "generation_half_cycle_s_guidance_0": round(g0, 3),
                     "full_cycle_s_guidance_m1": round(f + g, 3), "peak_vram_bytes": r["peak_vram_bytes"]})
    _write_csv(out, rows)


def tab_inference_flow(out: Path, src: list[Path]) -> None:
    rows = _read_csv(src[0])
    keys = ("image_psnr_db", "image_ssim", "latent_repeat_max_abs",
            "latent_reinversion_shared_noise_cosine", "latent_reinversion_unknown_noise_cosine",
            "runtime_seconds_primary_cycle")
    _write_csv(out, [{"metric": key, "mean_of_4": float(np.mean([float(r[key]) for r in rows])),
                      "min": float(np.min([float(r[key]) for r in rows])),
                      "max": float(np.max([float(r[key]) for r in rows]))} for key in keys])


# ---------------------------------------------------------------- registry
WEEKS: dict[int, list[Item]] = {
    1: [
        Item("W1-F1", "figure", "W1-F1_storage_reversibility.png", "正文", "Results／可逆性歸因",
             "M1 還原影像對原圖 x0 的 PSNR（dev_v1.1，每方案 20 張，95% bootstrap CI）。前作的 legacy PNG（per-image min-max、丟棄值域）使 S0/S1 只有約 31 dB；保存值域的 range-preserving PNG 讓 S0/S1 回到 38.4–39.2 dB，接近 float 交接上限；P 不受存檔方式影響。",
             "前作約 31 dB 的損失主要來自存檔流程，不是翻號／置換加密（AUD-20261010-05 裁決）",
             ["paper_assets/tables/table_m1_storage_comparison.csv"],
             ["E2.5_20261009T191319376995Z_f990c7d7", "AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8"],
             fig_storage_reversibility),
        Item("W1-F2", "figure", "W1-F2_pixel_cosine_vs_psnr.png", "正文", "Metric semantics／對前作指標的評析",
             "像素空間 cosine 與 PSNR 的關係（dev_v1.1 20 張加高斯雜訊，y 軸為 1−cosine 對數）。灰帶為不同病人兩兩 cosine 的 p05–p95。前作報告的 0.9989 對應約 30 dB，且不同病人的中位數已達 0.884。",
             "前作 0.9989 是輸出影像 cosine，只等於約 30 dB，不代表 latent 一致或零失真（AF-022）",
             ["results/E2.4_metric_semantics.json"], [], fig_pixel_cosine_vs_psnr),
        Item("W1-F3", "figure", "W1-F3_latent_vs_image_fidelity.png", "正文", "Results／指標選擇",
             "解密後 latent 與原 latent 的 cosine 對 M1 還原 PSNR（每點一張 dev 影像）。P 的 latent cosine 僅約 0.3 但影像 37.7 dB；S0/S1 在 legacy PNG 下 latent cosine 0.99 卻只有約 31 dB。latent 距離無法預測影像可逆性。",
             "可逆性必須以影像端指標判斷，latent 指標只作診斷（AF-019、AUD-20261010-02）",
             ["results/E2.5_roundtrip_per_sample.csv", "results/AF024_m1_storage_per_sample.csv"],
             ["E2.5_20261009T191319376995Z_f990c7d7", "AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8"],
             fig_latent_vs_image),
        Item("W1-F4", "figure", "W1-F4_roundtrip_grid.png", "正文", "Results／定性比較",
             "預先固定的 dev_v1.1_000（健康）與 _010（積水）在 P/S0/S1 下的原圖、匿名圖、T1 還原、legacy PNG M1 還原與固定色階 0–0.1 差異圖。",
             "S0/S1 匿名圖視覺上已打亂；legacy PNG 還原可見顆粒紋理與局部幻覺",
             ["image/E2.5_roundtrip_grid.png"], ["E2.5_20261009T191319376995Z_f990c7d7"]),
        Item("W1-F5", "figure", "W1-F5_P_anonymization_smoke.png", "附錄", "附錄／P 基線",
             "恆等金鑰 P 在學長加密流程上的 4 張 smoke：原圖、P 輸出、M1 float 與 PNG 還原、差異圖。可見還原圖出現原圖沒有的細小字樣狀紋理。",
             "P 基線走與 S0/S1 相同的流程（AF-017）；生成模型局部幻覺的觀察",
             ["image/AF017_P_anonymization_smoke.png"], ["AF017_P_SMOKE_20261009T152434977798Z_df76812e"]),
        Item("W1-F6", "figure", "W1-F6_inference_flow_smoke.png", "僅內部", "（不放論文；推論流程）",
             "推論流程（病灶定位用）4 張 smoke 的原圖、重建與差異圖。不是 P/S0/S1 比較的基線。",
             "僅作 E1.4 歷史紀錄",
             ["image/E1.4_ddim_smoke_contact_sheet.png"], ["E1.4_20261009T153121667310Z_aa04e79c"]),
        Item("W1-T1", "table", "W1-T1_environment_model.csv", "附錄", "Experimental setup",
             "實驗環境、checkpoint 與 diffusion 設定（含生成條件、T=500、batch 1、前處理）。",
             "實驗設定可重現（E1.1）", ["artifacts/environment_baseline.txt", "model_inventory.csv"], [], tab_environment),
        Item("W1-T2", "table", "W1-T2_splits.csv", "正文", "Experimental setup／資料",
             "dev_v1.1 與 security_v1 的樣本數、病人數、類別數、抽樣規則與 split hash。security_v1 排除了重現前作訓練抽樣的 22,253 位病人。",
             "評估資料與模型訓練資料 patient-disjoint（E1.3、E2.1）",
             ["splits/dev_v1.1.csv", "splits/security_v1.csv", "splits/security_v1_manifest.json"], [], tab_splits),
        Item("W1-T3", "table", "W1-T3_e0_canonical_preflight.csv", "附錄", "附錄／前置可行性",
             "E0 canonical synthetic preflight（SOT-WHT、CDF torus pad、KCI toy；d=4,096 與 65,536；N=100）。舊 artifact 已標為 historical。",
             "E0.1 evidence reset；WHT 保留 norm（norm-only Top-1=1.0）",
             ["canonical_preflight.csv"], []),
        Item("W1-T4", "table", "W1-T4_e2_2_legacy_regression.csv", "附錄", "附錄／前作重現",
             "E2.2：P/S0/S1 wrapper 與直接呼叫學長函式、guidance −1 與 0、deanonymize、transform round-trip 的 MaxAbs（T=500，皆為 0）。",
             "本專案以前作原始程式與參數逐位元重現前作流程（E2.2、AF-021）",
             ["results/E2.2.json"], [], tab_legacy_regression),
        Item("W1-T5", "table", "W1-T5_e2_3_latent_cache.csv", "附錄", "附錄／可重現性",
             "E2.3 latent cache：security_v1 200 張與 dev 20 張，batch 1 抽樣重算全部逐位元相同。batch 4 建 cache 的負結果（MaxAbs 6.93e-4）記錄於 reports/compute_budget.md。",
             "攻擊用的 latent cache 可逐位元重現；batch 組成會改變 GPU 結果",
             ["results/E2.3_DONE.json", "results/E2.3_dev_v1.1_cache.json"],
             ["E2.3_20261009T183736363217Z_84cf642a", "E2.3_20261009T190608331743Z_6edaf199"], tab_latent_cache),
        Item("W1-T6", "table", "W1-T6_e2_4_pixel_cosine_baselines.csv", "正文", "Metric semantics",
             "像素 cosine 的基準：不同病人、去均值、加雜訊、y=0.5x 反例與前作報告值。",
             "cosine=1 不等於相等；0.9989 只對應約 30 dB（E2.4、AF-022）",
             ["results/E2.4_metric_semantics.json"], [], tab_pixel_cosine),
        Item("W1-T7", "table", "W1-T7_e2_5_baseline_correctness.csv", "正文主表", "Results／前作可逆性基線",
             "E2.5 主表：P/S0/S1 的 transform-only、T1、M1（legacy PNG 與 float）影像與 latent 指標，平均與 95% bootstrap CI。",
             "前作在其實際流程下的可逆性（M1 legacy PNG 約 31 dB）",
             ["paper_assets/tables/table_baseline_correctness.csv"], ["E2.5_20261009T191319376995Z_f990c7d7"]),
        Item("W1-T8", "table", "W1-T8_m1_storage_comparison.csv", "正文", "Results／可逆性歸因",
             "AF-024：P/S0/S1 × legacy PNG／range-preserving PNG／float 的影像與 latent 指標（含攻擊者視角 |z| Pearson）。W1-F1 的數值表。",
             "存檔協定歸因；保存值域同時提升攻擊者保真度（AF-025）",
             ["paper_assets/tables/table_m1_storage_comparison.csv"],
             ["AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8"]),
        Item("W1-T9", "table", "W1-T9_compute_benchmark.csv", "附錄", "附錄／計算成本",
             "學長加密流程每張影像的 inversion 與生成半週期時間（batch 1/4/8；guidance −1 與 0）。",
             "正式實驗算力預算的依據（AF-021）",
             ["results/AF021_anonymization_benchmark.json"], ["AF021_ANON_BENCH_20261009T152410677702Z_a5f3dcdc"], tab_compute),
        Item("W1-T10", "table", "W1-T10_e1_4_inference_flow.csv", "僅內部", "（不放論文；推論流程）",
             "推論流程 smoke 的影像指標與 re-inversion cosine：共用雜訊 0.906 為樂觀上限，未知雜訊 0.371。",
             "0.906 不能當可逆性證據（AF-019）",
             ["results/E1.4_ddim_smoke.csv"], ["E1.4_20261009T153121667310Z_aa04e79c"], tab_inference_flow),
    ],
}

MANIFEST_FIELDS = ("id", "kind", "file", "paper_use", "section", "caption_zh", "supports_claim",
                   "sources", "run_ids", "sha256")


def build(week: int) -> Path:
    if week not in WEEKS:
        raise SystemExit(f"no registry for week {week}")
    target = ROOT / "appendix" / f"week{week}"
    for sub in ("figures", "tables"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in WEEKS[week]:
        sources = [ROOT / source for source in item.sources]
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise SystemExit(f"{item.item_id}: missing sources {missing}")
        out = target / ("figures" if item.kind == "figure" else "tables") / item.filename
        if item.build is None:
            shutil.copyfile(sources[0], out)
        else:
            item.build(out, sources)
        manifest.append({
            "id": item.item_id, "kind": item.kind, "file": str(out.relative_to(ROOT)),
            "paper_use": item.paper_use, "section": item.section, "caption_zh": item.caption,
            "supports_claim": item.supports,
            "sources": "; ".join(f"{source}@{sha256(ROOT / source)[:12]}" for source in item.sources),
            "run_ids": "; ".join(item.run_ids), "sha256": sha256(out),
        })
        print(f"{item.item_id}: {out.relative_to(ROOT)}")
    _write_csv(target / "MANIFEST.csv", manifest, list(MANIFEST_FIELDS))
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([ROOT / 'appendix' / f'week{args.week}' / 'MANIFEST.csv'], args.force)
    target = build(args.week)
    print(f"wrote {target.relative_to(ROOT)}/MANIFEST.csv")


if __name__ == "__main__":
    main()
