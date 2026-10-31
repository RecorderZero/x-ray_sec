#!/usr/bin/env python3
"""把 Week 9（W9 加速取樣器＋null 部署＋utility/security 收尾）論文可用的原始檔
「複製」進 appendix_audit/week09_W9-accel-deploy/，並產生 MANIFEST.csv。

只複製小型檔案；不複製 latent cache、dataset、checkpoint、run 目錄。原檔不動。
本週以 spec-first ＋ 已 commit／凍結 artifact 逐值讀取為主（CODE／核對，未重跑 GPU），
故 `data/` 僅有實作方 `w09_src_*`；本週無稽核方獨立重算（`w09_aud_*`）。
執行（repo 根目錄）：
  scripts/run_cfg_ddim.sh python appendix_audit/week09_W9-accel-deploy/scripts/collect_assets.py
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEEK = Path(__file__).resolve().parents[1]

# (來源, 目的子資料夾, 產生者, 說明, 論文用途[, 目的檔名])
ASSETS = [
    # ---- 層 B 加速（AF-044 decode 分解＋取樣器掃描）
    ("paper_assets/rq4_redo/decompose.json", "data", "implementer", "AF-044 decode×sampler 二維分解（legacy/aligned DDIM N=499＋DPM++ N=100/200；decode 增益隨 scheme 變 P +2.03/S2fn +6.50 dB；n=4 dev）", "方法/附錄：decode×sampler 分解", "rq4_redo_decompose.json"),
    ("scripts/rq4_redo_decode_sampler.py", "docs", "implementer", "AF-044 decode×sampler 分解腳本（decode 增益=legacy→aligned DDIM N=499；sampler 增益=aligned DDIM→DPM++ N=100）", "方法：分解腳本", "rq4_redo_decode_sampler.py"),
    ("paper_assets/exact_inv_sweep/sweep.json", "data", "implementer", "層 B 加速掃描（DDIM/BDIA/DPM++ × NFE∈{25,50,100,200,499}；PSNR vs NFE＋legacy dense anchor 42.348；gate pass）", "方法：DPM++(2M) 加速", "exact_inv_sweep.json"),
    ("scripts/exact_inv_sweep.py", "docs", "implementer", "層 B exact-inversion 掃描腳本（DPM-Solver++(2M) data-prediction＋BDIA γ=1；近似反演非 bit-exact 已標註）", "方法：加速掃描腳本", "exact_inv_sweep.py"),
    ("new_model.md", "docs", "implementer", "W9 加速/部署規格（§9.3 decode 警語、§10.4 AF-045 範圍、§11.2 追蹤項含 RQ4 aligned-null 重報）", "方法/附錄：加速與部署規格", "new_model.md"),
    # ---- 最終部署慣例（null）＋ off-by-one 診斷
    ("paper_assets/respaced_ddim_sweep/backward_offbyone_check.json", "data", "implementer", "null 部署 off-by-one 診斷（forward bit-identical max_abs=0；A_aligned 47.36 / B_legacy 42.79 / C_shifted 6.40 dB）", "附錄：部署對齊診斷", "backward_offbyone_check.json"),
    ("paper_assets/respaced_ddim_sweep/backward_offset_sweep.json", "data", "implementer", "null 部署 backward offset 掃描（delta0/1/2/3=47.36/47.01/44.69/42.24；legacy≈delta2–3，非單步偏移）", "附錄：部署對齊診斷", "backward_offset_sweep.json"),
    # ---- classification utility（null 部署）＋外部泛化
    ("paper_assets/classification_utility/anon_utility_dpm++_n200_s100_k50.json", "data", "implementer", "classification utility 矩陣（null，n=400、50 splits、95%CI；real→recovered 0.909/0.937≈real，→anon 0.50 chance；recon 46.7 dB）", "結果：utility 矩陣", "anon_utility_dpm++_n200_s100_k50.json"),
    ("paper_assets/classification_utility/external_utility_nih_n150.json", "data", "implementer", "外部 NIH 泛化（真實影像零重訓；in-domain 0.900/0.902、external 0.823/0.836）", "結果：外部泛化", "external_utility_nih_n150.json"),
    ("paper_assets/classification_utility/privacy_utility_table.csv", "tables", "implementer", "隱私-utility 聯合表（單一 null run；病理 0.916/0.909、身份 P/S2 0.811/0.481）", "結果：隱私-utility 聯合表", "privacy_utility_table.csv"),
    ("paper_assets/classification_utility/privacy_utility_table.json", "data", "implementer", "隱私-utility 聯合表完整來源（各欄 n/scheme/provenance）", "結果：聯合表 provenance", "privacy_utility_table.json"),
    ("reports/classification_utility_protocol.md", "docs", "implementer", "classification utility 協定（Design A 二元 No Finding vs Effusion、patient-disjoint、frozen probe＋fine-tune、null 部署）", "方法：utility 協定", "classification_utility_protocol.md"),
    # ---- AF-045 部署忠實學習式 re-id
    ("paper_assets/tables/table_t2bb_learned_attack_dpm_deploy.csv", "tables", "implementer", "AF-045 部署忠實學習式 re-id（DPM++ N=100 null；p_deploy 學得起來 victim AUC 0.635、S2 回隨機）", "安全：學習式 re-id", "table_t2bb_learned_attack_dpm_deploy.csv"),
    ("paper_assets/tables/table_t2bb_learned_attack_dpm_deploy.json", "data", "implementer", "AF-045 學習式 re-id 完整結果（三通道 × shadow/victim/fixed；verdict net_learned/claim_random/efficacy）", "安全：re-id provenance", "table_t2bb_learned_attack_dpm_deploy.json"),
    ("scripts/t2bb_learned_attack.py", "docs", "implementer", "AF-045 學習式 re-id 攻擊（Siamese shadow-1000、victim 20×2；只消費反演 latent）", "方法：學習式攻擊", "t2bb_learned_attack.py"),
    ("scripts/t2bb_invert_cache.py", "docs", "implementer", "AF-045 部署忠實反演 cache（DPM++ N=100 null；獨立隔離）", "方法：反演 cache", "t2bb_invert_cache.py"),
    ("paper_assets/t2bb_dpm_parity/parity.json", "data", "implementer", "AF-045 frozen-encoder 連結 parity（DPM++ N=100；P 0.811/0.977 顯著 p=0.0002，S0/S1/S2 近隨機）", "安全：frozen-encoder 連結", "t2bb_dpm_parity.json"),
    # ---- AF-046 多病理裁判＋HF-FIDELITY
    ("paper_assets/referee_pathology/referee_pathology_n400.json", "data", "implementer", "AF-046 多病理裁判（torchxrayvision 零重訓 11 病理，n=400；recovered≈real macro Δ −0.003/−0.000，anon ~chance）", "utility：多病理保全", "referee_pathology_n400.json"),
    ("scripts/referee_pathology_check.py", "docs", "implementer", "AF-046 torchxrayvision 裁判腳本（densenet121-res224-all/-chex；recovered clip [0,1] 修復 c299e62）", "方法：多病理裁判腳本", "referee_pathology_check.py"),
    ("paper_assets/hf_fidelity/hf_fidelity.json", "data", "implementer", "AF-046 HF-FIDELITY（null，n=200；高頻 retention 0.964、HP-SSIM 0.988、Sobel-r 0.997，分塊 min 0.650；C1 到頂、C3 結構崩塌）", "utility：高頻保真 proxy", "hf_fidelity.json"),
    ("scripts/hf_fidelity_check.py", "docs", "implementer", "AF-046 HF-FIDELITY 腳本（稽核方規格：RAPSD 帶／HP-SSIM／Sobel-r／Laplacian／32² 分塊＋C1/C2/C3 控制）", "方法：HF 保真腳本", "hf_fidelity_check.py"),
    # ---- localization 負面結果
    ("reports/localization_evaluation_protocol.md", "docs", "implementer", "CheXlocalize 定位評估協定＋負面結果（新 pipeline 與前作皆 chance 等級；資料隔離、gs0 一致性修復）", "附錄：定位負面結果＋協定", "localization_evaluation_protocol.md"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    rows = []
    for entry in ASSETS:
        src, sub, by, desc, use = entry[:5]
        name = entry[5] if len(entry) > 5 else None
        s = ROOT / src
        if not s.is_file():
            print(f"SKIP (missing): {src}")
            continue
        prefix = "w09_src_" if by == "implementer" else "w09_aud_"
        dst = WEEK / sub / (prefix + (name or s.name))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dst)
        rows.append({"published_path": str(dst.relative_to(WEEK)), "source_path": src, "sha256": sha256(dst),
                     "bytes": dst.stat().st_size, "produced_by": by, "git_head_at_copy": head,
                     "description": desc, "paper_use": use})
    for p in sorted((WEEK / "figures").glob("w09_fig*.png")) + sorted((WEEK / "tables").glob("w09_tab*.csv")):
        rows.append({"published_path": str(p.relative_to(WEEK)), "source_path": "scripts/make_figures.py",
                     "sha256": sha256(p), "bytes": p.stat().st_size, "produced_by": "auditor (generated)",
                     "git_head_at_copy": head, "description": "本週摘要圖表（見 AUDIT_summary.md § Week 9）", "paper_use": "見 AUDIT_summary.md § Week 9 §7"})
    with (WEEK / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} entries; HEAD {head}")


if __name__ == "__main__":
    main()
