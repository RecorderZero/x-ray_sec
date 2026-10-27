#!/usr/bin/env python3
"""把 Week 8（P2 附錄＋P11／Z12／Q11）論文可用的原始檔「複製」進 appendix_audit/week08_P2-P11/，並產生 MANIFEST.csv。

只複製小型檔案；不複製 latent cache、dataset、checkpoint、run 目錄。原檔不動。
注意：外部 re-id 的原作者權重 sanity（`/data2/external_gpl/...`）為 out-of-tree GPL
artifact，**刻意不複製進庫**；其數字已由 make_figures 讀入並凍存在 `w08_tab02_*`。
執行（repo 根目錄）：
  scripts/run_cfg_ddim.sh python appendix_audit/week08_P2-P11/scripts/collect_assets.py
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
    # ---- P2 附錄：T2-BB-M1-Frozen（實作方凍結表）
    ("paper_assets/tables/table_t2bb_m1_frozen.csv", "tables", "implementer", "T2-BB-M1-Frozen 黑箱 anon↔anon 連結（DenseNet121＋RAD-DINO；P 顯著、S0/S1/S2 近隨機）", "附錄：黑箱連結", "table_t2bb_m1_frozen.csv"),
    ("paper_assets/tables/table_t2bb_m1_frozen.json", "data", "implementer", "T2-BB-M1-Frozen 完整結果（AUC/CI/permutation p/encoder provenance/truth-isolation sha）", "附錄：黑箱連結 provenance", "table_t2bb_m1_frozen.json"),
    ("paper_assets/tables/table_bdia_2x2.csv", "tables", "implementer", "BDIA 2×2（sampler×對稱 一致性/成本 parity；negative/robustness，NFE=1996）", "附錄：sampler robustness", "table_bdia_2x2.csv"),
    ("reports/t2bb.md", "docs", "implementer", "T2-BB 協定與結果（§5 minmax/range_preserving、§6 frozen-encoder headline 綁雙重前提，AF-041 CLOSED）", "附錄：T2-BB 協定", "t2bb.md"),
    ("reports/method_comparison.md", "docs", "implementer", "方法家族對照＋§3 外部 re-id（§3.2 identity latent-view 控制 0.550；AF-042/AF-043 CLOSED 後措辭）", "相關研究/附錄：對照與 re-id", "method_comparison.md"),
    # ---- P11／Z12 凍結交付（實作方）
    ("paper_assets/reproduction/reproduction_report.json", "data", "implementer", "P11.2 關鍵結果重現（DDIM eta=0 反演 N=10，10/10 bit-exact，max_abs=0）", "附錄：可重現性", "reproduction_report.json"),
    ("paper_assets/freeze_manifest.json", "data", "implementer", "P11.7/Z12.4 freeze manifest（combined fingerprint、n_files=337、git head）", "附錄：凍結指紋", "freeze_manifest.json"),
    ("paper_assets/claim_audit_p11.md", "docs", "implementer", "P11.6 claim audit 報告（AF-043 後：詞表補精確還原、verdict 逐命中人工 disposition）", "寫作：claim 對照", "claim_audit_p11.md"),
    ("scripts/p11_claim_audit.py", "docs", "implementer", "P11.6 claim-audit scanner（BANNED 詞表含 lossless/IND-CPA/精確還原；bit-exact 刻意不 ban）", "方法：claim 掃描器", "p11_claim_audit.py"),
    ("reports/final_test_report.md", "docs", "implementer", "Z12.1 全測試報告（193 passed；稽核方自跑重現）", "附錄：測試凍結", "final_test_report.md"),
    ("paper_assets/final_experiment_matrix.csv", "tables", "implementer", "P11.1 實驗矩陣（34 列，每列 rq_track/cell_id/key_table/provenance@hash/status）", "附錄：實驗矩陣", "final_experiment_matrix.csv"),
    # ---- Q11 定位 parity（實作方）
    ("paper_assets/tables/table_localization_parity.csv", "tables", "implementer", "Q11 下游定位 parity（restored/healthy/heatmap × S2-S0/S2-S1/S0-S1，N=11）", "附錄：下游定位 parity", "table_localization_parity.csv"),
    ("image/localization_parity_grid.png", "figures", "implementer", "Q11 定位 parity 視覺格（原圖/密文/還原/健康/heatmap × S0/S1/S2）", "附錄：定位 parity 視覺", "localization_parity_grid.png"),
    # ---- 稽核方獨立量測
    ("audit/out/t2bb_m1_recompute.json", "data", "auditor", "T2-BB-M1-Frozen spec-first 獨立重算（8/8 AUC 吻合、sklearn 交叉驗、truth-isolation 檢查）", "附錄：獨立驗證", "t2bb_m1_recompute.json"),
    ("audit/out/p11_independent_verify.json", "data", "auditor", "P11/Z12 freeze 獨立驗證（337/337 checksum、combined fingerprint 重算、擴充詞表 over-claim 掃描）", "附錄：獨立驗證", "p11_independent_verify.json"),
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
        prefix = "w08_src_" if by == "implementer" else "w08_aud_"
        dst = WEEK / sub / (prefix + (name or s.name))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dst)
        rows.append({"published_path": str(dst.relative_to(WEEK)), "source_path": src, "sha256": sha256(dst),
                     "bytes": dst.stat().st_size, "produced_by": by, "git_head_at_copy": head,
                     "description": desc, "paper_use": use})
    for p in sorted((WEEK / "figures").glob("w08_fig*.png")) + sorted((WEEK / "tables").glob("w08_tab*.csv")):
        rows.append({"published_path": str(p.relative_to(WEEK)), "source_path": "scripts/make_figures.py",
                     "sha256": sha256(p), "bytes": p.stat().st_size, "produced_by": "auditor (generated)",
                     "git_head_at_copy": head, "description": "本週摘要圖表（見 AUDIT_summary.md § Week 8）", "paper_use": "見 AUDIT_summary.md § Week 8 §7"})
    with (WEEK / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} entries; HEAD {head}")


if __name__ == "__main__":
    main()
