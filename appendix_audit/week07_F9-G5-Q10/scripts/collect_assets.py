#!/usr/bin/env python3
"""把 Week 7（F9／G5／Q10）論文可用的原始檔「複製」進 appendix_audit/week07_F9-G5-Q10/，並產生 MANIFEST.csv。

只複製小型檔案；不複製 latent cache、dataset、checkpoint、run 目錄。原檔不動。
每個檔案記錄：來源路徑、SHA-256、大小、產生者（implementer／auditor）、複製時的 git HEAD、用途說明。
執行（repo 根目錄）：
  scripts/run_cfg_ddim.sh python appendix_audit/week07_F9-G5-Q10/scripts/collect_assets.py
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
    # ---- 凍結主表（實作方）
    ("paper_assets/tables/table_statistics_formal.csv", "tables", "implementer", "F9.4 配對統計（Wilcoxon＋效果量＋Holm；mixed 與 same_storage 兩 family，每列自我說明存檔）", "本文：配對顯著性", "table_statistics_formal.csv"),
    ("paper_assets/q8_formal/table_fidelity.csv", "tables", "implementer", "F9.2 formal 還原品質（mixed 主欄＋range-attribution、patient-bootstrap CI）", "本文：還原品質主表", "table_fidelity_formal.csv"),
    ("paper_assets/tables/table_error_attribution.csv", "tables", "implementer", "Q10.1 誤差歸因階梯（crypto 無損→M2 41.87 dB→sampler→legacy 存檔）", "結果/附錄：誤差歸因", "table_error_attribution.csv"),
    ("paper_assets/tables/table_numeric_precision.csv", "tables", "implementer", "Q10.2 變換精度掃描（dim/dtype/rounds/perm；float32 worst 1.55e-6、float64 3.05e-15；AF-040 note 已與資料一致）", "附錄：變換精度", "table_numeric_precision.csv"),
    ("paper_assets/tables/table_aead_reference.csv", "tables", "implementer", "Q10.4 AEAD 下限（ChaCha20-Poly1305 round-trip byte-exact、竄改 fail-closed）", "附錄：M2 原始 bytes bit-exact 依據", "table_aead_reference.csv"),
    ("paper_assets/tables/table_partial_key.csv", "tables", "implementer", "Q10.5 金鑰雪崩（sign subkey Hamming≈0.5、ciphertext |cos|≈0；KDF 擴散診斷）", "附錄：金鑰雪崩診斷", "table_partial_key.csv"),
    ("paper_assets/claim_matrix/claims.csv", "tables", "implementer", "F9.5 claim-evidence matrix（10 假設全 resolved；RQ4 分部署對照 +7.68 與加密-only +0.30）", "本文：claim matrix", "claims.csv"),
    # ---- 視覺素材（實作方）
    ("image/Q10.5_partial_key_avalanche.png", "figures", "implementer", "Q10.5 金鑰雪崩圖（左 Hamming≈0.5、右 ciphertext |cos|≈0）", "附錄：金鑰雪崩診斷", "Q10.5_partial_key_avalanche.png"),
    # ---- 報告快照（實作方）
    ("reports/deployment_modes.md", "docs", "implementer", "Q10.3 M1／M2 部署對照（storage 引用逐格核對 formal 表）", "本文：部署模式", "deployment_modes.md"),
    ("reports/final_test_report.md", "docs", "implementer", "Z12.1 全測試報告（193 passed；稽核方自跑重現）", "附錄：測試凍結", "final_test_report.md"),
    # ---- 稽核方獨立量測
    ("audit/out/q10_verify.json", "data", "auditor", "Q10 spec-first 獨立重算（誤差歸因/變換精度/AEAD round-trip＋竄改/金鑰雪崩）", "附錄：獨立驗證", "q10_verify.json"),
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
        prefix = "w07_src_" if by == "implementer" else "w07_aud_"
        dst = WEEK / sub / (prefix + (name or s.name))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dst)
        rows.append({"published_path": str(dst.relative_to(WEEK)), "source_path": src, "sha256": sha256(dst),
                     "bytes": dst.stat().st_size, "produced_by": by, "git_head_at_copy": head,
                     "description": desc, "paper_use": use})
    for p in sorted((WEEK / "figures").glob("w07_fig*.png")) + sorted((WEEK / "tables").glob("w07_tab*.csv")):
        rows.append({"published_path": str(p.relative_to(WEEK)), "source_path": "scripts/make_figures.py",
                     "sha256": sha256(p), "bytes": p.stat().st_size, "produced_by": "auditor (generated)",
                     "git_head_at_copy": head, "description": "本週摘要圖表（見 AUDIT_summary.md § Week 7）", "paper_use": "見 AUDIT_summary.md § Week 7 §7"})
    with (WEEK / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} entries; HEAD {head}")


if __name__ == "__main__":
    main()
