#!/usr/bin/env python3
"""F9.5 claim matrix v1: map every PROPOSAL hypothesis (RQ1-RQ4, H1a-H3c) and
the Section-5 claim constraints to its supporting evidence.

Headlines are pulled from the already-frozen, audited package claim files
(``attack_package/claims.csv`` for the predecessor attacks, A3/A4/A4.6;
``defense_package/claims.csv`` for S2/D7 and the AF-025 fixed-norm remediation).
Numbers are never typed here.  Quality hypotheses (H3c, RQ4) depend on the
Week-7 formal quality run and are emitted as ``pending`` until that run lands,
so the matrix also documents exactly what the formal run must fill.

  scripts/run_cfg_ddim.sh python -m scripts.build_claim_matrix
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from scripts.output_guard import add_force_flag, refuse_overwrite

ROOT = Path(__file__).resolve().parents[1]
ATTACK = ROOT / "paper_assets/attack_package/claims.csv"
DEFENSE = ROOT / "paper_assets/defense_package/claims.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _index(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


# Each hypothesis -> (package, [claim_ids], status).  The headline is assembled
# from those claims' committed headline fields; the package file is the source.
ATTACK_HL = "headline_value"   # attack_package column holding the number
DEFENSE_HL = "headline"        # defense_package column holding the number

ROWS = [
    # RQ1: predecessor weaknesses reproduce on real CXR latents
    ("H1a", "RQ1", "S0 magnitude gallery attack 顯著高於 random baseline",
     "supported", "attack", ["C4-t1-s0-abs", "C6-t2wb-s0"]),
    ("H1b", "RQ1", "S1 sorted-magnitude attack 顯著高於 random baseline",
     "supported", "attack", ["C5-t1-s1-sorted", "C7-t2wb-s1"]),
    ("H1c", "RQ1", "重用 transform 時，一組已知明密文（或少數影像對）可危及其他影像",
     "supported", "attack", ["C9-kpa", "C10-cpa", "C12-imgonly-s0", "C13-imgonly-s1", "C14-recon-e2e"]),
    # RQ2: which attacks S2 patches
    ("H2a", "RQ2", "S2 顯著降低 original／sorted magnitude attack 至接近 random",
     "supported", "defense", ["D7-C1", "D7-C2"]),
    ("H2b", "RQ2", "不同 nonce 下，某張影像取得的 KPA／CPA／geometry 資訊不能直接解開其他影像",
     "supported", "defense", ["D7-C5", "D7-C6"]),
    ("H2c", "RQ2", "HMAC 對所有預定 bit／header／metadata tamper case 均拒絕（S2a 無認證作對照）",
     "supported", "defense", ["D7-C8"]),
    # RQ3: residual leakage
    ("H3a", "RQ3", "T1（精確 latent）norm correlation 保持 1；AF-025 固定 norm 將其降到 random",
     "supported_remediated", "defense", ["D7-C3", "D7-C10"]),
    ("H3b", "RQ3", "norm-only linkage 在 T1 成功，T2-WB 成功率取決於 inversion error／存檔；固定 norm 後兩者降到 random",
     "supported_remediated", "defense", ["D7-C4", "D7-C9", "D7-C10"]),
    ("H3c", "RQ3", "真實 inversion latent 並非理想 N(0,I)，structured transform 可能改變生成品質",
     "pending", "quality", []),
    # RQ4: quality / reversibility trade-off (formal quality run)
    ("RQ4", "RQ4", "P／S0／S1／S2 的 latent round-trip、PSNR、SSIM、LPIPS、cosine、MaxAbsError 與 runtime 比較（含 paired test）",
     "pending", "quality", []),
]

# Section-5 claim constraints the paper must honour (documented, not a number).
CONSTRAINTS = [
    ("§5.1", "不宣稱 structured transform 等同 AES／AEAD／一般 IND-CPA；只針對已驗證的 norm 旁道與逐元素／排序幅值攻擊強化", "PROPOSAL.md §5.1；AUD-20261015-01"),
    ("§5.2", "nonce 只讓每張影像用不同 subkey／Q，不隱藏單一 ciphertext 的 L2 norm，也不提供 integrity", "PROPOSAL.md §5.2；D7-C5/C6"),
    ("§5.3", "不以 bit-exact=100% 作為 structured transform 的通過條件；DICOM byte 完全一致須走 AEAD-ref（Q10.4）", "PROPOSAL.md §5.3"),
    ("§5.4", "original／sorted／norm-only 與 T1／T2-WB 分開報；固定 norm S2 v2 以相同 N=200 補驗，不因假說篩選", "PROPOSAL.md §5.4；AUD-20261014-04/15-01"),
]

PENDING_NOTE = "pending Week-7 formal quality run Q8_FORMAL (N=200, security_v1); F9.4 paired statistics"


def build_headline(kind: str, claim_ids: list[str], attack_ix, defense_ix) -> tuple[str, str]:
    if kind == "quality":
        return PENDING_NOTE, "Q8_FORMAL formal quality run (running)"
    ix, col, src = (attack_ix, ATTACK_HL, "attack_package") if kind == "attack" else (defense_ix, DEFENSE_HL, "defense_package")
    parts, ev = [], []
    for cid in claim_ids:
        if cid not in ix:
            raise SystemExit(f"claim id {cid} not found in {src}")
        parts.append(f"{cid}: {ix[cid][col]}")
        ev.append(cid)
    return " | ".join(parts), f"{src}:" + ";".join(ev)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_assets/claim_matrix")
    add_force_flag(parser)
    args = parser.parse_args()
    claims_path, index_path = args.out_dir / "claims.csv", args.out_dir / "index.md"
    refuse_overwrite([claims_path, index_path], args.force)

    attack_ix = _index(read_csv(ATTACK), "claim_id")
    defense_ix = _index(read_csv(DEFENSE), "claim_id")

    rows = []
    for hid, rq, text, status, kind, cids in ROWS:
        headline, evidence = build_headline(kind, cids, attack_ix, defense_ix)
        rows.append({
            "hypothesis": hid, "research_question": rq, "statement_zh": text,
            "status": status, "headline": headline, "evidence": evidence,
            "scope": "security_v1 N=200" if status != "pending" else "pending (Week 7)",
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with claims_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    n_supported = sum(1 for r in rows if r["status"].startswith("supported"))
    n_pending = sum(1 for r in rows if r["status"] == "pending")
    lines = [
        "# Claim matrix v1（F9.5）",
        "",
        "> 由 `scripts/build_claim_matrix.py` 從已凍結、已稽核的 `attack_package/claims.csv`（A3／A4／A4.6）與 "
        "`defense_package/claims.csv`（D5／D6／D7／AF-025）彙整；headline 數字不手抄。",
        f"> 來源：`attack_package/claims.csv`@{sha12(ATTACK)}、`defense_package/claims.csv`@{sha12(DEFENSE)}。",
        f"> 狀態：{n_supported} 項已由 N=200 formal 證據支持，{n_pending} 項待 Week-7 formal 品質 run（F9.4）。",
        "",
        "| 假設 | RQ | 一句話 | 狀態 | headline | 證據 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['hypothesis']} | {r['research_question']} | {r['statement_zh']} | {r['status']} | {r['headline']} | {r['evidence']} |")
    lines += ["", "## Section-5 宣稱限縮（必須在論文遵守）", "", "| 條款 | 內容 | 出處 |", "|---|---|---|"]
    for cid, text, src in CONSTRAINTS:
        lines.append(f"| {cid} | {text} | {src} |")
    lines += [
        "",
        "## 待補（Week 7 formal 品質 run 落地後）",
        "- H3c、RQ4：以 `Q8_FORMAL`（N=200）的 P／S0／S1／S2 latent round-trip、PSNR／SSIM／LPIPS／cosine／runtime 填入，並加 F9.4 paired test、effect size 與 multiple-comparison 規則。",
        "- 品質主欄為 mixed deployment（P／S0／S1 legacy、S2 private-range），另列全 range-preserving attribution；不得把 S2−S0 的差異歸因為加密本身（含存檔效應）。",
        "",
    ]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {claims_path.relative_to(ROOT)} ({len(rows)} hypotheses, {n_supported} supported, {n_pending} pending), {index_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
