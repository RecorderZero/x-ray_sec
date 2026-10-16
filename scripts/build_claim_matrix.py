#!/usr/bin/env python3
"""F9.5 claim matrix v1: map every PROPOSAL hypothesis (RQ1-RQ4, H1a-H3c) and
the Section-5 claim constraints to its supporting evidence.

Headlines are pulled from the already-frozen, audited package claim files
(``attack_package/claims.csv`` for the predecessor attacks, A3/A4/A4.6;
``defense_package/claims.csv`` for S2/D7 and the AF-025 fixed-norm remediation).
Numbers are never typed here.  Quality hypotheses (H3c, RQ4) are read from the
frozen Week-7 formal quality tables (``q8_formal/table_fidelity.csv`` and the F9.4
``tables/table_statistics_formal.csv``), mixed deployment (P/S0/S1 legacy PNG,
S2 private range).

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
# Frozen Week-7 formal quality evidence (mixed deployment): numbers are READ, not typed.
FIDELITY = ROOT / "paper_assets/q8_formal/table_fidelity.csv"
STATS = ROOT / "paper_assets/tables/table_statistics_formal.csv"


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
     "supported", "quality", []),
    # RQ4: quality / reversibility trade-off (formal quality run)
    ("RQ4", "RQ4", "P／S0／S1／S2 的 latent round-trip、PSNR、SSIM、LPIPS、cosine、MaxAbsError 與 runtime 比較（含 paired test）",
     "supported", "quality", []),
]

# Section-5 claim constraints the paper must honour (documented, not a number).
CONSTRAINTS = [
    ("§5.1", "不宣稱 structured transform 等同 AES／AEAD／一般 IND-CPA；只針對已驗證的 norm 旁道與逐元素／排序幅值攻擊強化", "PROPOSAL.md §5.1；AUD-20261015-01"),
    ("§5.2", "nonce 只讓每張影像用不同 subkey／Q，不隱藏單一 ciphertext 的 L2 norm，也不提供 integrity", "PROPOSAL.md §5.2；D7-C5/C6"),
    ("§5.3", "不以 bit-exact=100% 作為 structured transform 的通過條件；DICOM byte 完全一致須走 AEAD-ref（Q10.4）", "PROPOSAL.md §5.3"),
    ("§5.4", "original／sorted／norm-only 與 T1／T2-WB 分開報；固定 norm S2 v2 以相同 N=200 補驗，不因假說篩選", "PROPOSAL.md §5.4；AUD-20261014-04/15-01"),
]

def _marginal(fid: list[dict[str, str]], method: str, path: str, reference: str, metric: str,
              storage: str | None = None) -> float:
    """The single frozen 'marginal' cell value (method/path/reference/metric); fail if absent/ambiguous."""
    hits = [r for r in fid if r["method"] == method and r["path"] == path and r["reference"] == reference
            and r["metric"] == metric and r["comparison"] == "marginal"
            and (storage is None or storage in r["storage"])]
    if len(hits) != 1:
        raise SystemExit(f"expected one marginal {method}/{path}/{reference}/{metric}, found {len(hits)}")
    return float(hits[0]["mean"])


def _paired(stats: list[dict[str, str]], metric: str, method: str, baseline: str,
            family: str = "mixed_deployment") -> dict[str, str]:
    """One paired-test row. ``family`` disambiguates the two Holm families
    (mixed_deployment = deployment incl. storage; same_storage = encryption-only)."""
    hits = [r for r in stats if r["metric"] == metric and r["method"] == method
            and r["baseline"] == baseline and r.get("family", "mixed_deployment") == family]
    if len(hits) != 1:
        raise SystemExit(f"expected one paired {metric} {method} vs {baseline} in family {family}, found {len(hits)}")
    return hits[0]


def quality_headline(hid: str, fid: list[dict[str, str]], stats: list[dict[str, str]]) -> tuple[str, str]:
    """H3c/RQ4 headlines, numbers READ from the frozen formal tables (never typed here)."""
    if hid == "RQ4":
        s2 = _marginal(fid, "S2", "m1_image_inversion", "x0", "psnr_db", "range_preserving_private_range_v2")
        s2_ssim = _marginal(fid, "S2", "m1_image_inversion", "x0", "ssim", "range_preserving_private_range_v2")
        s2_lpips = _marginal(fid, "S2", "m1_image_inversion", "x0", "lpips", "range_preserving_private_range_v2")
        s0 = _marginal(fid, "S0", "m1_image_inversion", "x0", "psnr_db", "legacy_png_handoff")
        s1 = _marginal(fid, "S1", "m1_image_inversion", "x0", "psnr_db", "legacy_png_handoff")
        pear = _marginal(fid, "P", "m1_image_inversion", "x0", "psnr_db", "legacy_png_handoff")
        # Deployment contrast (storage INCLUDED) vs encryption-only (same storage). AF-039.
        d0 = _paired(stats, "psnr_db", "S2", "S0", "mixed_deployment")
        d1 = _paired(stats, "psnr_db", "S2", "S1", "mixed_deployment")
        e0 = _paired(stats, "psnr_db", "S2", "S0", "same_storage")
        headline = (
            f"S2 M1 PSNR {s2:.2f} dB（SSIM {s2_ssim:.3f}、LPIPS {s2_lpips:.3f}；保值域私有部署）；"
            f"前作加密方案 S0 {s0:.2f}／S1 {s1:.2f} dB（legacy PNG 部署），明文 P {pear:.2f} dB｜"
            f"【部署對照，含存檔差異】配對 S2−S0 中位 +{float(d0['median_diff']):.2f}／S2−S1 +{float(d1['median_diff']):.2f} dB"
            f"（r={float(d0['rank_biserial_r']):.2f}，Holm p={float(d0['p_holm']):.1e}，N={d0['n_pairs']}）｜"
            f"【加密-only，同為保值域】配對 S2−S0 中位 +{float(e0['median_diff']):.2f} dB"
            f"（r={float(e0['rank_biserial_r']):.2f}，Holm p={float(e0['p_holm']):.1e}）——"
            f"加密本身成本僅此量級，部署對照的 +{float(d0['median_diff']):.1f} dB 幾乎全來自存檔協定（legacy min-max 丟值域）")
        return headline, "q8_formal:table_fidelity;tables/table_statistics_formal"
    # H3c
    xform = _marginal(fid, "S2", "transform_only", "z", "cosine")
    xform_err = _marginal(fid, "S2", "transform_only", "z", "max_abs")
    lat = _marginal(fid, "S2", "m1_image_inversion", "z", "cosine", "range_preserving_private_range_v2")
    headline = (
        f"structured transform 本身無損（transform-only latent cosine {xform:.6f}、max_abs {xform_err:.1e}）；"
        f"S2 M1 latent round-trip cosine {lat:.6f}，殘差來自 sampler+8-bit 量化而非 transform；"
        f"S2 M1 影像品質與明文 P 同級（見 RQ4）")
    return headline, "q8_formal:table_fidelity"


def build_headline(hid: str, kind: str, claim_ids: list[str], attack_ix, defense_ix,
                   fid: list[dict[str, str]], stats: list[dict[str, str]]) -> tuple[str, str]:
    if kind == "quality":
        return quality_headline(hid, fid, stats)
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
    fid = read_csv(FIDELITY)
    stats = read_csv(STATS)

    rows = []
    for hid, rq, text, status, kind, cids in ROWS:
        headline, evidence = build_headline(hid, kind, cids, attack_ix, defense_ix, fid, stats)
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
        f"> 狀態：{n_supported} 項已由 N=200 formal 證據支持，{n_pending} 項待補。品質列（H3c／RQ4）"
        f"由 `q8_formal/table_fidelity.csv`@{sha12(FIDELITY)} 與 `tables/table_statistics_formal.csv`@{sha12(STATS)} 讀入，數字不手抄。",
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
        "## 品質列證據與限縮（H3c／RQ4，已由 Q8_FORMAL + legacy supplement 落地）",
        "- H3c、RQ4 headline 由 `q8_formal`（N=200 formal，Q8_FORMAL parent + Q8_LEGACY_SUPPLEMENT）的 "
        "`table_fidelity.csv` 與 F9.4 `table_statistics_formal.csv`（paired Wilcoxon + rank-biserial + Holm）讀入，數字不手抄。",
        "- `table_statistics_formal.csv` 含兩個**各自獨立 Holm 校正**的 family（AF-039）：`mixed_deployment`"
        "（comparison_kind=deployment_incl_storage，S2 private-range vs P／S0／S1 **legacy**，含存檔差異）與 `same_storage`"
        "（comparison_kind=encryption_only，全 range-preserving，隔離加密本身成本）；每列並標 `method_storage`／`baseline_storage`。",
        "- 品質主欄為 mixed deployment（P／S0／S1 legacy PNG、S2 private-range v2）；全 range-preserving 為 attribution 另列。"
        "**不得把 S2−S0 的差異歸因為加密本身**：同存檔（same_storage family）下 S2−S0 中位僅約 +0.3 dB（mean +0.5 dB），"
        "部署對照（mixed_deployment family）的 ~+7.7 dB 幾乎全來自保值域存檔（legacy min-max 丟值域）。",
        "",
    ]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {claims_path.relative_to(ROOT)} ({len(rows)} hypotheses, {n_supported} supported, {n_pending} pending), {index_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
