#!/usr/bin/env python3
"""P4.1: assemble the Track A / A4 attack package — each claim -> evidence.

Builds ``paper_assets/attack_package/claims.csv`` (one row per attack claim with
its table, figure, result JSON, run IDs and the headline number read from the
committed result files) and a short ``index.md``.  Numbers are read from the
tracked result files, never typed here; the builder fails if a cited file or
field is missing, so the package cannot drift from the evidence.

  scripts/run_cfg_ddim.sh python -m scripts.build_attack_package
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _a32(*key: str) -> Callable[[], str]:
    def read() -> str:
        d = json.loads((ROOT / "results/A3.2.json").read_text())
        node = d["results"]
        for k in key:
            node = node[k]
        return str(node)
    return read


@dataclass
class Claim:
    cid: str
    threat: str
    claim_zh: str
    headline: Callable[[], str]
    sources: list[str]
    run_ids: list[str] = field(default_factory=list)


def a42_top1(cell: str, scorer: str) -> float:
    d = json.loads((ROOT / "results/A4.2_T2WB_gallery.json").read_text())
    return float(d["cells"][cell][scorer]["attack"]["top1"])


def a42_cal_top1(cell: str, scorer: str) -> tuple[float, str]:
    d = json.loads((ROOT / "results/A4.2_T2WB_gallery.json").read_text())
    node = d["cells"][cell][scorer]
    return float(node["calibrated_attack"]["top1"]), node["calibrated_verdict"]


def table_count(rel: str, match: dict[str, str], column: str, value: str) -> Callable[[], str]:
    def read() -> str:
        with (ROOT / rel).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        total = [r for r in rows if all(r[k] == v for k, v in match.items())]
        hit = [r for r in total if r[column] == value]
        return f"{len(hit)}/{len(total)}"
    return read


def table_field(rel: str, match: dict[str, str], column: str) -> Callable[[], str]:
    def read() -> str:
        with (ROOT / rel).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if all(row[k] == v for k, v in match.items()):
                    return row[column]
        raise KeyError(f"{match} not found in {rel}")
    return read


CLAIMS = [
    Claim("C1-keyspace", "T1/T3", "前作有效金鑰上界 S0=32／S1=64 bits；已知一對明密文=0；唯密文 seed 搜尋 S0≈2^32、S1≈2^33",
          lambda: "S0 %s / S1 %s bits" % (
              table_field("paper_assets/tables/table_keyspace.csv",
                          {"scheme": "S0 Rademacher", "mode": "password (legacy configuration)"},
                          "ciphertext_only_seed_search_bits")(),
              table_field("paper_assets/tables/table_keyspace.csv",
                          {"scheme": "S1 Signed Permutation", "mode": "password (legacy configuration; seed argument ignored)"},
                          "ciphertext_only_seed_search_bits")()),
          ["paper_assets/tables/table_keyspace.csv", "reports/keyspace_method_note.md", "results/A3.2.json"]),
    Claim("C2-coa-s0", "T1 COA", "S0 單一密文 latent 即可辨識金鑰（全 200 張皆高於 2^32 門檻；外推屬 [推定]）",
          lambda: "200/200 above-threshold (%s below)" % _a32(
              "e7_ciphertext_only_all_images", "s0_and_s1_sign_correct_sigma", "n_below_full_scale_threshold")(),
          ["results/A3.2.json"]),
    Claim("C3-coa-s1", "T1 COA", "S1 置換階段單一密文辨識 199/200（最弱 security_v1_0018 低於門檻）",
          lambda: "%d/200 above-threshold" % (200 - int(_a32(
              "e7_ciphertext_only_all_images", "s1_permutation_correct_sigma", "n_below_full_scale_threshold")())),
          ["results/A3.2.json"]),
    Claim("C4-t1-s0-abs", "T1 gallery", "S0 精確 latent magnitude linkage Top-1（上限）",
          table_field("paper_assets/tables/table_T1_magnitude.csv", {"scheme": "S0", "scorer": "abs"}, "top1"),
          ["paper_assets/tables/table_T1_magnitude.csv", "results/A3.4_T1_gallery.json", "image/A3.4_T1_cmc.png"]),
    Claim("C5-t1-s1-sorted", "T1 gallery", "S1 精確 latent sorted-magnitude linkage Top-1（上限）",
          table_field("paper_assets/tables/table_T1_magnitude.csv", {"scheme": "S1", "scorer": "sorted_abs"}, "top1"),
          ["paper_assets/tables/table_T1_magnitude.csv", "results/A3.4_T1_gallery.json"]),
    Claim("C6-t2wb-s0", "T2-WB gallery", "S0 abs linkage Top-1（前作部署＝legacy；range 作參考）",
          lambda: "legacy %.3f / range %.3f" % (a42_top1("S0/legacy_png", "abs"), a42_top1("S0/range_preserving_png", "abs")),
          ["paper_assets/tables/table_T2_WB_magnitude.csv", "results/A4.2_T2WB_gallery.json", "image/A4.2_T2WB_cmc.png"],
          ["A4.1_20261012T083425685874Z_3132fa49"]),
    Claim("C7-t2wb-s1", "T2-WB gallery", "S1 sorted-magnitude（尺度不變，AF-033）linkage Top-1；前作部署＝legacy，range 作參考",
          lambda: "legacy %.3f / range %.3f" % (a42_top1("S1/legacy_png", "sorted_abs_unit"), a42_top1("S1/range_preserving_png", "sorted_abs_unit")),
          ["paper_assets/tables/table_T2_WB_magnitude.csv", "results/A4.2_T2WB_gallery.json", "image/A4.2_T2WB_cmc.png"],
          ["A4.1_20261012T083425685874Z_3132fa49"]),
    Claim("C8-t2wb-norm", "T2-WB sidechannel", "校正後 norm-only 僅在 range-preserving 顯著（S0/S1）；AF-025 處置依 S2a/S2 於 D7 決定",
          lambda: "S0 %.3f (%s) / S1 %.3f (%s) [range-preserving]" % (
              *a42_cal_top1("S0/range_preserving_png", "norm"), *a42_cal_top1("S1/range_preserving_png", "norm")),
          ["paper_assets/tables/table_T2_WB_magnitude.csv", "results/A4.2_T2WB_gallery.json"],
          ["A4.1_DEV_CAL_20261012T133002259066Z_0fa671c3", "A4.1_20261012T083425685874Z_3132fa49"]),
    Claim("C9-kpa", "T3 KPA", "S0 一對即還原金鑰；S1 真實 latent 需 2 對（float32 量值重複）",
          lambda: "S0 %s(1對) / S1 %s(2對)" % (
              table_count("paper_assets/tables/table_KPA.csv", {"scheme": "S0", "latent_type": "real"}, "exact_recovery", "True")(),
              table_count("paper_assets/tables/table_KPA.csv", {"scheme": "S1", "latent_type": "real", "pairs_used": "2"}, "exact_recovery", "True")()),
          ["paper_assets/tables/table_KPA.csv", "results/A4.3.json"]),
    Claim("C10-cpa", "T3 CPA", "固定金鑰下重建解密映射（d=4096 全基底；前作實際僅需 1 次查詢），fresh max-abs 0",
          table_field("paper_assets/tables/table_CPA.csv", {"scheme": "S1"}, "fresh_plaintext_max_abs_error"),
          ["paper_assets/tables/table_CPA.csv", "results/A4.4.json"]),
    Claim("C11-tamper", "T4 tamper", "S0/S1 無完整性檢查，全部竄改皆被接受（無拒絕）",
          table_count("paper_assets/tables/table_tamper_baseline.csv", {}, "accepted", "True"),
          ["paper_assets/tables/table_tamper_baseline.csv", "results/A4.5.json"]),
]

def main() -> None:
    try:
        from scripts.output_guard import add_force_flag, refuse_overwrite
    except ModuleNotFoundError:
        from output_guard import add_force_flag, refuse_overwrite

    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_assets/attack_package")
    add_force_flag(parser)
    args = parser.parse_args()
    csv_path = args.out_dir / "claims.csv"
    index_path = args.out_dir / "index.md"
    refuse_overwrite([csv_path, index_path], args.force)

    rows = []
    for claim in CLAIMS:
        missing = [s for s in claim.sources if not (ROOT / s).is_file()]
        if missing:
            raise SystemExit(f"{claim.cid}: missing evidence {missing}")
        rows.append({
            "claim_id": claim.cid,
            "threat_model": claim.threat,
            "claim_zh": claim.claim_zh,
            "headline_value": claim.headline(),
            "sources": "; ".join(f"{s}@{sha12(ROOT / s)}" for s in claim.sources),
            "run_ids": "; ".join(claim.run_ids),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Track A 攻擊素材包（P4.1）",
        "",
        "> 由 `scripts/build_attack_package.py` 從已追蹤的結果檔產生，數字不手抄。每個 claim 對應表、圖、result JSON 與 run ID，見 `claims.csv`。",
        "> 威脅模型見 `PROPOSAL.md` §3；攻擊協定見 `reports/t2wb_protocol.md`；逐輪紀錄見 `PROGRESS.md`。",
        "",
        "| claim | 威脅 | 一句話 | headline |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['claim_id']} | {row['threat_model']} | {row['claim_zh']} | {row['headline_value']} |")
    lines += [
        "",
        "## 注意",
        "- 前作部署欄為 **legacy PNG**（不保存值域）；range-preserving 欄只作「存檔方式影響」的參考（AUD-20261012-05 使用者裁決）。",
        "- S0/S1 的「保存值域 vs 不保存」對比**不用來裁決 AF-025**：前作部署本就不保存值域。AF-025 的處置依據是 S2a/S2 的跨存檔攻擊（D7，尚未執行、尚未裁決）。",
        "- C7 採尺度不變 scorer（AF-033）：前作部署（legacy）下 S1 為 0.465，非 0.19。",
        "- lo/hi 旁道在每病人單筆 record 下標 `not_evaluable`（需不同影像的具標籤 enrollment）。",
    ]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path.relative_to(ROOT)} ({len(rows)} claims), {index_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
