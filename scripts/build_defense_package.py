#!/usr/bin/env python3
"""Build the traceable P7.1 defense package from frozen D7 tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.output_guard import add_force_flag, refuse_overwrite


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def select(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    found = [row for row in rows if all(row.get(key) == value for key, value in conditions.items())]
    if len(found) != 1:
        raise RuntimeError(f"expected one row for {conditions}, found {len(found)}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path("paper_assets/defense_package"))
    parser.add_argument("--figure", type=Path, default=Path("image/D7_attack_summary.png"))
    add_force_flag(parser)
    args = parser.parse_args()
    claims_path, index_path = args.package_root / "claims.csv", args.package_root / "index.md"
    refuse_overwrite([claims_path, index_path, args.figure], args.force)
    sources = {
        "attack": ROOT / "paper_assets/tables/table_attack_original.csv",
        "norm": ROOT / "paper_assets/tables/table_norm_leakage.csv",
        "cross": ROOT / "paper_assets/tables/table_cross_nonce.csv",
        "geometry": ROOT / "paper_assets/tables/table_geometry.csv",
        "repeat": ROOT / "paper_assets/tables/table_repeat.csv",
        "tamper": ROOT / "paper_assets/tables/table_tamper.csv",
        "inversion": ROOT / "paper_assets/tables/table_D7_T2WB_inversion.csv",
        "fixed_norm": ROOT / "paper_assets/tables/table_AF025_fixed_norm_norm_leakage.csv",
    }
    data = {name: read_csv(path) for name, path in sources.items()}
    abs_t1 = select(data["attack"], threat_model="T1", scorer="abs")
    abs_t2_legacy = select(data["attack"], threat_model="T2-WB", storage="legacy_png", scorer="abs")
    abs_t2_range = select(data["attack"], threat_model="T2-WB", storage="range_preserving_png", scorer="abs")
    norm_t1 = select(data["norm"], threat_model="T1")
    norm_legacy = select(data["norm"], threat_model="T2-WB", storage="legacy_png", calibration="dev_median")
    norm_range = select(data["norm"], threat_model="T2-WB", storage="range_preserving_png", calibration="dev_median")
    cpa_same = select(data["cross"], attack="CPA_basis", target="same_nonce")
    cpa_cross = select(data["cross"], attack="CPA_basis", target="cross_nonce")
    fixed = select(data["geometry"], mode="fixed_Q")
    per_image = select(data["geometry"], mode="per_image_Q")
    tamper_s2 = [row for row in data["tamper"] if row["method"] == "S2"]
    tamper_s2a = [row for row in data["tamper"] if row["method"] == "S2a"]
    repeat_unique = int(data["repeat"][0]["unique_payloads_total"])
    inv_legacy = select(data["inversion"], storage="legacy_png", metric="attacker_latent_cosine")
    inv_range = select(data["inversion"], storage="range_preserving_png", metric="attacker_latent_cosine")
    fixed_norm_t1 = select(data["fixed_norm"], threat_model="T1")
    fixed_norm_t2 = select(data["fixed_norm"], threat_model="T2-WB", calibration="dev_median_to_C")
    claims = [
        ("D7-C1", "T1 original-|z| linkage", f"Top-1={float(abs_t1['top1']):.3f} (random=0.005)", "attack"),
        ("D7-C2", "T2-WB original-|z| linkage across storage", f"legacy={float(abs_t2_legacy['top1']):.3f}; range={float(abs_t2_range['top1']):.3f}", "attack"),
        ("D7-C3", "Exact-latent norm limitation", f"T1 norm Top-1={float(norm_t1['top1']):.3f}", "norm"),
        ("D7-C4", "Storage-dependent adaptive norm linkage", f"T2-WB dev-calibrated Top-1 legacy={float(norm_legacy['top1']):.3f}; range={float(norm_range['top1']):.3f}", "norm"),
        ("D7-C5", "Full-map transfer across nonce", f"same nonce MaxAbs={float(cpa_same['max_abs']):.3g}; cross nonce cosine={float(cpa_cross['cosine_mean']):.4f}", "cross"),
        ("D7-C6", "Pairwise geometry", f"fixed-Q Top-1={float(fixed['distance_signature_top1']):.3f}; per-image-Q={float(per_image['distance_signature_top1']):.3f}", "geometry"),
        ("D7-C7", "Same-plaintext repeat", f"unique protected payloads={repeat_unique}/100", "repeat"),
        ("D7-C8", "Authentication", f"S2 rejected={sum(r['rejected']=='True' for r in tamper_s2)}/{len(tamper_s2)}; S2a rejected={sum(r['rejected']=='True' for r in tamper_s2a)}/{len(tamper_s2a)}", "tamper"),
        ("D7-C9", "T2-WB inversion positive-control quality", f"latent cosine legacy={float(inv_legacy['mean']):.6f}; range={float(inv_range['mean']):.6f}", "inversion"),
        ("D7-C10", "AF-025 fixed-norm remediation", f"norm Top-1 T1={float(fixed_norm_t1['top1']):.3f}; T2-WB={float(fixed_norm_t2['top1']):.3f}", "fixed_norm"),
    ]
    claim_rows = [{
        "claim_id": claim_id, "claim": claim, "headline": headline,
        "source": str(sources[source_name].relative_to(ROOT)),
        "source_sha256_12": sha12(sources[source_name]),
        "scope": "security_v1 N=200 unless explicitly synthetic diagnostic",
    } for claim_id, claim, headline, source_name in claims]
    write_csv(claims_path, claim_rows)

    labels = ("T1\nabs", "T2 legacy\nabs", "T2 range\nabs", "v1 T1\nnorm", "v1 T2 legacy\nnorm cal", "v1 T2 range\nnorm cal", "v2 T1\nnorm", "v2 T2\nnorm cal")
    values = [float(abs_t1["top1"]), float(abs_t2_legacy["top1"]), float(abs_t2_range["top1"]),
              float(norm_t1["top1"]), float(norm_legacy["top1"]), float(norm_range["top1"]),
              float(fixed_norm_t1["top1"]), float(fixed_norm_t2["top1"])]
    colours = ["#2a78d6"] * 3 + ["#eb6834"] * 3 + ["#2a9d66"] * 2
    fig, ax = plt.subplots(figsize=(12, 5.0))
    bars = ax.bar(labels, values, color=colours)
    ax.axhline(1 / 200, color="black", linestyle="--", linewidth=1, label="random Top-1 = 0.005")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Top-1 linkage rate"); ax.set_title("D7 S2 linkage and AF-025 fixed-norm remediation")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8); ax.legend()
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180); plt.close(fig)

    lines = [
        "# P7.1 Defense Package", "",
        "本素材包由 `scripts/build_defense_package.py` 從 D7 表格重建；headline 不手抄。",
        "v1 顯示正交 transform 的 norm limitation；AF-025 v2 加入固定 norm 與 private recovery metadata。", "",
        "## 主素材", "",
        "- 方法／整合圖：`image/D6.7_s2_dev.png`",
        "- 攻擊、norm limitation 與修補：`image/D7_attack_summary.png`、`paper_assets/tables/table_attack_original.csv`、`table_norm_leakage.csv`、`table_AF025_fixed_norm_norm_leakage.csv`",
        "- cross-nonce／geometry／probe／repeat／tamper：同目錄各 D7 表格。", "",
        "## Claim index", "",
        "| ID | Claim | Headline | Source |", "|---|---|---|---|",
    ]
    lines.extend(f"| {row['claim_id']} | {row['claim']} | {row['headline']} | `{row['source']}@{row['source_sha256_12']}` |" for row in claim_rows)
    lines.extend(["", "## 限制", "", "- magnitude／norm 與 geometry 結果是指定攻擊的實測，不是通用安全證明。",
                  "- D7.4 的顯式 full-map KPA/CPA 為 d=1,024 synthetic diagnostic；同 nonce positive control 通過。",
                  "- v1 range-preserving T2-WB norm linkage 明顯高於 random；AF-025 v2 的指定 T1/T2-WB norm attack 已回到 random，但仍不是通用安全證明。", ""])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {claims_path}, {index_path}, {args.figure}")


if __name__ == "__main__":
    main()
