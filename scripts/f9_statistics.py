#!/usr/bin/env python3
"""F9.4 statistical comparison: paired tests, effect size and a fixed
multiple-comparison rule for the Q8 M1 reconstruction fidelity.

For each fidelity metric and each (method, baseline) pair, pair the per-sample
values by ``sample_id`` and run a two-sided Wilcoxon signed-rank test; report
the median paired difference and the matched-pairs rank-biserial correlation as
effect size.  All (metric x comparison) tests form one family corrected with
Holm-Bonferroni at alpha=0.05 -- the multiple-comparison rule is fixed here, not
chosen per result.  Direction (higher-is-better) is encoded per metric so the
sign of the effect is interpretable.

Runs on the deployment long table (pilot now; the same command runs on the
Week-7 formal output once Q8_FORMAL + the legacy supplement land).

  scripts/run_cfg_ddim.sh python -m scripts.f9_statistics \
    --long-csv paper_assets/q8_pilot_deployment/per_sample_long.csv \
    --manifest paper_assets/q8_pilot_deployment/analysis_manifest.json \
    --out paper_assets/tables/table_statistics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import wilcoxon

from scripts.output_guard import add_force_flag, refuse_overwrite

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05

# (group, metric, space, higher_is_better) for the main mixed-deployment M1.
METRICS = [
    ("m1", "psnr_db", "image", True),
    ("m1", "ssim", "image", True),
    ("m1", "lpips", "image", False),
    ("m1", "linf", "image", False),
    ("m1lat", "cosine", "latent", True),
    ("m1lat", "rmse", "latent", False),
    ("m1lat", "max_abs", "latent", False),
]
# Paired comparisons (method vs baseline); S2 is the remediated scheme of record.
COMPARISONS = [("S2", "S0"), ("S2", "S1"), ("S2", "P"), ("S0", "P"), ("S1", "P")]


def read_long(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_values(rows, group, metric, method, baseline):
    """Return (a_values, b_values) paired by sample_id for method vs baseline."""
    def by_sample(m):
        return {r["sample_id"]: float(r["value"]) for r in rows
                if r["group"] == group and r["metric"] == metric and r["method"] == m and r["value"] != ""}
    a, b = by_sample(method), by_sample(baseline)
    common = sorted(set(a) & set(b))
    return [a[s] for s in common], [b[s] for s in common]


def rank_biserial(diffs: list[float]) -> float:
    """Matched-pairs rank-biserial correlation: (R+ - R-) / (R+ + R-).
    +1 means method > baseline on every nonzero pair."""
    nz = [d for d in diffs if d != 0]
    if not nz:
        return 0.0
    import statistics  # noqa: F401 (kept explicit; ranking below is manual)
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = [0.0] * len(nz)
    i = 0
    while i < len(nz):
        j = i
        while j + 1 < len(nz) and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1  # average rank (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(rk for d, rk in zip(nz, ranks) if d > 0)
    r_neg = sum(rk for d, rk in zip(nz, ranks) if d < 0)
    total = r_pos + r_neg
    return (r_pos - r_neg) / total if total else 0.0


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--long-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_statistics.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    add_force_flag(parser)
    args = parser.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)

    rows = read_long(args.long_csv)
    manifest = json.loads(args.manifest.read_text())
    evidence_role = "formal" if manifest.get("formal") else manifest.get("evidence_role", "pilot")

    records, pvals = [], []
    for group, metric, space, higher in METRICS:
        for method, baseline in COMPARISONS:
            a, b = paired_values(rows, group, metric, method, baseline)
            n = len(a)
            diffs = [x - y for x, y in zip(a, b)]
            median_diff = sorted(diffs)[n // 2] if n else float("nan")
            mean_diff = sum(diffs) / n if n else float("nan")
            if n >= 1 and any(d != 0 for d in diffs):
                stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
                stat = float(stat)
            else:
                stat, p = 0.0, 1.0
            records.append({
                "metric": metric, "space": space, "higher_is_better": higher,
                "method": method, "baseline": baseline, "n_pairs": n,
                "median_diff": median_diff, "mean_diff": mean_diff,
                "wilcoxon_stat": stat, "p_raw": float(p),
                "rank_biserial_r": rank_biserial(diffs),
            })
            pvals.append(float(p))

    for rec, padj in zip(records, holm(pvals)):
        rec["p_holm"] = padj
        rec["significant_holm_0.05"] = bool(padj < ALPHA)
        rec["evidence_role"] = evidence_role

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)

    summary = {
        "schema_version": 1, "task_id": "F9.4", "evidence_role": evidence_role,
        "source_long_csv_sha256": __import__("hashlib").sha256(args.long_csv.read_bytes()).hexdigest(),
        "n_samples": manifest.get("n") or manifest.get("sample_count"),
        "test": "two-sided Wilcoxon signed-rank (paired by sample_id, zero_method=wilcox)",
        "effect_size": "matched-pairs rank-biserial correlation",
        "multiple_comparison": f"Holm-Bonferroni across all {len(records)} (metric x comparison) tests, alpha={ALPHA}",
        "family_size": len(records), "comparisons": [f"{m}-{b}" for m, b in COMPARISONS],
        "metrics": [m for _, m, _, _ in METRICS],
        "note": "pilot N=20 validates the method; Week-7 formal N=200 replaces it. Mixed deployment: P/S0/S1 legacy, S2 private-range; do not attribute S2-S0 to the cipher alone (storage effect included).",
    }
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sig = sum(1 for r in records if r["significant_holm_0.05"])
    print(f"wrote {args.out.relative_to(ROOT)} ({len(records)} tests, {sig} significant after Holm; role={evidence_role})")


if __name__ == "__main__":
    main()
