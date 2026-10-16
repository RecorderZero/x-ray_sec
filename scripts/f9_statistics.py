#!/usr/bin/env python3
"""F9.4 statistical comparison: paired tests, effect size and a fixed
multiple-comparison rule for the Q8 M1 reconstruction fidelity.

For each fidelity metric and each (method, baseline) pair, pair the per-sample
values by ``sample_id`` and run a two-sided Wilcoxon signed-rank test; report
the median paired difference and the matched-pairs rank-biserial correlation as
effect size.  Direction (higher-is-better) is encoded per metric so the sign of
the effect is interpretable.

Two comparison families are reported, each corrected with its OWN
Holm-Bonferroni at alpha=0.05 (never pooled -- see AF-039):

  * ``mixed_deployment`` (comparison_kind=deployment_incl_storage): the real
    end-to-end deployment contrast -- S2 in the authenticated private-range
    container vs P/S0/S1 in the predecessor's legacy min-max PNG hand-off. The
    S2-S0/S1 gap here is dominated by the STORAGE protocol (legacy discards the
    latent range), NOT by the cipher.
  * ``same_storage`` (comparison_kind=encryption_only): the cipher-only contrast
    -- every method range-preserving (S2 private-range v2 vs P/S0/S1 public-range
    v1, both avoiding the legacy min-max loss). This isolates the cost of the
    S2 transform+container itself (~+0.5 dB vs S0), and is the number the paper
    must cite for "encryption barely costs quality".

Every row carries ``method_storage`` / ``baseline_storage`` so it is
self-describing and cannot be misread as a pure-cipher effect.

Runs on the deployment long table (pilot now; the same command runs on the
Week-7 formal output once Q8_FORMAL + the legacy supplement land). The
same_storage family requires the all-range arm (groups ``m1range``/
``m1rangelat``) to be present in the long table; if absent it is skipped with
a note.

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

# (metric, space, higher_is_better) for the M1 reconstruction-vs-x0 fidelity.
METRICS = [
    ("psnr_db", "image", True),
    ("ssim", "image", True),
    ("lpips", "image", False),
    ("linf", "image", False),
    ("cosine", "latent", True),
    ("rmse", "latent", False),
    ("max_abs", "latent", False),
]
# Paired comparisons (method vs baseline); S2 is the remediated scheme of record.
COMPARISONS = [("S2", "S0"), ("S2", "S1"), ("S2", "P"), ("S0", "P"), ("S1", "P")]

# Which long-table group holds each (family, space, method) cell. In the
# same_storage family the all-range arm lives in m1range/m1rangelat, except S2
# which only exists in its private-range container (m1/m1lat); both are
# range-preserving, so pairing them isolates the cipher cost.
_DEPLOY_GROUP = {"image": "m1", "latent": "m1lat"}
_RANGE_GROUP = {"image": "m1range", "latent": "m1rangelat"}


def group_for(family: str, space: str, method: str) -> str:
    if family == "mixed_deployment" or method == "S2":
        return _DEPLOY_GROUP[space]
    return _RANGE_GROUP[space]


def read_long(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def groups_present(rows) -> set[str]:
    return {r["group"] for r in rows}


def storage_for(rows, group, method) -> str:
    """The (non-empty) storage label recorded for this group+method, or ''."""
    for r in rows:
        if r["group"] == group and r["method"] == method and r["storage"]:
            return r["storage"]
    return ""


def paired_values(rows, group_a, group_b, metric, method, baseline):
    """Return (a_values, b_values) paired by sample_id for method vs baseline.

    ``method`` is read from ``group_a`` and ``baseline`` from ``group_b`` so the
    same_storage family can pair S2's private-range arm against the public-range
    arm of the other methods.
    """
    def by_sample(group, m):
        return {r["sample_id"]: float(r["value"]) for r in rows
                if r["group"] == group and r["metric"] == metric and r["method"] == m and r["value"] != ""}
    a, b = by_sample(group_a, method), by_sample(group_b, baseline)
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

    present = groups_present(rows)
    # mixed_deployment is always available; same_storage needs the all-range arm.
    families = [("mixed_deployment", "deployment_incl_storage")]
    skipped = []
    if _RANGE_GROUP["image"] in present and _RANGE_GROUP["latent"] in present:
        families.append(("same_storage", "encryption_only"))
    else:
        skipped.append("same_storage (all-range arm m1range/m1rangelat absent)")

    records = []
    for family, kind in families:
        fam_records, fam_pvals = [], []
        for metric, space, higher in METRICS:
            for method, baseline in COMPARISONS:
                g_a = group_for(family, space, method)
                g_b = group_for(family, space, baseline)
                a, b = paired_values(rows, g_a, g_b, metric, method, baseline)
                n = len(a)
                diffs = [x - y for x, y in zip(a, b)]
                median_diff = sorted(diffs)[n // 2] if n else float("nan")
                mean_diff = sum(diffs) / n if n else float("nan")
                if n >= 1 and any(d != 0 for d in diffs):
                    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
                    stat = float(stat)
                else:
                    stat, p = 0.0, 1.0
                fam_records.append({
                    "family": family, "comparison_kind": kind,
                    "metric": metric, "space": space, "higher_is_better": higher,
                    "method": method, "baseline": baseline,
                    "method_storage": storage_for(rows, g_a, method),
                    "baseline_storage": storage_for(rows, g_b, baseline),
                    "n_pairs": n,
                    "median_diff": median_diff, "mean_diff": mean_diff,
                    "wilcoxon_stat": stat, "p_raw": float(p),
                    "rank_biserial_r": rank_biserial(diffs),
                })
                fam_pvals.append(float(p))
        # Holm correction is PER FAMILY -- the two families are never pooled (AF-039).
        for rec, padj in zip(fam_records, holm(fam_pvals)):
            rec["p_holm"] = padj
            rec["significant_holm_0.05"] = bool(padj < ALPHA)
            rec["holm_family_size"] = len(fam_records)
            rec["evidence_role"] = evidence_role
        records.extend(fam_records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)

    per_family = {f: sum(1 for r in records if r["family"] == f) for f, _ in families}
    summary = {
        "schema_version": 2, "task_id": "F9.4", "evidence_role": evidence_role,
        "source_long_csv_sha256": __import__("hashlib").sha256(args.long_csv.read_bytes()).hexdigest(),
        "n_samples": manifest.get("n") or manifest.get("sample_count"),
        "test": "two-sided Wilcoxon signed-rank (paired by sample_id, zero_method=wilcox)",
        "effect_size": "matched-pairs rank-biserial correlation",
        "multiple_comparison": f"Holm-Bonferroni PER FAMILY (never pooled), alpha={ALPHA}",
        "families": {
            f: {"comparison_kind": kind, "family_size": per_family[f]}
            for f, kind in families
        },
        "families_skipped": skipped,
        "comparisons": [f"{m}-{b}" for m, b in COMPARISONS],
        "metrics": [m for m, _, _ in METRICS],
        "note": (
            "Two separately-corrected families. mixed_deployment: S2 private-range vs "
            "P/S0/S1 legacy -- the S2-S0/S1 gap is dominated by STORAGE (legacy min-max "
            "loss), NOT the cipher. same_storage (encryption_only): all methods "
            "range-preserving -- isolates the cipher cost (~+0.5 dB S2 vs S0). "
            "Cite same_storage for 'encryption barely costs quality'; cite "
            "mixed_deployment for the real predecessor-vs-S2 deployment contrast."
        ),
    }
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sig = sum(1 for r in records if r["significant_holm_0.05"])
    fam_str = ", ".join(f"{f}={per_family[f]}" for f, _ in families)
    skip_str = f"; skipped {skipped}" if skipped else ""
    print(f"wrote {args.out.relative_to(ROOT)} ({len(records)} tests [{fam_str}], "
          f"{sig} significant after per-family Holm; role={evidence_role}){skip_str}")


if __name__ == "__main__":
    main()
