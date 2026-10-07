#!/usr/bin/env python3
"""Build the tracked E0 evidence reset from reproducible canonical inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_timing(item)
            for key, item in value.items()
            if "time" not in key.lower() and not key.lower().endswith("_ms_per_vector_python")
        }
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def stable_hash(payload: dict) -> str:
    data = json.dumps(without_timing(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def metric_rows(payload: dict, source_hash: str) -> list[list[object]]:
    rows = []
    for method, body in (
        ("sot_wht", payload["sot_wht"]),
        ("cdf_torus_pad", payload["cdf_torus_pad"]),
        ("kci_toy", payload["kci_reversible_chain_toy"]),
    ):
        for metric, value in body.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append([
                    2, source_hash, payload["seed"], body.get("dimension", ""),
                    body.get("gallery_size", ""), method, metric, value,
                ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4096", type=Path, required=True)
    parser.add_argument("--d65536", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("canonical_preflight.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/evidence_reset.md"))
    args = parser.parse_args()

    p4096 = json.loads(args.d4096.read_text(encoding="utf-8"))
    p65536 = json.loads(args.d65536.read_text(encoding="utf-8"))
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    h4096, h65536, hsweep = map(stable_hash, (p4096, p65536, sweep))
    rows = metric_rows(p4096, h4096) + metric_rows(p65536, h65536)
    for rounds, body in sweep["rounds"].items():
        for metric in ("passed", "total", "rate"):
            rows.append([2, hsweep, f"{sweep['seed_start']}..{sweep['seed_stop_exclusive'] - 1}", sweep["dimension"], sweep["seed_stop_exclusive"] - sweep["seed_start"], f"sot_wht_R{rounds}", f"float32_bit_exact_{metric}", body[metric]])

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["schema_version", "stable_source_sha256", "seed", "dimension", "gallery_size", "method", "metric", "value"])
        writer.writerows(rows)

    s4, s65 = p4096["sot_wht"], p65536["sot_wht"]
    c65 = p65536["cdf_torus_pad"]
    r1, r2, r4 = (sweep["rounds"][key] for key in ("1", "2", "4"))
    report = f"""# E0 Evidence Reset

## Canonical evidence

All commands must run through `scripts/run_cfg_ddim.sh`, which disables user-site packages.
Stable hashes below exclude runtime/timing fields but retain every scientific value.

| Source | Command summary | Stable SHA-256 |
|---|---|---|
| `{args.d4096}` | `validate_direction_candidates.py --dimension 4096 --gallery-size 100` | `{h4096}` |
| `{args.d65536}` | `validate_direction_candidates.py --dimension 65536 --gallery-size 100` | `{h65536}` |
| `{args.sweep}` | `validate_sot_bitexact_sweep.py --dimension 65536 --seed-start 0 --seed-stop 100 --rounds 1 2 4` | `{hsweep}` |

## Referenced-number reconciliation

| Prior reference | Current reconstruction | Status |
|---|---|---|
| d=65,536, one canonical gallery, 97/100 | `{s65['float32_bit_exact_rate'] * s65['gallery_size']:.0f}/{s65['gallery_size']}` (seed 1911 gallery) | REPRODUCED; this is a gallery-level rate |
| d=65,536, seeds 0–99, R=2, 98/100 | `{r2['passed']}/{r2['total']}` | REPRODUCED; distinct seed-sweep protocol |
| d=4,096 bit-exact 100% | `{s4['float32_bit_exact_rate'] * s4['gallery_size']:.0f}/{s4['gallery_size']}` (N={s4['gallery_size']}) | REPRODUCED for current N=100 protocol |
| R=1/2/4 seed sweep | `{r1['passed']}/{r1['total']}`, `{r2['passed']}/{r2['total']}`, `{r4['passed']}/{r4['total']}` | REPRODUCED |
| 2026-09-10 gallery=2000 Householder/SOT table in `defense_design_check.py` | no current script reproduces its full Householder table | HISTORICAL / WITHDRAWN from canonical claims |

The apparent 97/100 versus 98/100 difference is not a contradiction: the first tests 100 vectors in one seeded gallery with fixed transform keys; the second tests one vector under each of 100 RNG seeds. They are different estimands.

## Current conclusions

- d=4,096 SOT-WHT float32 bit-exact rate is `{s4['float32_bit_exact_rate']:.6g}`; d=65,536 is `{s65['float32_bit_exact_rate']:.6g}` for the gallery protocol.
- d=65,536 SOT KS statistic/p-value are `{s65['gaussian_ks_statistic']:.6g}` / `{s65['gaussian_ks_pvalue']:.6g}`; d=4,096 values are `{s4['gaussian_ks_statistic']:.6g}` / `{s4['gaussian_ks_pvalue']:.6g}`. A single p-value is reported, not used as proof of normality.
- SOT preserves L2 norm (`r={s65['norm_correlation']:.6g}`), so exact-pair norm linkage remains possible.
- Fresh-pad CDF has d=65,536 KS statistic/p-value `{c65['gaussian_ks_statistic']:.6g}` / `{c65['gaussian_ks_pvalue']:.6g}`, but reused pads remain broken by one known pair in this synthetic test.

## Claim boundaries

- SOT-WHT is not universally float32 bit-exact; the rate depends on dimension, rounds, implementation and sampled vectors.
- Cosine similarity is not byte equality or lossless recovery.
- CPU Gaussian tests do not establish checkpoint/CXR quality, diagnostic utility, IND-CPA, or IND-CCA security.
- The old gallery=2000 table is retained only as history; it is not cited as current evidence.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
