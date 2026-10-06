#!/usr/bin/env python3
"""Build the tracked E0 evidence-reset summary from one canonical preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("canonical_preflight.csv"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/evidence_reset.md")
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    sot = payload["sot_wht"]
    cdf = payload["cdf_torus_pad"]
    kci = payload["kci_reversible_chain_toy"]
    source_hash = sha256(args.input)

    rows = [
        ("sot_wht", "float32_bit_exact_rate", sot["float32_bit_exact_rate"]),
        ("sot_wht", "max_abs_roundtrip_error_float64", sot["max_abs_roundtrip_error_float64"]),
        ("sot_wht", "norm_correlation", sot["norm_correlation"]),
        ("sot_wht", "magnitude_gallery_top1", sot["magnitude_gallery_top1"]),
        ("sot_wht", "norm_only_gallery_top1", sot["norm_only_gallery_top1"]),
        ("cdf_torus_pad", "max_abs_roundtrip_error_exact_latent", cdf["max_abs_roundtrip_error_exact_latent"]),
        ("cdf_torus_pad", "norm_correlation", cdf["norm_correlation"]),
        ("cdf_torus_pad", "magnitude_gallery_top1", cdf["magnitude_gallery_top1"]),
        ("cdf_torus_pad", "norm_only_gallery_top1", cdf["norm_only_gallery_top1"]),
        ("cdf_torus_pad", "known_pair_decrypts_second_message_cosine", cdf["known_pair_decrypts_second_message_cosine"]),
        ("kci_toy", "correct_key_max_abs_error", kci["correct_key_max_abs_error"]),
        ("kci_toy", "wrong_key_cosine", kci["wrong_key_cosine"]),
    ]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["schema_version", "source_sha256", "seed", "dimension", "gallery_size", "method", "metric", "value"]
        )
        for method, metric, value in rows:
            writer.writerow(
                [1, source_hash, payload["seed"], sot["dimension"], sot["gallery_size"], method, metric, value]
            )

    report = f"""# E0 Evidence Reset

## Canonical evidence

- Generator: `attack/validate_direction_candidates.py`
- Command: `conda run -n CFG_DDIM python attack/validate_direction_candidates.py --dimension {sot['dimension']} --gallery-size {sot['gallery_size']} --output {args.input}`
- Seed: `{payload['seed']}`
- Source JSON SHA-256: `{source_hash}`
- Tracked summary: `{args.csv}`

The JSON above and `canonical_preflight.csv` are the only canonical synthetic preflight for the current code revision. Earlier `direction_candidates_cpu*.json` files are **historical, non-canonical artifacts** because their gallery sizes differ. Their 31/32, 98/100, or similar bit-exact counts must not be mixed with the current N=100 result.

## Current conclusions

- SOT-WHT float32 bit-exact vectors: `{sot['float32_bit_exact_rate'] * sot['gallery_size']:.0f}/{sot['gallery_size']}`. This is an implementation diagnostic, not a security or end-to-end image-quality guarantee.
- SOT-WHT preserves L2 norm (`r={sot['norm_correlation']:.6g}`), so norm-only exact-pair linkage remains possible (Top-1 `{sot['norm_only_gallery_top1']:.6g}`).
- SOT-WHT elementwise-magnitude gallery Top-1 is `{sot['magnitude_gallery_top1']:.6g}` for this synthetic gallery.
- Fresh-pad CDF removes the same simple linkage in this run (norm correlation `{cdf['norm_correlation']:.6g}`, norm-only Top-1 `{cdf['norm_only_gallery_top1']:.6g}`), but a reused pad is recovered by one known pair and decrypts a second message with cosine `{cdf['known_pair_decrypts_second_message_cosine']:.6g}`.
- KCI is only a toy algebraic cancellation test here: correct-key max error `{kci['correct_key_max_abs_error']:.6g}`; it is not evidence for O-BELM image quality or IND-CPA security.

## Withdrawn or bounded claims

- Withdraw any unqualified statement that SOT-WHT is always float32 bit-exact; the rate depends on dimension, implementation, and sampled vectors.
- Do not treat cosine 0.9989 as byte equality or lossless recovery.
- Do not cite these CPU Gaussian tests as checkpoint/CXR evidence, diagnostic-utility evidence, or a complete cryptographic proof.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
