#!/usr/bin/env python3
"""Q10.1 error attribution: decompose the M1 reconstruction gap into engineering
stages, reading every number from the FROZEN formal fidelity table (never typed).

The ladder answers "where does the 0.9989-style residual come from?".  Stages,
from the cipher outward (image PSNR vs x0, latent cosine vs z):

  1. crypto_transform_only : structured transform round-trip, no diffusion, no
     storage. Latent cosine ~1.0, max_abs ~8e-7 -> the cipher is lossless
     (corroborated across dim/dtype/rounds by Q10.2 table_numeric_precision).
  2. m2_exact_latent       : M2 deployment -- exact latent payload decoded to an
     image, NO sampler inversion. The image ceiling (~41.9 dB); residual is VAE
     decode + 8-bit, not the cipher. Latent is byte-exact (Q10.4 AEAD floor).
  3. m1_sampler_range      : M1 -- image inverted by the sampler then regenerated,
     stored range-preserving. The drop from stage 2 is the DDIM invert/regen
     latent error (the dominant honest residual).
  4. m1_deploy_legacy      : predecessor legacy min-max PNG hand-off (P/S0/S1
     only). The further drop is STORAGE (range discarded), NOT the cipher; S2
     deploys range-preserving so it has no stage-4 loss (its advantage).

The conditioning axis (y=0 "pseudo-healthy" vs unconditional null generation) is
a sampler change measured in the GPU BDIA 2x2 appendix (AUD-20261015-02/03, n=3
diagnostic so far); it is cross-referenced here as a deferred row, not fabricated.

  scripts/run_cfg_ddim.sh python -m scripts.q10_error_attribution \
    --fidelity paper_assets/q8_formal/table_fidelity.csv \
    --out paper_assets/tables/table_error_attribution.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.output_guard import add_force_flag, refuse_overwrite

ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    """Path relative to repo root when possible, else the raw path (robust print)."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_fid(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell(fid, method, path, comparison, space, reference, metric, storage_sub=None):
    """Single frozen mean value; fail loudly if absent/ambiguous."""
    hits = [r for r in fid if r["method"] == method and r["path"] == path
            and r["comparison"] == comparison and r["space"] == space
            and r["reference"] == reference and r["metric"] == metric
            and (storage_sub is None or storage_sub in r["storage"])]
    vals = {round(float(r["mean"]), 12) for r in hits}
    if len(vals) != 1:
        raise SystemExit(f"expected one {method}/{path}/{comparison}/{space}/{reference}/{metric}"
                         f"/{storage_sub}, found {len(hits)} rows / {len(vals)} distinct values")
    return float(hits[0]["mean"])


def _row(method, order, stage, psnr, lat_cos, lat_max_abs, bit_exact, uint8_eq,
         prev_psnr, attribution, source, status="measured"):
    delta = (psnr - prev_psnr) if (psnr is not None and prev_psnr is not None) else ""
    return {
        "method": method, "stage_order": order, "stage": stage,
        "image_psnr_db_vs_x0": "" if psnr is None else round(psnr, 4),
        "delta_psnr_from_prev": "" if delta == "" else round(delta, 4),
        "latent_cosine_vs_z": "" if lat_cos is None else round(lat_cos, 9),
        "latent_max_abs_vs_z": "" if lat_max_abs is None else lat_max_abs,
        "float32_bit_exact_rate": "" if bit_exact is None else bit_exact,
        "uint8_pixel_equality_rate": "" if uint8_eq is None else uint8_eq,
        "attribution": attribution, "source": source, "status": status,
    }


def ladder(fid, method: str) -> list[dict]:
    rows = []
    # Stage 1: crypto transform only (latent space; image not applicable).
    t_cos = cell(fid, method, "transform_only", "marginal", "latent", "z", "cosine")
    t_max = cell(fid, method, "transform_only", "marginal", "latent", "z", "max_abs")
    t_be = cell(fid, method, "transform_only", "marginal", "latent", "z", "float32_bit_exact_rate")
    rows.append(_row(method, 1, "crypto_transform_only", None, t_cos, t_max, t_be, None,
                     None, "cipher round-trip lossless (max_abs~8e-7); corroborated by Q10.2",
                     "q8_formal/table_fidelity:transform_only"))
    # Stage 2: M2 exact-latent image ceiling (no sampler inversion).
    m2_psnr = cell(fid, method, "t1_exact_latent_m2", "marginal", "image", "x0", "psnr_db")
    m2_cos = cell(fid, method, "t1_exact_latent_m2", "marginal", "latent", "z", "cosine")
    m2_be = cell(fid, method, "t1_exact_latent_m2", "marginal", "latent", "z", "float32_bit_exact_rate")
    m2_u8 = cell(fid, method, "t1_exact_latent_m2", "marginal", "image", "x0", "uint8_pixel_equality_rate")
    rows.append(_row(method, 2, "m2_exact_latent", m2_psnr, m2_cos, None, m2_be, m2_u8,
                     None, "M2 ceiling: exact latent -> image; residual = VAE decode + 8-bit, not cipher",
                     "q8_formal/table_fidelity:t1_exact_latent_m2"))
    # Stage 3: M1 sampler inversion, stored range-preserving.
    if method == "S2":
        s3_psnr = cell(fid, method, "m1_image_inversion", "marginal", "image", "x0", "psnr_db",
                       "range_preserving_private_range_v2")
        s3_cos = cell(fid, method, "m1_image_inversion", "marginal", "latent", "z", "cosine",
                      "range_preserving_private_range_v2")
        s3_max = cell(fid, method, "m1_image_inversion", "marginal", "latent", "z", "max_abs",
                      "range_preserving_private_range_v2")
        s3_src = "q8_formal/table_fidelity:m1_image_inversion@range_preserving_private_range_v2"
    else:
        s3_psnr = cell(fid, method, "m1_image_inversion_attribution", "attribution_marginal", "image",
                       "x0", "psnr_db", "range_preserving_public_range_v1")
        s3_cos = cell(fid, method, "m1_image_inversion_attribution", "attribution_marginal", "latent",
                      "z", "cosine", "range_preserving_public_range_v1")
        s3_max = cell(fid, method, "m1_image_inversion_attribution", "attribution_marginal", "latent",
                      "z", "max_abs", "range_preserving_public_range_v1")
        s3_src = "q8_formal/table_fidelity:m1_image_inversion_attribution@range_preserving_public_range_v1"
    rows.append(_row(method, 3, "m1_sampler_range", s3_psnr, s3_cos, s3_max, None, None,
                     m2_psnr, "drop from M2 = DDIM invert/regen latent error (dominant honest residual)",
                     s3_src))
    # Stage 4: deployed storage. P/S0/S1 legacy min-max loss; S2 has none.
    if method == "S2":
        rows.append(_row(method, 4, "m1_deploy_legacy", s3_psnr, None, None, None, None,
                         s3_psnr, "S2 deploys range-preserving -> NO legacy min-max loss (its advantage)",
                         s3_src, status="not_applicable"))
    else:
        d_psnr = cell(fid, method, "m1_image_inversion", "marginal", "image", "x0", "psnr_db",
                      "legacy_png_handoff")
        rows.append(_row(method, 4, "m1_deploy_legacy", d_psnr, None, None, None, None,
                         s3_psnr, "drop from range = legacy min-max storage loss (STORAGE, not cipher)",
                         "q8_formal/table_fidelity:m1_image_inversion@legacy_png_handoff"))
    # Stage 5: conditioning axis -- deferred to GPU BDIA appendix, not fabricated.
    rows.append(_row(method, 5, "conditioning_null_vs_cond", None, None, None, None, None,
                     None, "y=0 cond vs unconditional null: sampler change; see GPU BDIA 2x2 appendix "
                     "(AUD-20261015-02/03, n=3 diagnostic ~+0.3-0.5 dB, not yet formal)",
                     "appendix/bdia (deferred)", status="deferred_gpu"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--fidelity", type=Path, default=ROOT / "paper_assets/q8_formal/table_fidelity.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_error_attribution.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    add_force_flag(parser)
    args = parser.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)

    fid = read_fid(args.fidelity)
    records = []
    for method in ("P", "S0", "S1", "S2"):
        records.extend(ladder(fid, method))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)

    # headline deltas for S0 (legacy scheme) and S2 (scheme of record)
    def delta(method, stage):
        return next(r["delta_psnr_from_prev"] for r in records if r["method"] == method and r["stage"] == stage)
    summary = {
        "schema_version": 1, "task_id": "Q10.1",
        "source_fidelity_sha256": hashlib.sha256(args.fidelity.read_bytes()).hexdigest(),
        "stages": ["crypto_transform_only", "m2_exact_latent", "m1_sampler_range",
                   "m1_deploy_legacy", "conditioning_null_vs_cond"],
        "s0_sampler_drop_db": delta("S0", "m1_sampler_range"),
        "s0_legacy_storage_drop_db": delta("S0", "m1_deploy_legacy"),
        "s2_sampler_drop_db": delta("S2", "m1_sampler_range"),
        "note": (
            "Numbers read from the frozen q8_formal fidelity table. Crypto transform is "
            "lossless (stage 1). The honest M1 residual is the sampler invert/regen error "
            "(stage 2->3). The large P/S0/S1 deployment drop (stage 3->4) is legacy min-max "
            "STORAGE loss, not the cipher; S2 avoids it by deploying range-preserving. The "
            "conditioning axis is deferred to the GPU BDIA appendix (not fabricated here)."
        ),
    }
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {_rel(args.out)} ({len(records)} rows); "
          f"S0 sampler drop={summary['s0_sampler_drop_db']} dB, "
          f"S0 legacy storage drop={summary['s0_legacy_storage_drop_db']} dB, "
          f"S2 sampler drop={summary['s2_sampler_drop_db']} dB")


if __name__ == "__main__":
    main()
