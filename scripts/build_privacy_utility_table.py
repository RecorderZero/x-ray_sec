#!/usr/bin/env python3
"""Item 1: joint privacy-utility table on ONE pipeline (DPM++ N=100, S2fn).

Pulls three already-computed results onto a single ruler to show that the SAME
DPM++/S2 deployment pipeline, scored with the SAME frozen encoders, preserves
the PATHOLOGY signal (recovered ~= real) while destroying the IDENTITY signal
(S2 anon linkage ~= random):

  utility   : classification_utility_anon.py  (train real -> test real / recovered)
  identity  : t2bb_dpm_parity.py              (anon<->anon patient linkage, P..S2)
  external  : reid_external_attacker.py        (Packhauser Siamese, latent ciphertext)

No new compute; this is an assembly step.  Emits CSV + Markdown + JSON.

  scripts/run_cfg_ddim.sh python -m scripts.build_privacy_utility_table \
      --utility paper_assets/classification_utility/anon_utility_dpm++_n200_s100.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ENC_LABEL = {"densenet121_imagenet": "DenseNet-121 (ImageNet)", "rad_dino": "rad-dino (CXR)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--utility", type=Path,
                    default=Path("paper_assets/classification_utility/anon_utility_dpm++_n200_s100_k50.json"))
    ap.add_argument("--parity", type=Path, default=Path("paper_assets/t2bb_dpm_parity/parity.json"))
    ap.add_argument("--reid", type=Path, default=Path("paper_assets/tables/table_reid_external.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/classification_utility"))
    args = ap.parse_args()

    util = json.loads(args.utility.read_text())
    parity = json.loads(args.parity.read_text())
    reid = json.loads(args.reid.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # identity linkage AUC per (encoder, scheme) from the DPM++ parity run
    link = {(r["encoder"], r["scheme"]): r["auc"] for r in parity["results"]}
    # external attacker S2 (single Siamese, not per-encoder)
    reid_s2 = next((r for r in reid["results"] if r.get("scheme") == "S2"), {})
    reid_pt = next((r for r in reid["results"] if r.get("condition") == "plaintext"), {})

    rows = []
    for enc, res in util["results"].items():
        if "matrix" not in res:
            continue
        m = res["matrix"]
        rows.append({
            "encoder": ENC_LABEL.get(enc, enc),
            "pathology_real_auroc": m["train_real__test_real"]["auroc"],
            "pathology_recovered_auroc": m["train_real__test_recovered"]["auroc"],
            "pathology_anon_auroc": m["train_real__test_anon"]["auroc"],
            "identity_P_link_auc": link.get((enc, "P")),
            "identity_S2_link_auc": link.get((enc, "S2")),
        })

    meta = {
        "sampler": util.get("sampler"), "steps": util.get("steps"),
        "backward": "null (forward/backward aligned)",
        "utility_per_class": util.get("per_class"), "utility_test_n": util.get("test_n"),
        "utility_n_splits": util.get("n_splits"), "utility_scheme": util.get("scheme"),
        "recon_psnr_db": util.get("recon_psnr_vs_x0_db", {}).get("mean"),
        "parity_sampler": parity.get("sampler"), "parity_N": parity.get("N"),
        "parity_pairs": parity.get("pairs"), "parity_scheme": "S2",
        "external_attacker": reid.get("architecture"),
        "external_S2_top1": reid_s2.get("top1"), "external_S2_auc": round(reid_s2.get("auc", 0), 3) if reid_s2 else None,
        "external_random_top1": reid_s2.get("random_top1"),
        "external_plaintext_top1": reid_pt.get("top1"), "external_plaintext_auc": round(reid_pt.get("auc", 0), 3) if reid_pt else None,
        "per_column_n": {
            "pathology_utility": f"per_class {util.get('per_class')} (test {util.get('test_n')}/split x {util.get('n_splits')} patient-splits)",
            "identity_linkage": f"{parity.get('pairs')} patient pairs (gallery+query), scheme S2",
            "external_reid": f"{reid.get('architecture')} Siamese, latent ciphertext",
        },
        "note": ("Same DPM++ N=100 pipeline (null backward) and same frozen encoders throughout. "
                 "Pathology (utility) uses the S2fn deployment recovery; identity linkage is scored on the "
                 "S2 scheme (same per-image-nonce orthogonal family; S2fn is its fixed-norm fidelity variant, "
                 "both yield noise-like anon). Pathology AUROC near real after recovery; identity linkage AUC "
                 "near random (0.5) for S2; external latent-ciphertext attacker S2 at random top-1. "
                 "P (no encryption) is the identity-leak control. Per-column n differs by source and is stated "
                 "above (AF-047): utility n is per-class x splits, identity n is patient pairs."),
    }
    payload = {"meta": meta, "rows": rows}
    (args.out_dir / "privacy_utility_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    with (args.out_dir / "privacy_utility_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    md = ["# Privacy–Utility (DPM++ N={}, S2fn, recon {} dB)".format(meta["steps"], meta["recon_psnr_db"]), "",
          "| Encoder | Pathology real→real | Pathology real→**recovered** | Pathology real→anon | Identity P (link AUC) | Identity **S2** (link AUC) |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| {encoder} | {pathology_real_auroc:.3f} | **{pathology_recovered_auroc:.3f}** | "
                  "{pathology_anon_auroc:.3f} | {identity_P_link_auc:.3f} | **{identity_S2_link_auc:.3f}** |".format(**r))
    md += ["", f"External attacker ({meta['external_attacker']}): plaintext top-1 "
           f"{meta['external_plaintext_top1']} (AUC {meta['external_plaintext_auc']}) → "
           f"**S2 top-1 {meta['external_S2_top1']}** (AUC {meta['external_S2_auc']}; random "
           f"{meta['external_random_top1']}).", "",
           "**Per-column n (AF-047; sources differ, stated explicitly):**",
           f"- Pathology (utility): {meta['per_column_n']['pathology_utility']}",
           f"- Identity (linkage): {meta['per_column_n']['identity_linkage']}",
           f"- External (re-id): {meta['per_column_n']['external_reid']}",
           "", meta["note"]]
    (args.out_dir / "privacy_utility_table.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {args.out_dir}/privacy_utility_table.{{json,csv,md}}")


if __name__ == "__main__":
    main()
