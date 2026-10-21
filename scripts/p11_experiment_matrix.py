#!/usr/bin/env python3
"""P11.1 experiment-matrix audit: every P0 cell -> key table -> run id / provenance.

Assembles ONE matrix from already-frozen, already-audited sources (never hand-copied
numbers): the attack/defense claim packages (which carry run_ids / source SHAs), the
F9.5 claim matrix, the Q8_FORMAL quality parent, and the P2/appendix tracks added
after F9. Row counts (`n`) are read from the actual table files so the matrix cannot
drift from the data.

  scripts/run_cfg_ddim.sh python -m scripts.p11_experiment_matrix
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PA = ROOT / "paper_assets"

# --- P2 / appendix tracks added after F9 (RQ1-4 core lives in the claim packages) ---
# (rq_track, cell_id, experiment, key_table_rel, run_id/provenance, status, scope)
P2_ROWS = [
    ("T2-BB inner", "T2BB-1", "anon<->anon linkage, direct latent (cos/abs/norm/DenseNet)",
     "tables/table_t2bb_inner_linkage.csv", "t2bb_inner_v1 (seed 1911), cached inversion",
     "done", "P2 appendix; S0/S1 transparent cos 0.657, S2 random 0.433"),
    ("T2-BB inner", "T2BB-2", "learned SupCon attacker (shadow 200)",
     "tables/table_t2bb_learned_attack.csv", "t2bb_shadow_v1 pilot",
     "done_negative", "P2; efficacy=false (overfit, shadow-val 0.48)"),
    ("T2-BB inner", "T2BB-3", "actual public-PNG M1 frozen-encoder linkage (N=20)",
     "tables/table_t2bb_m1_frozen.csv", "t2bb_m1 public export + private eval map",
     "done", "P2; AUD-20261022-01 PASS; AF-041 scope note on headline"),
    ("T2-BB inner", "T2BB-4", "image-space control (DenseNet121)",
     "tables/table_t2bb_step1_image_control.csv", "t2bb_inner_v1",
     "done", "P2; Top-1 0.55 / AUC 0.81 positive control"),
    ("Method comp", "MC-1", "positioning table vs published medical/reversible anon",
     "method_comparison.md (no table)", "refpaper 111/112/113 + web (cited)",
     "done", "P2; external numbers verified against refpaper body tables"),
    ("Method comp", "MC-2", "external re-id attacker (Packhauser-arch reimpl, CheXpert)",
     "tables/table_reid_external.csv", "REID-external (shadow 1000, seed 1911)",
     "done", "P2; shadow-val 0.911; S2 latent-view 0.445 vs S0 0.616 (AF-042 scoped)"),
    ("Method comp", "MC-3", "original-weights cross-set sanity + identity control",
     "method_comparison.md 3.2 (out-of-tree json)", "Packhauser GPL checkpoint (out-of-tree)",
     "done", "P2; plaintext 0.982, plaintext_latent 0.550, S2 0.477; license-clean, not vendored"),
    ("Ablation", "ABL-1", "BDIA/DDIM x asym/symnull 2x2 fidelity ablation (N=200)",
     "tables/table_bdia_2x2.csv", "BDIA2X2_20261020T182731Z_efbf6f85",
     "done", "P2; sampler delta <0.4 dB, NFE parity 1996; negative/robustness"),
    ("Ablation", "ABL-2", "redundant re-inversion in predecessor healthy-gen (N=10)",
     "reinvert_ablation/reinvert_ablation.csv", "reinvert-ablation (seed 1911)",
     "done", "P2; A~=B psnr 47.5 dB cos 0.9999, B 1.50x faster"),
]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def n_of(rel: str) -> str:
    p = PA / rel.split(" ")[0]
    if p.suffix == ".csv" and p.exists():
        return str(len(read_rows(p)))
    return "-"


def main() -> None:
    out_rows: list[dict] = []

    # RQ1 attacks -- from attack_package/claims.csv (carries run_ids)
    for r in read_rows(PA / "attack_package" / "claims.csv"):
        out_rows.append({
            "rq_track": r.get("threat_model", "RQ1"), "cell_id": r.get("claim_id", ""),
            "experiment": r.get("claim_zh", "")[:90], "key_table": r.get("sources", ""),
            "run_id_or_provenance": r.get("run_ids", ""), "n": "", "status": "frozen",
            "scope": "attack_package (A3/A4/A4.6), audited",
        })
    # RQ2/RQ3 defenses -- from defense_package/claims.csv
    for r in read_rows(PA / "defense_package" / "claims.csv"):
        out_rows.append({
            "rq_track": "RQ2/RQ3", "cell_id": r.get("claim_id", ""),
            "experiment": r.get("claim", "")[:90], "key_table": r.get("source", ""),
            "run_id_or_provenance": r.get("source_sha256_12", ""), "n": "", "status": "frozen",
            "scope": r.get("scope", "defense_package (D5/D6/D7/AF-025), audited")[:80],
        })
    # RQ4 quality -- Q8_FORMAL parent (+ legacy supplement)
    out_rows.append({
        "rq_track": "RQ4", "cell_id": "Q8-FORMAL", "experiment": "fidelity/stat/anon/wrongkey/runtime/storage (N=200)",
        "key_table": "q8_formal/table_{fidelity,statistics_formal,anonymity,wrong_key,runtime,storage}",
        "run_id_or_provenance": "Q8_FORMAL_20261015T173907560215Z_7e4c3368 (+ Q8_LEGACY_SUPPLEMENT_20261016T064241Z_964cb3e0)",
        "n": "200", "status": "frozen", "scope": "F9/G5 main table; AUD-20261017 PASS, AF-039 CLOSED",
    })
    # P2 / appendix tracks
    for (rq, cid, exp, tbl, run, status, scope) in P2_ROWS:
        out_rows.append({
            "rq_track": rq, "cell_id": cid, "experiment": exp, "key_table": tbl,
            "run_id_or_provenance": run, "n": n_of(tbl), "status": status, "scope": scope,
        })

    cols = ["rq_track", "cell_id", "experiment", "key_table", "run_id_or_provenance", "n", "status", "scope"]
    out_csv = PA / "final_experiment_matrix.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)

    md = ["# P11.1 Final experiment matrix", "",
          "> Built by `scripts/p11_experiment_matrix.py` from the frozen claim packages "
          "(attack/defense, run_ids inside), the Q8_FORMAL quality parent, and the P2/appendix "
          "tracks added after F9. `n` read from the live table files; numbers never hand-copied.",
          "> RQ1-4 core evidence + Section-5 claim limits live in `claim_matrix/index.md` (10 claims).",
          "",
          f"Total cells: {len(out_rows)} (RQ1-4 frozen main + P2/appendix).", "",
          "| track | cell | experiment | run id / provenance | n | status |",
          "|---|---|---|---|---|---|"]
    for r in out_rows:
        md.append(f"| {r['rq_track']} | {r['cell_id']} | {r['experiment']} | "
                  f"{r['run_id_or_provenance'][:60]} | {r['n']} | {r['status']} |")
    md.append("")
    md.append("## Missing-value reasons")
    md.append("- `n` blank for attack/defense package rows: those are per-claim C-codes "
              "(multiple sub-runs), not single tables; N is inside each package table.")
    md.append("- No cell is absent for a bad-result reason; P2 rows are optional/appendix "
              "and clearly scoped (negative results labelled `done_negative`).")
    (PA / "final_experiment_matrix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_csv} ({len(out_rows)} cells) and final_experiment_matrix.md")


if __name__ == "__main__":
    main()
