#!/usr/bin/env python3
"""Build splits/t2bb_inner_v1.csv for the T2-BB inner-risk (anon<->anon) linkage
experiment (reports/t2bb_inner_linkage_protocol.md).

Unlike security_v1 (patient-disjoint, one image per patient), this split keeps
TWO frontal images per patient, taken from TWO distinct studies, so the same
patient appears as a {gallery, query} pair -- the enrollment set that inner-risk
linkage needs (reports/t2wb_protocol.md 3.1 marks it 'not_evaluable' with one
record per patient).

Confound control (reuses create_security_split logic): the predecessor's
training set is reproduced -- ChexpertDataset(sample_n=16000/class,
random_state=1911) over label_like_legacy(legacy_csv) -- and every patient in it
is EXCLUDED, so selected patients were never seen by the diffusion model. We also
exclude every patient already used in security_v1.

Determinism: seed=1911. Images stay in-distribution (Frontal + {no_finding,
pleural_effusion}, matching every other experiment).

  scripts/run_cfg_ddim.sh python -m scripts.build_t2bb_inner_split \
    --csv "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/train.csv" \
    --legacy-csv past/SourceCode/data/CheXpert-v1.0/train.csv \
    --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)" \
    --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)" \
    --image-root "dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)" \
    --patients 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from create_security_split import label_like_legacy, resolve_raw, PATIENT_RE  # noqa: E402
from create_dev_split import stable_split_hash, sha256_file  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

STUDY_RE = re.compile(r"(study\d+)")
LABEL_NAME = {0: "no_finding", 1: "pleural_effusion"}


def reproduce_excluded_patients(legacy_frame: pd.DataFrame) -> tuple[set[str], int]:
    """Exactly create_security_split: 16000/class sampled at random_state=1911."""
    labelled = label_like_legacy(legacy_frame)
    healthy = labelled[labelled["Label"] == 0]
    diseased = labelled[labelled["Label"] == 1]
    sampled_h = healthy.sample(n=min(16000, len(healthy)), random_state=1911)
    sampled_d = diseased.sample(n=min(16000, len(diseased)), random_state=1911)
    senior_sample = pd.concat([sampled_d, sampled_h])
    excluded = set(senior_sample["Path"].str.extract(PATIENT_RE)[0])
    return excluded, len(senior_sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True, help="raw CheXpert train.csv for path resolution")
    parser.add_argument("--legacy-csv", type=Path, required=True, help="exact CSV used by the senior training loader")
    parser.add_argument("--image-root", type=Path, action="append", required=True)
    parser.add_argument("--security-split", type=Path, default=ROOT / "splits/security_v1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "splits/t2bb_inner_v1.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/t2bb_inner_v1_manifest.json")
    parser.add_argument("--patients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1911)
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.output, args.manifest], args.force)

    raw_frame = pd.read_csv(args.csv)
    legacy_frame = pd.read_csv(args.legacy_csv)
    non_path = [c for c in raw_frame.columns if c != "Path"]
    if len(raw_frame) != len(legacy_frame) or not raw_frame[non_path].equals(legacy_frame[non_path]):
        raise RuntimeError("raw and legacy train CSV rows differ outside the expected Path rewrite")

    excluded_patients, legacy_rows = reproduce_excluded_patients(legacy_frame)
    security_patients: set[str] = set()
    if args.security_split.exists():
        with args.security_split.open(newline="", encoding="utf-8") as handle:
            security_patients = {row["patient_id"] for row in csv.DictReader(handle)}

    cand = label_like_legacy(raw_frame).copy()
    cand["patient_id"] = cand["Path"].str.extract(PATIENT_RE)[0]
    cand["study_id"] = cand["Path"].str.extract(STUDY_RE)[0]
    cand = cand.dropna(subset=["patient_id", "study_id"])
    cand = cand[~cand["patient_id"].isin(excluded_patients)]
    cand = cand[~cand["patient_id"].isin(security_patients)]

    # Eligible = patient has >=2 frontal label-filtered images across >=2 studies.
    studies_per_patient = cand.groupby("patient_id")["study_id"].nunique()
    eligible = set(studies_per_patient[studies_per_patient >= 2].index)
    cand = cand[cand["patient_id"].isin(eligible)]

    rng = np.random.default_rng(args.seed)
    patient_order = rng.permutation(sorted(eligible))

    records: list[dict[str, object]] = []
    chosen: list[str] = []
    for patient in patient_order:
        if len(chosen) == args.patients:
            break
        sub = cand[cand["patient_id"] == patient].sort_values(["study_id", "Path"])
        # one image per study, then the two earliest studies (deterministic)
        per_study = sub.groupby("study_id", as_index=False).first().sort_values("study_id")
        if len(per_study) < 2:
            continue
        pair_rows = per_study.iloc[:2]
        resolved = []
        for _, row in pair_rows.iterrows():
            path = resolve_raw(str(row["Path"]), args.image_root)
            if path is None:
                break
            with Image.open(path) as image:
                image.load()
            resolved.append((row, path))
        if len(resolved) != 2:
            continue
        pair_idx = len(chosen)
        chosen.append(str(patient))
        for role_idx, (role, (row, path)) in enumerate(zip(("gallery", "query"), resolved)):
            label = int(row["Label"])
            records.append({
                "sample_id": f"t2bb_inner_v1_{pair_idx:04d}_{role}",
                "patient_id": str(patient),
                "pair_id": f"pair_{pair_idx:04d}",
                "pair_role": role,
                "label": label,
                "label_name": LABEL_NAME[label],
                "view": row["Frontal/Lateral"],
                "ap_pa": row["AP/PA"],
                "study_id": str(row["study_id"]),
                "source_path": row["Path"],
                "local_path": path.as_posix(),
                "file_sha256": sha256_file(path),
            })

    if len(chosen) != args.patients:
        raise RuntimeError(f"selected {len(chosen)} of {args.patients} requested patients (eligible={len(eligible)})")

    split_sha256 = stable_split_hash(records)
    for record in records:
        record["split_sha256"] = split_sha256

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    overlap_train = bool(set(chosen) & excluded_patients)
    overlap_security = bool(set(chosen) & security_patients)
    manifest = {
        "schema_version": 1,
        "split": str(args.output.relative_to(ROOT)),
        "split_sha256": split_sha256,
        "seed": args.seed,
        "purpose": "T2-BB inner-risk anon<->anon linkage enrollment set",
        "protocol": "reports/t2bb_inner_linkage_protocol.md",
        "patients": len(chosen),
        "images_per_patient": 2,
        "distinct_studies_per_patient": 2,
        "total_images": len(records),
        "legacy_sample_rows": legacy_rows,
        "legacy_random_state": 1911,
        "legacy_csv_sha256": sha256_file(args.legacy_csv),
        "raw_csv_sha256": sha256_file(args.csv),
        "excluded_training_patient_count": len(excluded_patients),
        "excluded_security_v1_patient_count": len(security_patients),
        "eligible_patient_count": len(eligible),
        "patient_disjoint_from_reproduced_training_sample": not overlap_train,
        "patient_disjoint_from_security_v1": not overlap_security,
        "label_counts": {str(k): sum(1 for r in records if r["label"] == k) for k in (0, 1)},
        "hash_fields": ["sample_id", "patient_id", "label", "source_path", "file_sha256"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
