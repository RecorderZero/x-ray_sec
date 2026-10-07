#!/usr/bin/env python3
"""Create security_v1 from train images not sampled for senior training.

Reproduces ChexpertDataset(sample_n=16000, random_state=1911), excludes every
patient represented by any sampled training image, then samples one readable
image per remaining patient with a fixed seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scripts.env_guard import enforce_active_prefix
except ModuleNotFoundError:
    from env_guard import enforce_active_prefix

enforce_active_prefix({"numpy": np, "pandas": pd, "pillow": Image})

from create_dev_split import stable_split_hash, sha256_file

PATIENT_RE = re.compile(r"(patient\d+)")


def label_like_legacy(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["Frontal/Lateral"] == "Frontal"].copy()
    frame.loc[frame["No Finding"] == 1, "Label"] = 0
    frame.loc[frame["Pleural Effusion"] == 1, "Label"] = 1
    return frame[frame["Label"].isin([0, 1])].copy()


def resolve_raw(source_path: str, roots: list[Path]) -> Path | None:
    relative = Path(*Path(source_path).parts[2:])
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def frame_hash(frame: pd.DataFrame) -> str:
    material = "\n".join(frame["Path"].astype(str)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="raw CheXpert train.csv used to resolve paths")
    parser.add_argument("--legacy-csv", type=Path, required=True, help="exact CSV used by the senior training loader")
    parser.add_argument("--image-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("splits/security_v1_manifest.json"))
    parser.add_argument("--patients", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1911)
    args = parser.parse_args()
    if args.patients < 2 or args.patients % 2:
        raise SystemExit("--patients must be an even integer >= 2")

    raw_frame = pd.read_csv(args.csv)
    legacy_frame = pd.read_csv(args.legacy_csv)
    non_path = [column for column in raw_frame.columns if column != "Path"]
    if len(raw_frame) != len(legacy_frame) or not raw_frame[non_path].equals(legacy_frame[non_path]):
        raise RuntimeError("raw and legacy train CSV rows differ outside the expected Path rewrite")
    labelled = label_like_legacy(legacy_frame)
    healthy = labelled[labelled["Label"] == 0]
    diseased = labelled[labelled["Label"] == 1]
    sampled_h = healthy.sample(n=min(16000, len(healthy)), random_state=1911)
    sampled_d = diseased.sample(n=min(16000, len(diseased)), random_state=1911)
    senior_sample = pd.concat([sampled_d, sampled_h])
    excluded_patients = set(senior_sample["Path"].str.extract(PATIENT_RE)[0])

    candidates = label_like_legacy(raw_frame)
    candidates["patient_id"] = candidates["Path"].str.extract(PATIENT_RE)[0]
    candidates = candidates[~candidates["patient_id"].isin(excluded_patients)]
    rng = np.random.default_rng(args.seed)
    patient_order = rng.permutation(candidates["patient_id"].dropna().unique())
    rank = {patient: index for index, patient in enumerate(patient_order)}
    candidates["rank"] = candidates["patient_id"].map(rank)
    candidates = candidates.sort_values(["rank", "Path"])

    per_class = args.patients // 2
    selected: list[tuple[int, str, pd.Series, Path]] = []
    used: set[str] = set()
    for label in (0, 1):
        for _, row in candidates[candidates["Label"] == label].iterrows():
            patient = str(row["patient_id"])
            if patient in used:
                continue
            path = resolve_raw(str(row["Path"]), args.image_root)
            if path is None:
                continue
            with Image.open(path) as image:
                image.load()
            selected.append((label, patient, row, path))
            used.add(patient)
            if sum(item[0] == label for item in selected) == per_class:
                break
    if len(selected) != args.patients:
        raise RuntimeError(f"selected {len(selected)} of {args.patients} requested patients")
    selected.sort(key=lambda item: (item[0], item[1]))

    records: list[dict[str, object]] = []
    for index, (label, patient, row, path) in enumerate(selected):
        records.append({
            "sample_id": f"security_v1_{index:04d}",
            "patient_id": patient,
            "label": label,
            "label_name": "pleural_effusion" if label else "no_finding",
            "view": row["Frontal/Lateral"],
            "ap_pa": row["AP/PA"],
            "source_path": row["Path"],
            "local_path": path.as_posix(),
            "file_sha256": sha256_file(path),
        })
    split_sha256 = stable_split_hash(records)
    for record in records:
        record["split_sha256"] = split_sha256

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema_version": 1,
        "split": str(args.output),
        "split_sha256": split_sha256,
        "seed": args.seed,
        "selection": "AF-012 option (a)",
        "legacy_sample_n_per_class": 16000,
        "legacy_random_state": 1911,
        "legacy_sample_rows": len(senior_sample),
        "legacy_csv_sha256": sha256_file(args.legacy_csv),
        "raw_csv_sha256": sha256_file(args.csv),
        "raw_legacy_non_path_rows_identical": True,
        "legacy_sample_path_sha256": frame_hash(senior_sample),
        "excluded_patient_count": len(excluded_patients),
        "formal_patient_count": len(records),
        "label_counts": {"0": per_class, "1": per_class},
        "patient_disjoint_from_reproduced_training_sample": not bool(used & excluded_patients),
        "hash_fields": ["sample_id", "patient_id", "label", "source_path", "file_sha256"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
