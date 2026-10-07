#!/usr/bin/env python3
"""Create a deterministic, patient-disjoint CheXpert development split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

from PIL import Image


PATIENT_RE = re.compile(r"/(patient\d+)/")
HASH_FIELDS = ("sample_id", "patient_id", "label", "source_path", "file_sha256")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split_hash(records: list[dict[str, object]]) -> str:
    """Hash sample identity/content only; local mount paths are intentionally excluded."""
    canonical = "\n".join(
        ",".join(str(record[key]) for key in HASH_FIELDS) for record in records
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--patients", type=int, default=20)
    args = parser.parse_args()
    if args.patients < 2 or args.patients % 2:
        raise SystemExit("--patients must be an even integer >= 2")

    candidates: dict[int, dict[str, dict[str, str]]] = {0: {}, 1: {}}
    with args.csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Frontal/Lateral"] != "Frontal":
                continue
            match = PATIENT_RE.search("/" + row["Path"])
            if not match:
                continue
            patient_id = match.group(1)
            # Match the legacy loader: pleural effusion overrides No Finding.
            if row["Pleural Effusion"] == "1.0":
                label = 1
            elif row["No Finding"] == "1.0":
                label = 0
            else:
                continue
            candidates[label].setdefault(patient_id, row)

    selected: list[tuple[int, str, dict[str, str], Path]] = []
    per_class = args.patients // 2
    used: set[str] = set()
    for label in (0, 1):
        for patient_id in sorted(candidates[label]):
            if patient_id in used:
                continue
            row = candidates[label][patient_id]
            parts = Path(row["Path"]).parts
            local_path = args.image_root.joinpath(*parts[1:])
            if not local_path.is_file():
                continue
            with Image.open(local_path) as image:
                image.verify()
            selected.append((label, patient_id, row, local_path))
            used.add(patient_id)
            if sum(item[0] == label for item in selected) == per_class:
                break

    if len(selected) != args.patients or len(used) != args.patients:
        raise RuntimeError(
            f"could only select {len(selected)} readable, unique-patient images"
        )
    selected.sort(key=lambda item: (item[0], item[1]))

    records = []
    for index, (label, patient_id, row, local_path) in enumerate(selected):
        records.append(
            {
                "sample_id": f"dev_v1.1_{index:03d}",
                "patient_id": patient_id,
                "label": label,
                "label_name": "pleural_effusion" if label else "no_finding",
                "view": row["Frontal/Lateral"],
                "ap_pa": row["AP/PA"],
                "source_path": row["Path"],
                "local_path": local_path.as_posix(),
                "file_sha256": sha256_file(local_path),
            }
        )
    split_sha256 = stable_split_hash(records)
    for record in records:
        record["split_sha256"] = split_sha256

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows; split_sha256={split_sha256}")


if __name__ == "__main__":
    main()
