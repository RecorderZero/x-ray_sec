#!/usr/bin/env python3
"""CPU-only CheXlocalize preflight; never interpret absent annotations as negatives.

--image-root is the directory containing patient*/study*/view*.jpg (valid/ or
test/). No image preprocessing is performed. Mask positivity is unknown unless
--decode-masks validates masks using the optional pycocotools dependency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

PATHOLOGY = "Pleural Effusion"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def image_identity(source: str, split: str):
    parts = PurePosixPath(source).parts
    if ".." in parts or PurePosixPath(source).is_absolute():
        raise ValueError(f"Unsafe source image path: {source}")
    markers = ("valid", "val") if split == "val" else ("test",)
    positions = [i for i, part in enumerate(parts) if part in markers]
    if len(positions) != 1:
        raise ValueError(f"Expected {split} path, got: {source}")
    suffix = parts[positions[0] + 1:]
    if (len(suffix) != 3 or not suffix[0].startswith("patient")
            or not suffix[1].startswith("study") or not suffix[2].endswith("_frontal.jpg")):
        raise ValueError(f"Unexpected frontal image identity: {source}")
    return Path(*suffix), "_".join((*suffix[:2], Path(suffix[2]).stem)), suffix[0]


def validate_rle(rle, height, width, decoder=None):
    if not isinstance(rle, dict) or rle.get("size") != [height, width]:
        raise ValueError("Annotation RLE dimensions do not match original image")
    counts = rle.get("counts")
    if isinstance(counts, str):
        if not counts or not counts.isascii():
            raise ValueError("Compressed RLE counts must be a nonempty ASCII string")
    elif isinstance(counts, list):
        if (not counts or any(type(x) is not int or x < 0 for x in counts)
                or sum(counts) != height * width):
            raise ValueError("Invalid uncompressed RLE counts")
    else:
        raise ValueError("RLE counts must be a compressed string or integer list")
    if decoder is None:
        return None
    encoded = dict(rle)
    if isinstance(counts, list):
        encoded = decoder.frPyObjects(encoded, height, width)
    else:
        encoded["counts"] = counts.encode("ascii")
    mask = decoder.decode(encoded)
    if mask.shape != (height, width) or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Decoded RLE is not a binary original-resolution mask")
    return int(mask.sum())


def build_manifest(annotations: Path, labels: Path, image_root: Path,
                   split: str = "val", decode_masks: bool = False,
                   allow_unannotated: bool = False):
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    decoder = None
    if decode_masks:
        try:
            from pycocotools import mask as decoder
        except ImportError as exc:
            raise RuntimeError("--decode-masks requires pycocotools; install it in the "
                               "evaluation environment, or omit this flag for structural "
                               "preflight only (mask positives remain unknown).") from exc
    with annotations.open(encoding="utf-8") as stream:
        gt = json.load(stream, object_pairs_hook=unique_object)
    if not isinstance(gt, dict):
        raise ValueError("Expected annotation object keyed by image ID")
    root = image_root.resolve()
    records, seen_ids, seen_paths = [], set(), set()
    lateral_count = 0
    unannotated = []
    with labels.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"Path", PATHOLOGY}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Label CSV requires columns: {sorted(required)}")
        for row in reader:
            view = row.get("Frontal/Lateral")
            if view is None:
                name = PurePosixPath(row["Path"]).name
                view = ("Frontal" if name.endswith("_frontal.jpg") else
                        "Lateral" if name.endswith("_lateral.jpg") else None)
            if view == "Lateral":
                lateral_count += 1
                continue
            if view != "Frontal":
                raise ValueError("Unknown projection label; refusing silent exclusion")
            suffix, sample_id, patient_id = image_identity(row["Path"], split)
            path = (root / suffix).resolve()
            if root not in path.parents:
                raise ValueError("Resolved image escapes image root")
            if sample_id in seen_ids or path in seen_paths:
                raise ValueError(f"Duplicate image: {sample_id}")
            seen_ids.add(sample_id)
            seen_paths.add(path)
            if sample_id not in gt or PATHOLOGY not in gt[sample_id]:
                if allow_unannotated:
                    unannotated.append(sample_id)
                    continue
                raise ValueError(f"Missing annotation for {sample_id}/{PATHOLOGY}; "
                                 "missing is not a negative label")
            try:
                label = float(row[PATHOLOGY])
            except ValueError as exc:
                raise ValueError(f"Missing/nonbinary label: {sample_id}") from exc
            if label not in (0.0, 1.0):
                raise ValueError(f"Expected adjudicated binary label: {sample_id}")
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            area = validate_rle(gt[sample_id][PATHOLOGY], height, width, decoder)
            records.append({"sample_id": sample_id, "patient_id": patient_id,
                            "split": split, "source_path": row["Path"],
                            "image_path": str(path), "image_sha256": sha256(path),
                            "gt_key": sample_id, "pathology": PATHOLOGY,
                            "label": int(label), "view": "Frontal",
                            "original_h": height, "original_w": width,
                            "mask_area_pixels": area,
                            "mask_positive": None if area is None else area > 0})
    if not records:
        raise ValueError("No frontal images available")
    records.sort(key=lambda row: row["sample_id"])
    patients = Counter(row["patient_id"] for row in records)
    return {"schema_version": "chexlocalize_manifest_v1", "split": split,
            "pathology": PATHOLOGY, "annotations_path": str(annotations.resolve()),
            "annotations_sha256": sha256(annotations), "labels_path": str(labels.resolve()),
            "labels_sha256": sha256(labels), "image_root": str(root),
            "preprocessing": "none; original geometry validated; downstream must record preprocessing",
            "mask_validation": "decoded_binary" if decode_masks else "structural_only",
            "unannotated_policy": "exclude_explicitly" if allow_unannotated else "error",
            "unannotated_image_ids": sorted(unannotated),
            "counts": {"images": len(records), "unique_patients": len(patients),
                       "patients_with_multiple_images": sum(n > 1 for n in patients.values()),
                       "lateral_rows_excluded": lateral_count,
                       "unannotated_frontal_excluded": len(unannotated),
                       "label_positive": sum(row["label"] for row in records),
                       "label_negative": sum(1 - row["label"] for row in records),
                       "mask_positive": (sum(row["mask_positive"] for row in records)
                                         if decode_masks else None)},
            "records": records}


def write_manifest(manifest, output: Path):
    # Exclusive creation: never replace an existing manifest, even if identical.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--decode-masks", action="store_true")
    parser.add_argument("--allow-unannotated", action="store_true",
                        help="Explicitly exclude and list missing annotations; never label them negative")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.annotations, args.labels, args.image_root,
                            args.split, args.decode_masks, args.allow_unannotated)
    write_manifest(result, args.output)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
