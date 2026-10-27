"""Synthetic CPU tests; no public dataset or model dependencies."""
import csv
import json

import pytest
from PIL import Image

from scripts.chexlocalize_manifest import build_manifest, image_identity, validate_rle, write_manifest


@pytest.fixture
def fixture_data(tmp_path):
    root = tmp_path / "valid"
    image = root / "patient1/study1/view1_frontal.jpg"
    image.parent.mkdir(parents=True)
    Image.new("L", (4, 3), 80).save(image)
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Path", "Frontal/Lateral", "Pleural Effusion"])
        writer.writerow(["CheXpert-v1.0/valid/patient1/study1/view1_frontal.jpg", "Frontal", 1])
    annotations = tmp_path / "gt.json"
    annotations.write_text(json.dumps({"patient1_study1_view1_frontal": {
        "Pleural Effusion": {"size": [3, 4], "counts": [12]}}}))
    return annotations, labels, root


def test_structural_preflight_keeps_mask_positive_unknown(fixture_data):
    result = build_manifest(*fixture_data)
    assert result["counts"]["images"] == result["counts"]["unique_patients"] == 1
    assert result["counts"]["label_positive"] == 1
    assert result["counts"]["mask_positive"] is None
    assert result["records"][0]["mask_positive"] is None
    assert len(result["records"][0]["image_sha256"]) == 64


def test_missing_annotation_is_not_negative(fixture_data):
    fixture_data[0].write_text("{}")
    with pytest.raises(ValueError, match="Missing annotation"):
        build_manifest(*fixture_data)


def test_explicit_exclusion_is_recorded(fixture_data):
    with fixture_data[1].open("a") as stream:
        stream.write("CheXpert-v1.0/valid/patient2/study1/view1_frontal.jpg,Frontal,0\n")
    result = build_manifest(*fixture_data, allow_unannotated=True)
    assert result["counts"]["unannotated_frontal_excluded"] == 1
    assert result["counts"]["label_negative"] == 0
    assert result["unannotated_image_ids"] == ["patient2_study1_view1_frontal"]


def test_test_csv_view_inferred(fixture_data):
    fixture_data[1].write_text("Path,Pleural Effusion\n"
                              "test/patient1/study1/view1_frontal.jpg,1\n")
    assert build_manifest(*fixture_data, split="test")["counts"]["images"] == 1


def test_dimension_mismatch(fixture_data):
    fixture_data[0].write_text(json.dumps({"patient1_study1_view1_frontal": {
        "Pleural Effusion": {"size": [4, 3], "counts": [12]}}}))
    with pytest.raises(ValueError, match="dimensions"):
        build_manifest(*fixture_data)


def test_duplicate_image_rejected(fixture_data):
    with fixture_data[1].open("a") as stream:
        stream.write("CheXpert-v1.0/valid/patient1/study1/view1_frontal.jpg,Frontal,1\n")
    with pytest.raises(ValueError, match="Duplicate image"):
        build_manifest(*fixture_data)


def test_duplicate_json_key_rejected(fixture_data):
    fixture_data[0].write_text('{"same": {}, "same": {}}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        build_manifest(*fixture_data)


@pytest.mark.parametrize("counts", [[], [-1, 13], [11], [True, 11], "", None])
def test_invalid_counts(counts):
    with pytest.raises(ValueError):
        validate_rle({"size": [3, 4], "counts": counts}, 3, 4)


@pytest.mark.parametrize("source", ["../valid/patient1/study1/view1_frontal.jpg",
                                    "CheXpert-v1.0/test/patient1/study1/view1_frontal.jpg"])
def test_invalid_source_split(source):
    with pytest.raises(ValueError):
        image_identity(source, "val")


def test_no_overwrite(tmp_path):
    output = tmp_path / "manifest.json"
    write_manifest({"first": True}, output)
    with pytest.raises(FileExistsError):
        write_manifest({"first": False}, output)
    assert json.loads(output.read_text()) == {"first": True}


def test_decoded_masks_distinct_from_labels(fixture_data):
    pytest.importorskip("pycocotools.mask")
    result = build_manifest(*fixture_data, decode_masks=True)
    assert result["counts"]["label_positive"] == 1
    assert result["counts"]["mask_positive"] == 0
