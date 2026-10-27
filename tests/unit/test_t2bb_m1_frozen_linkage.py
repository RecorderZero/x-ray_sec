from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image

from scripts.t2bb_m1_frozen_linkage import evaluate_similarity, load_export


def test_evaluate_similarity_perfect_retrieval():
    similarity = np.eye(8, dtype=np.float64)
    result = evaluate_similarity(similarity, repeats=100, seed=7)
    assert result["top1"] == 1.0
    assert result["top5"] == 1.0
    assert result["auc"] == 1.0
    assert result["random_top1"] == 0.125


def test_load_export_rejects_modified_png(tmp_path):
    export = tmp_path / "export"
    method_dir = export / "P"
    method_dir.mkdir(parents=True)
    image_path = method_dir / "g.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "t2bb_m1_public/v2", "images": 1, "methods": ["P"],
        "artifacts": [{"method": "P", "relative_path": "P/g.png", "png_sha256": digest,
                       "public_id": "opaque"}],
    }
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    image_path.write_bytes(image_path.read_bytes() + b"tamper")
    try:
        load_export(export)
    except RuntimeError as error:
        assert "SHA mismatch" in str(error)
    else:
        raise AssertionError("tampered PNG was accepted")


def test_load_export_rejects_truth_leak(tmp_path):
    export = tmp_path / "export"
    method_dir = export / "P"
    method_dir.mkdir(parents=True)
    image_path = method_dir / "opaque.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(image_path)
    manifest = {
        "schema": "t2bb_m1_public/v2", "images": 1, "methods": ["P"],
        "artifacts": [{"method": "P", "relative_path": "P/opaque.png",
                       "png_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                       "public_id": "opaque", "pair_role": "gallery"}],
    }
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        load_export(export)
    except RuntimeError as error:
        assert "leaks evaluation truth" in str(error)
    else:
        raise AssertionError("truth-bearing public manifest was accepted")


def test_load_export_rejects_path_escape(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    outside = tmp_path / "outside.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(outside)
    manifest = {
        "schema": "t2bb_m1_public/v2", "images": 1, "methods": ["P"],
        "artifacts": [{"method": "P", "relative_path": "../outside.png",
                       "png_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                       "public_id": "opaque"}],
    }
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        load_export(export)
    except RuntimeError as error:
        assert "escapes export directory" in str(error)
    else:
        raise AssertionError("path traversal was accepted")
