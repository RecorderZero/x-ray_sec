from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.e1_ddim_runner import preprocess


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "past/SourceCode"


def load_legacy_oracles():
    source = (SOURCE / "scripts/chexpert_preproc.py").read_text(encoding="utf-8")
    source = source[: source.index("class ChexpertFilterPipeline")]
    sys.modules.setdefault(
        "tqdm", types.SimpleNamespace(tqdm=types.SimpleNamespace(pandas=lambda: None))
    )
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - frozen legacy source is the oracle
    preprocess_image = namespace["ChexpertResNormPipeline"]._preprocess_image

    train_source = (SOURCE / "guided_diffusion/train_util.py").read_text(encoding="utf-8")
    start = train_source.index("def visualize(")
    end = train_source.index("\ndef ", start + 1)
    visualize_namespace = {"np": np}
    exec(train_source[start:end], visualize_namespace)  # noqa: S102
    return preprocess_image, visualize_namespace["visualize"]


def test_preprocess_matches_legacy_source_oracle(tmp_path: Path) -> None:
    y, x = np.mgrid[0:96, 0:80]
    source = ((3 * x + 5 * y + (x // 7) * 19) % 256).astype(np.uint8)
    path = tmp_path / "source.png"
    assert cv2.imwrite(str(path), source)

    legacy_preprocess, legacy_visualize = load_legacy_oracles()
    legacy_image = legacy_preprocess(None, str(path), 256)
    legacy_jpg = tmp_path / "legacy.jpg"
    assert cv2.imwrite(
        str(legacy_jpg), legacy_image, [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    loaded = np.asarray(Image.open(legacy_jpg).convert("L"))[:, :, None]
    expected = np.transpose(
        legacy_visualize(loaded).astype(np.float32), [2, 0, 1]
    )

    observed_tensor, observed_array = preprocess(path)
    assert observed_tensor.shape == (1, 256, 256)
    assert observed_tensor.dtype == torch.float32
    assert np.array_equal(observed_array, expected[0])
    assert torch.equal(observed_tensor, torch.from_numpy(expected))
    assert np.isfinite(observed_array).all()
