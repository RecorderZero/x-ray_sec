import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.e1_ddim_runner import preprocess


def test_preprocess_matches_legacy_chexpert_chain(tmp_path: Path) -> None:
    y, x = np.mgrid[0:96, 0:80]
    source = ((3 * x + 5 * y + (x // 7) * 19) % 256).astype(np.uint8)
    path = tmp_path / "source.png"
    assert cv2.imwrite(str(path), source)

    observed_tensor, observed_array = preprocess(path)

    equalized = cv2.equalizeHist(source)
    resized = cv2.resize(equalized, (256, 256), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    assert ok
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        expected = np.asarray(image.convert("L"), dtype=np.float32)
    expected = (expected - expected.min()) / (expected.max() - expected.min())

    assert observed_tensor.shape == (1, 256, 256)
    assert observed_tensor.dtype == torch.float32
    assert np.array_equal(observed_array, expected)
    assert torch.equal(observed_tensor, torch.from_numpy(expected[None, ...]))
    assert np.isfinite(observed_array).all()
    assert observed_array.min() == 0.0
    assert observed_array.max() == 1.0
