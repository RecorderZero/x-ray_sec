"""AF-024: range-preserving PNG storage for M1 images."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.e2_anonymization_runner import legacy_png_handoff
from scripts.m1_storage import (
    HI_KEY,
    LO_KEY,
    RANGE_PRESERVING_PNG,
    decode_range_preserving_png,
    encode_range_preserving_png,
    range_preserving_png_handoff,
)


def sample_image(low: float, high: float, seed: int = 1911) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.uniform(low, high, size=(256, 256)).astype(np.float32)
    image[0, 0], image[-1, -1] = np.float32(low), np.float32(high)
    return image


@pytest.mark.parametrize("low,high", [(-0.404, 0.935), (0.0, 1.0), (-1.2, 2.5), (0.3, 0.31)])
def test_round_trip_error_is_within_half_a_quantization_step(low: float, high: float) -> None:
    image = sample_image(low, high)
    payload, record = encode_range_preserving_png(image)
    restored = decode_range_preserving_png(payload)
    span = float(image.max() - image.min())
    tolerance = span / 510 + 1e-6 * max(1.0, abs(low), abs(high))
    assert float(np.max(np.abs(restored - image))) <= tolerance
    assert record["storage_protocol"] == RANGE_PRESERVING_PNG
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_value_range_is_stored_exactly() -> None:
    image = sample_image(-0.3999999, 0.8671234)
    payload, record = encode_range_preserving_png(image)
    assert np.float32(record["lo"]) == image.min() and np.float32(record["hi"]) == image.max()
    restored = decode_range_preserving_png(payload)
    assert restored.min() == image.min()
    assert abs(float(restored.max()) - float(image.max())) <= 1e-6


def test_constant_image_round_trips() -> None:
    image = np.full((256, 256), -0.25, dtype=np.float32)
    payload, _ = encode_range_preserving_png(image)
    assert np.array_equal(decode_range_preserving_png(payload), image)


def test_missing_range_metadata_fails_closed() -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(buffer, format="PNG")
    with pytest.raises(ValueError, match="value range"):
        decode_range_preserving_png(buffer.getvalue())


def test_non_finite_pixels_are_rejected() -> None:
    image = sample_image(0.0, 1.0)
    image[3, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        encode_range_preserving_png(image)


def test_range_preserving_beats_legacy_min_max_on_out_of_range_images() -> None:
    """Legacy min-max discards [lo, hi]; the new protocol keeps it."""
    tensor = torch.from_numpy(sample_image(-0.404, 0.935))[None, None]
    restored, record = range_preserving_png_handoff(tensor)
    legacy = legacy_png_handoff(tensor)
    new_error = float((restored - tensor).abs().max())
    legacy_error = float((legacy - tensor).abs().max())
    assert new_error <= (0.935 + 0.404) / 510 + 1e-6
    assert legacy_error > 0.3
    assert {LO_KEY, HI_KEY} and record["png_sha256"]
