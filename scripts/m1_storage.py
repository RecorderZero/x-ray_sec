#!/usr/bin/env python3
"""M1 image storage protocols for anonymous (and any re-inverted) images.

``legacy_png`` reproduces the predecessor: per-image min-max, uint8 truncation,
the value range is discarded (``scripts.e2_anonymization_runner.legacy_png_handoff``).

``range_preserving_png`` (AF-024, user ruling 2026-10-10): the image is mapped
to 8 bits with its own [lo, hi], quantised with round-half-to-even (``np.rint``)
and written as a real grayscale PNG; lo and hi are kept as exact float32 values
(PNG text chunks hold their IEEE-754 bit patterns, the record holds the same
values) and the reader restores ``x = q / 255 * (hi - lo) + lo``.
lo/hi change the reconstruction, so the S2 container must authenticate them
(PROPOSAL 4.4; D5) before any decode, inversion or generation.
"""

from __future__ import annotations

import hashlib
import io
import struct
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo


RANGE_PRESERVING_PNG = "range_preserving_png/v1"
QUANTIZATION = "round-half-to-even np.rint((x-lo)/(hi-lo)*255), clipped to 0..255"
LO_KEY = "x-range-lo-float32-bits"
HI_KEY = "x-range-hi-float32-bits"


def _f32_bits(value: np.float32) -> str:
    return struct.pack(">f", float(value)).hex()


def _f32_from_bits(text: str) -> np.float32:
    return np.float32(struct.unpack(">f", bytes.fromhex(text))[0])


def encode_range_preserving_png(image: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    """Encode one HxW float32 image; return PNG bytes and the storage record."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("cannot store non-finite pixels")
    lo, hi = np.float32(array.min()), np.float32(array.max())
    span = np.float32(hi - lo)
    if span > 0:
        quantized = np.clip(np.rint((array - lo) / span * np.float32(255)), 0, 255).astype(np.uint8)
    else:
        quantized = np.zeros(array.shape, dtype=np.uint8)
    info = PngInfo()
    info.add_text(LO_KEY, _f32_bits(lo))
    info.add_text(HI_KEY, _f32_bits(hi))
    buffer = io.BytesIO()
    Image.fromarray(quantized, mode="L").save(buffer, format="PNG", pnginfo=info)
    payload = buffer.getvalue()
    record = {
        "storage_protocol": RANGE_PRESERVING_PNG,
        "quantization": QUANTIZATION,
        "lo": float(lo),
        "hi": float(hi),
        "lo_float32_bits": _f32_bits(lo),
        "hi_float32_bits": _f32_bits(hi),
        "png_sha256": hashlib.sha256(payload).hexdigest(),
        "png_bytes": len(payload),
    }
    return payload, record


def decode_range_preserving_png(payload: bytes) -> np.ndarray:
    """Decode PNG bytes written by ``encode_range_preserving_png`` to float32."""
    with Image.open(io.BytesIO(payload)) as handle:
        if handle.mode != "L":
            raise ValueError(f"expected an 8-bit grayscale PNG, got mode {handle.mode}")
        text = dict(handle.text)
        quantized = np.asarray(handle, dtype=np.uint8)
    if LO_KEY not in text or HI_KEY not in text:
        raise ValueError("PNG lacks the stored value range; refusing to guess it")
    lo, hi = _f32_from_bits(text[LO_KEY]), _f32_from_bits(text[HI_KEY])
    return quantized.astype(np.float32) / np.float32(255) * np.float32(hi - lo) + lo


def range_preserving_png_handoff(image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Store a (1,1,H,W) tensor through a real range-preserving PNG and read it back."""
    if image.dim() != 4 or image.shape[:2] != (1, 1):
        raise ValueError(f"expected shape (1,1,H,W), got {tuple(image.shape)}")
    payload, record = encode_range_preserving_png(image[0, 0].detach().float().cpu().numpy())
    restored = decode_range_preserving_png(payload)
    return torch.from_numpy(restored)[None, None].to(image.device), record
