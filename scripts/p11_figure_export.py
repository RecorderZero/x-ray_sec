#!/usr/bin/env python3
"""P11.4 (mechanical part): lossless PDF export of the frozen figure grids.

The paper's figure grids (`fig_P8_visual_grid.png`, `grid_gs2.png`) are PIL-rendered
raster panels of real CXR pixels + drawn labels -- they are photographic, not vector,
so a true SVG would only base64-wrap the same raster. What a publication needs is a
PDF that embeds the exact pixels at a defined physical size (300 dpi) so LaTeX can
place it crisply. Pillow 8.4's own PDF writer re-encodes RGB/L as DCTDecode (JPEG,
LOSSY), which would alter a frozen artifact; so we build a minimal PDF by hand with a
single FlateDecode image XObject -- bit-exact to the source PNG's pixels, deterministic
(no timestamps), re-runnable. Only the caption text + vector schematics (threat model /
pipeline) remain for the December writing phase.

  scripts/run_cfg_ddim.sh python -m scripts.p11_figure_export
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DPI = 300.0
# source PNG (relative to paper_assets) -> output PDF alongside it
FIGURES = [
    "q8_formal/fig_P8_visual_grid.png",
    "reinvert_ablation/grid_gs2.png",
]


def _png_to_pixels(path: Path) -> tuple[bytes, int, int, str]:
    """Return (raw top-to-bottom pixel bytes, width, height, pdf_colorspace)."""
    im = Image.open(path)
    if im.mode in ("L", "I;16", "1"):
        im = im.convert("L")
        cs = "/DeviceGray"
    else:
        im = im.convert("RGB")
        cs = "/DeviceRGB"
    w, h = im.size
    return im.tobytes(), w, h, cs


def _build_pdf(raw: bytes, w: int, h: int, cs: str) -> bytes:
    """Minimal single-page PDF embedding `raw` 8-bit pixels via FlateDecode."""
    w_pt = w / DPI * 72.0
    h_pt = h / DPI * 72.0
    img_stream = zlib.compress(raw, 9)  # level 9 is deterministic
    content = f"q {w_pt:.4f} 0 0 {h_pt:.4f} 0 0 cm /Im0 Do Q".encode("ascii")

    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt:.4f} {h_pt:.4f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode("ascii"),
        (b"<< /Type /XObject /Subtype /Image /Width " + str(w).encode() +
         b" /Height " + str(h).encode() + b" /ColorSpace " + cs.encode() +
         b" /BitsPerComponent 8 /Filter /FlateDecode /Length " +
         str(len(img_stream)).encode() + b" >>\nstream\n" + img_stream + b"\nendstream"),
        (b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"),
    ]

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_pos = len(out)
    n = len(objs) + 1
    out += f"xref\n0 {n}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n").encode("ascii")
    return bytes(out)


def _verify(pdf: bytes, raw: bytes) -> None:
    """Decompress the embedded image stream and assert it equals the source pixels."""
    marker = b"/Filter /FlateDecode /Length "
    i = pdf.index(marker)
    s = pdf.index(b"stream\n", i) + len(b"stream\n")
    e = pdf.index(b"\nendstream", s)
    assert zlib.decompress(pdf[s:e]) == raw, "PDF pixel round-trip is NOT bit-exact"


def main() -> None:
    for rel in FIGURES:
        src = ROOT / "paper_assets" / rel
        dst = src.with_suffix(".pdf")
        raw, w, h, cs = _png_to_pixels(src)
        pdf = _build_pdf(raw, w, h, cs)
        _verify(pdf, raw)
        dst.write_bytes(pdf)
        print(f"{rel}: {w}x{h} {cs[1:]} -> {dst.name}  "
              f"{w/DPI:.2f}x{h/DPI:.2f} in @ {DPI:.0f} dpi  "
              f"({len(pdf)/1024:.0f} KiB, bit-exact)")


if __name__ == "__main__":
    main()
