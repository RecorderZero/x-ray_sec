#!/usr/bin/env python3
"""Q10.4 AEAD reference: raw-byte round-trip and tamper rejection (CPU).

A standard authenticated-encryption lower bound, using the SAME primitive the
deployed S2 v2 container uses for its private metadata -- ChaCha20-Poly1305
keyed by the project HKDF (``derive_subkey(.., PRIVATE_METADATA_LABEL)``).  It
shows what authenticated encryption guarantees on its own, independent of the
diffusion sampler, the structured transform and 8-bit image storage:

  * exact reversibility: ``decrypt(encrypt(bytes)) == bytes`` at 100% byte
    equality for arbitrary payloads (random bytes and raw latent bytes);
  * integrity: any single-byte tamper of ciphertext or tag, a wrong nonce, a
    wrong key, or a modified AAD makes decryption FAIL closed (InvalidTag).

This is the security floor cited when the paper discusses M2 (latent-payload)
bit-exactness: raw bytes can be carried losslessly and authenticated; M1
(image-only) cannot, because the image passes through the sampler + uint8 PNG.
The deployed S2 container's end-to-end HMAC tamper matrix is table_tamper.csv
(D7-C8); this table is the primitive-level reference.

Deterministic: FIXTURE_KEY and nonce_for from scripts.d7_cpu_security, seed 1911
for payloads.  No key material is written -- only equality/rejection outcomes.

  scripts/run_cfg_ddim.sh python -m scripts.q10_aead_reference \
    --out paper_assets/tables/table_aead_reference.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for
from scripts.output_guard import add_force_flag, refuse_overwrite
from src.crypto.kdf import PRIVATE_METADATA_LABEL, derive_subkey

ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    """Path relative to repo root when possible, else the raw path (robust print)."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
SEED = 1911
AEAD_NONCE = bytes(12)  # fixed; the HKDF subkey is unique per image nonce (s2_fixed_norm pattern)
AAD = b"cxr-s2/Q10/aead-reference/v1"


def _payloads() -> list[tuple[str, bytes]]:
    rng = np.random.default_rng(SEED)
    out: list[tuple[str, bytes]] = [
        ("random_32B", rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()),
        ("random_4096B", rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes()),
    ]
    for dim in (4096, 65536):
        for dtype in ("float32", "float64"):
            vec = rng.standard_normal(dim).astype(dtype)
            out.append((f"latent_d{dim}_{dtype}", vec.tobytes()))
    return out


def _key(label_nonce: str) -> bytes:
    return derive_subkey(FIXTURE_KEY, nonce_for(label_nonce), PRIVATE_METADATA_LABEL)


def _roundtrip(name: str, plaintext: bytes) -> dict:
    key = _key(f"aead-ref/{name}")
    aead = ChaCha20Poly1305(key)
    sealed = aead.encrypt(AEAD_NONCE, plaintext, AAD)  # ct || 16-byte tag
    recovered = aead.decrypt(AEAD_NONCE, sealed, AAD)
    byte_exact = recovered == plaintext
    n = len(plaintext)
    equal_bytes = sum(1 for a, b in zip(recovered, plaintext) if a == b) if len(recovered) == n else 0

    # tamper cases: each must raise InvalidTag (fail closed).
    def rejected(fn) -> bool:
        try:
            fn()
            return False
        except InvalidTag:
            return True

    ct_flip = bytearray(sealed); ct_flip[0] ^= 0x01
    tag_flip = bytearray(sealed); tag_flip[-1] ^= 0x01
    wrong_key = ChaCha20Poly1305(_key(f"aead-ref/{name}/WRONG"))
    rej_ct = rejected(lambda: aead.decrypt(AEAD_NONCE, bytes(ct_flip), AAD))
    rej_tag = rejected(lambda: aead.decrypt(AEAD_NONCE, bytes(tag_flip), AAD))
    rej_nonce = rejected(lambda: aead.decrypt(bytes([1]) + AEAD_NONCE[1:], sealed, AAD))
    rej_key = rejected(lambda: wrong_key.decrypt(AEAD_NONCE, sealed, AAD))
    rej_aad = rejected(lambda: aead.decrypt(AEAD_NONCE, sealed, AAD + b"x"))

    return {
        "payload": name, "n_bytes": n, "tag_bytes": 16,
        "roundtrip_byte_exact": byte_exact,
        "roundtrip_byte_equality_rate": (equal_bytes / n) if n else 1.0,
        "tamper_ciphertext_rejected": rej_ct,
        "tamper_tag_rejected": rej_tag,
        "tamper_nonce_rejected": rej_nonce,
        "tamper_key_rejected": rej_key,
        "tamper_aad_rejected": rej_aad,
        "all_tamper_rejected": all([rej_ct, rej_tag, rej_nonce, rej_key, rej_aad]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_aead_reference.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    add_force_flag(parser)
    args = parser.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)

    records = [_roundtrip(name, pt) for name, pt in _payloads()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)

    all_exact = all(r["roundtrip_byte_exact"] for r in records)
    all_reject = all(r["all_tamper_rejected"] for r in records)
    summary = {
        "schema_version": 1, "task_id": "Q10.4", "seed": SEED,
        "primitive": "ChaCha20-Poly1305 (same as S2 v2 private-metadata AEAD), HKDF subkey via PRIVATE_METADATA_LABEL",
        "fixture": "d7_cpu_security.FIXTURE_KEY (public reproducibility fixture; not key material)",
        "aad": AAD.decode("ascii"), "n_payloads": len(records),
        "all_roundtrip_byte_exact": all_exact,
        "all_tamper_rejected": all_reject,
        "tamper_cases": ["ciphertext", "tag", "nonce", "key", "aad"],
        "note": (
            "AEAD security floor: raw bytes round-trip at 100% byte equality and every "
            "tamper fails closed. This is the M2 (latent-payload) bit-exact guarantee. M1 "
            "(image-only) cannot be byte-exact because the image traverses sampler + uint8 "
            "PNG; its reversibility is the fidelity numbers (Q8/RQ4), not this floor. The "
            "deployed S2 container end-to-end tamper matrix is table_tamper.csv (D7-C8)."
        ),
    }
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {_rel(args.out)} ({len(records)} payloads); "
          f"all_roundtrip_byte_exact={all_exact}; all_tamper_rejected={all_reject}")


if __name__ == "__main__":
    main()
