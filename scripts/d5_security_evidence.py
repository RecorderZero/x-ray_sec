#!/usr/bin/env python3
"""Build small, secret-free D5.4/D5.6 evidence artifacts (CPU only)."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cryptography
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from scripts.m1_storage import HI_KEY, LO_KEY, QUANTIZATION, encode_range_preserving_png
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import sha256_file
from src.crypto.container import AuthenticatedContainer, make_storage_record, validate_m1_png
from src.crypto.errors import AuthenticationError, NonceReuseError
from src.crypto.kdf import generate_master_key
from src.crypto.mac import open_authenticated, seal_container
from src.crypto.nonce import NonceRegistry


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def fixture():
    image = np.linspace(-0.4, 0.9, 4096, dtype=np.float32).reshape(64, 64)
    payload, png_record = encode_range_preserving_png(image)
    unsigned = AuthenticatedContainer(
        nonce=bytes(range(16)),
        parameters={"scheme": "S2", "rounds": 2, "permutation": True},
        aad={
            "study_instance_uid": "1.2.3", "sop_instance_uid": "1.2.3.4",
            "shape": [1, 256, 256], "dtype": "float32", "model_sha256": "ab" * 32,
            "sampler_version": "legacy-ddim-anonymization/v1",
        },
        storage=make_storage_record(
            mode="M1", storage_protocol=png_record["storage_protocol"], quantization=QUANTIZATION,
            lo_float32_bits=png_record["lo_float32_bits"],
            hi_float32_bits=png_record["hi_float32_bits"], payload=payload,
        ),
        ciphertext=payload,
    )
    key = bytes(range(32))  # public test vector only; never deployment material
    return key, unsigned, seal_container(key, unsigned, preseal_validator=validate_m1_png)


def mutate(serialized: bytes, callback) -> bytes:
    document = json.loads(serialized)
    callback(document)
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def change_png_lo(document) -> None:
    payload = base64.b64decode(document["ciphertext"])
    with Image.open(io.BytesIO(payload)) as handle:
        pixels = np.asarray(handle, dtype=np.uint8)
        text = dict(handle.text)
    info = PngInfo()
    info.add_text(LO_KEY, "00000000")
    info.add_text(HI_KEY, text[HI_KEY])
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG", pnginfo=info)
    changed = buffer.getvalue()
    document["ciphertext"] = base64.b64encode(changed).decode()
    document["storage"]["payload_sha256"] = hashlib.sha256(changed).hexdigest()


def change_payload_byte(document) -> None:
    payload = bytearray(base64.b64decode(document["ciphertext"]))
    payload[-12] ^= 1
    changed = bytes(payload)
    document["ciphertext"] = base64.b64encode(changed).decode()
    document["storage"]["payload_sha256"] = hashlib.sha256(changed).hexdigest()


def change_png_and_record_lo(document) -> None:
    change_png_lo(document)
    document["storage"]["lo_float32_bits"] = "00000000"


def tamper_rows() -> list[dict]:
    key, _, sealed = fixture()
    cases = {
        "untampered": (sealed, key, True),
        "wrong_key": (sealed, bytes(reversed(range(32))), False),
        "tag": (mutate(sealed, lambda d: d.update(tag="00" * 32)), key, False),
        "nonce": (mutate(sealed, lambda d: d.update(nonce="ff" * 16)), key, False),
        "version": (mutate(sealed, lambda d: d.update(version="cxr-auth-container/v2")), key, False),
        "parameters": (mutate(sealed, lambda d: d["parameters"].update(rounds=3)), key, False),
        "aad": (mutate(sealed, lambda d: d["aad"].update(sop_instance_uid="9.9.9")), key, False),
        "storage_protocol": (mutate(sealed, lambda d: d["storage"].update(storage_protocol="other")), key, False),
        "quantization": (mutate(sealed, lambda d: d["storage"].update(quantization="floor")), key, False),
        "record_lo_only": (mutate(sealed, lambda d: d["storage"].update(lo_float32_bits="00000000")), key, False),
        "record_hi_only": (mutate(sealed, lambda d: d["storage"].update(hi_float32_bits="3f800000")), key, False),
        "png_lo_only": (mutate(sealed, change_png_lo), key, False),
        "png_and_record_lo": (mutate(sealed, change_png_and_record_lo), key, False),
        "ciphertext_byte": (mutate(sealed, change_payload_byte), key, False),
    }
    output = []
    for name, (serialized, candidate_key, expected_accept) in cases.items():
        calls = []
        error = ""
        try:
            open_authenticated(
                serialized, candidate_key,
                lambda container: calls.append("open") or validate_m1_png(container),
            )
            accepted = True
        except AuthenticationError as caught:
            accepted, error = False, str(caught)
        output.append({
            "case": name, "expected_accept": expected_accept, "accepted": accepted,
            "opener_calls": len(calls), "generic_error": error,
            "passed": accepted == expected_accept and len(calls) == (1 if expected_accept else 0),
        })
    return output


def nonce_evidence(path: Path, count: int) -> dict:
    key = generate_master_key()
    nonces = [secrets.token_bytes(16) for _ in range(count)]
    unique = len(set(nonces))
    with NonceRegistry(path) as registry:
        registry.register_many(key, nonces)
        reuse_rejected = False
        try:
            registry.register(key, nonces[0])
        except NonceReuseError:
            reuse_rejected = True
        other_scope_accepted = True
        try:
            registry.register(generate_master_key(), nonces[0])
        except Exception:
            other_scope_accepted = False
        rows = registry.count()
    return {
        "requested": count, "unique": unique, "collisions": count - unique,
        "explicit_reuse_rejected": reuse_rejected,
        "same_nonce_other_master_scope_accepted": other_scope_accepted,
        "registry_rows": rows,
        "nonce_bytes": 16,
        "raw_nonce_or_key_recorded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce-count", type=int, default=100_000)
    parser.add_argument("--nonce-output", type=Path, default=Path("results/D5.4_nonce_registry.json"))
    parser.add_argument("--tamper-output", type=Path, default=Path("results/D5.6_tamper_matrix.json"))
    parser.add_argument("--tamper-table", type=Path, default=Path("paper_assets/tables/table_D5_tamper_matrix.csv"))
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.nonce_output, args.tamper_output, args.tamper_table], args.force)
    if args.nonce_count < 100_000:
        raise SystemExit("D5.4 evidence requires at least 100,000 nonces")
    temporary_registry = Path("/tmp") / f"d5_nonce_registry_{secrets.token_hex(8)}.sqlite3"
    nonce = nonce_evidence(temporary_registry, args.nonce_count)
    for suffix in ("", "-wal", "-shm"):
        Path(str(temporary_registry) + suffix).unlink(missing_ok=True)
    rows = tamper_rows()
    if nonce["collisions"] != 0 or not nonce["explicit_reuse_rejected"] or not all(row["passed"] for row in rows):
        raise RuntimeError("D5 security evidence gate failed")
    common = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain=v1", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version.split()[0], "cryptography": cryptography.__version__,
    }
    args.nonce_output.parent.mkdir(parents=True, exist_ok=True)
    args.nonce_output.write_text(json.dumps({**common, "task_id": "D5.4", "result": nonce}, indent=2) + "\n")
    args.tamper_output.write_text(json.dumps({**common, "task_id": "D5.6", "rows": rows}, indent=2) + "\n")
    args.tamper_table.parent.mkdir(parents=True, exist_ok=True)
    with args.tamper_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"nonce": nonce, "tamper_cases": len(rows), "all_passed": True}, indent=2))


if __name__ == "__main__":
    main()
