#!/usr/bin/env python3
"""Freeze v1 PRNG / SOT outputs as regression vectors before the S1 refactor.

Plan: reports/cross_institution_sharing.md §8.9 S1 / §8.8 V1. The vectors are
generated from the *pre-refactor* src/crypto/prng.py and must stay bit-identical
afterwards (tests/security/test_prng_v1_frozen.py).

Key material: the public test key bytes(range(32)) already used by
tests/security/test_prng.py -- not a real key. CPU only.

  scripts/run_cfg_ddim.sh python -m scripts.freeze_prng_v1_vectors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from src.crypto.prng import permutation, sign_vector  # noqa: E402
from src.crypto.sot import transform  # noqa: E402

TEST_KEY = bytes(range(32))
NONCES = {
    "ascending": bytes(range(16)),
    "descending": bytes(reversed(range(16))),
    "sha256_label": hashlib.sha256(b"soteria/prng-v1-frozen/nonce").digest()[:16],
}
ROUNDS = (0, 1, 3)
SIZES = (256, 65_536)
SEED = 1911


def _sha(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(array.astype(dtype)).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "tests/security/vectors/prng_v1_frozen.json")
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.output], args.force)

    signs, perms = [], []
    for nonce_name, nonce in NONCES.items():
        for round_index in ROUNDS:
            for size in SIZES:
                for purpose in ("sign-pre", "sign-post"):
                    signs.append({"nonce": nonce_name, "round": round_index, "purpose": purpose, "size": size,
                                  "sha256_int8": _sha(sign_vector(TEST_KEY, nonce, round_index, purpose, size), "<i1")})
                perms.append({"nonce": nonce_name, "round": round_index, "size": size,
                              "sha256_int64": _sha(permutation(TEST_KEY, nonce, round_index, size), "<i8")})

    x = torch.from_numpy(np.random.default_rng(SEED).standard_normal(65_536).astype(np.float32)).reshape(1, 1, 256, 256)
    sot = []
    for rounds, use_perm in ((1, True), (1, False), (2, True)):
        out = transform(x, TEST_KEY, NONCES["ascending"], rounds=rounds, use_permutation=use_perm)
        sot.append({"rounds": rounds, "permutation": use_perm, "input": f"default_rng({SEED}).standard_normal(65536) float32",
                    "sha256_float32": _sha(out.numpy().reshape(-1), "<f4")})

    prng_src = ROOT / "src/crypto/prng.py"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/crypto/prng.py", "src/crypto/sot.py", "src/crypto/kdf.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    payload = {
        "schema_version": 1,
        "purpose": "S1 v1 regression vectors (frozen BEFORE the prng.py refactor)",
        "test_key": "bytes(range(32)) -- public test key, not real key material",
        "nonces": {k: v.hex() for k, v in NONCES.items()},
        "generated_from": {
            "git_head": head,
            "crypto_files_dirty": bool(dirty),
            "prng_py_sha256": hashlib.sha256(prng_src.read_bytes()).hexdigest(),
            "sot_py_sha256": hashlib.sha256((ROOT / "src/crypto/sot.py").read_bytes()).hexdigest(),
            "kdf_py_sha256": hashlib.sha256((ROOT / "src/crypto/kdf.py").read_bytes()).hexdigest(),
            "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
        },
        "sign_vectors": signs,
        "permutations": perms,
        "sot_transform": sot,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(signs)} sign, {len(perms)} perm, {len(sot)} sot vectors; head={head} dirty={bool(dirty)}")


if __name__ == "__main__":
    main()
