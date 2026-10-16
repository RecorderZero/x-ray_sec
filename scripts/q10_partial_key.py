#!/usr/bin/env python3
"""Q10.5 (optional) key-sensitivity / avalanche diagnostic (CPU, no checkpoint).

Flip ONE bit of the 32-byte master key and measure how much the key-derived
material and the resulting ciphertext change:

  * derived-key Hamming distance -- the HKDF subkey and the downstream sign
    vectors should flip ~50% of their bits (sign avalanche); the permutation
    should change at almost every position.
  * ciphertext correlation -- the structured transform of a fixed latent under
    the original vs the one-bit-flipped key should be near-uncorrelated
    (|cosine| ~ 0).

This is a DIAGNOSTIC of the KDF's diffusion, not a security proof, and key-space
size is never used as a security argument (skill sec. 8).  It characterises the
S2/S2a construction's key schedule only.

Deterministic: FIXTURE_KEY / nonce_for from scripts.d7_cpu_security (public
reproducibility fixture; not key material), seed 1911 for latents.  No key bytes
are written -- only Hamming/correlation aggregates.

  scripts/run_cfg_ddim.sh python -m scripts.q10_partial_key \
    --out paper_assets/tables/table_partial_key.csv \
    --figure image/Q10.5_partial_key_avalanche.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for
from scripts.output_guard import add_force_flag, refuse_overwrite
from src.crypto.kdf import derive_subkey, round_label
from src.crypto.prng import permutation, sign_vector
from src.crypto.sot import transform

ROOT = Path(__file__).resolve().parents[1]
SEED = 1911
DIM = 65536            # the 4x128x128 model latent
ROUNDS = 1
USE_PERMUTATION = True  # exercise sign-pre, permutation and sign-post together
N_LATENT = 8
NONCE = None            # set in main from nonce_for


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _flip_bit(key: bytes, bit: int) -> bytes:
    b = bytearray(key)
    b[bit // 8] ^= (1 << (bit % 8))
    return bytes(b)


def _subkey_hamming_fraction(k0: bytes, k1: bytes, nonce: bytes) -> float:
    s0 = derive_subkey(k0, nonce, round_label(0, "sign-pre"))
    s1 = derive_subkey(k1, nonce, round_label(0, "sign-pre"))
    xor = np.frombuffer(bytes(a ^ b for a, b in zip(s0, s1)), dtype=np.uint8)
    return float(np.unpackbits(xor).mean())


def _sign_flip_fraction(k0, k1, nonce, purpose) -> float:
    a = sign_vector(k0, nonce, 0, purpose, DIM)
    b = sign_vector(k1, nonce, 0, purpose, DIM)
    return float((a != b).mean())


def _perm_change_fraction(k0, k1, nonce) -> float:
    a = permutation(k0, nonce, 0, DIM)
    b = permutation(k1, nonce, 0, DIM)
    return float((a != b).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_partial_key.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=ROOT / "image/Q10.5_partial_key_avalanche.png")
    add_force_flag(parser)
    args = parser.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out, args.figure], args.force)

    nonce = nonce_for("q10-partial-key")
    gen = torch.Generator().manual_seed(SEED)
    latents = torch.randn(N_LATENT, DIM, generator=gen, dtype=torch.float32)
    # ciphertext under the original key (cached once per latent)
    base_ct = [transform(latents[i], FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION)
               for i in range(N_LATENT)]
    base_ct = [c.to(torch.float64) for c in base_ct]

    n_bits = 8 * len(FIXTURE_KEY)  # 256
    records = []
    for bit in range(n_bits):
        k1 = _flip_bit(FIXTURE_KEY, bit)
        cos_abs = []
        for i in range(N_LATENT):
            ct = transform(latents[i], k1, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION).to(torch.float64)
            num = float(torch.dot(ct, base_ct[i]))
            den = float(ct.norm() * base_ct[i].norm())
            cos_abs.append(abs(num / den) if den else 0.0)
        records.append({
            "flipped_bit": bit, "flipped_byte": bit // 8, "bit_in_byte": bit % 8,
            "subkey_bit_hamming_fraction": _subkey_hamming_fraction(FIXTURE_KEY, k1, nonce),
            "sign_pre_flip_fraction": _sign_flip_fraction(FIXTURE_KEY, k1, nonce, "sign-pre"),
            "sign_post_flip_fraction": _sign_flip_fraction(FIXTURE_KEY, k1, nonce, "sign-post"),
            "perm_change_fraction": _perm_change_fraction(FIXTURE_KEY, k1, nonce),
            "ciphertext_cosine_abs_mean": float(np.mean(cos_abs)),
            "ciphertext_cosine_abs_max": float(np.max(cos_abs)),
        })

    # positive control: identical key -> no change, perfect correlation.
    ctrl_cos = []
    for i in range(N_LATENT):
        ct = transform(latents[i], FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION).to(torch.float64)
        ctrl_cos.append(float(torch.dot(ct, base_ct[i]) / (ct.norm() * base_ct[i].norm())))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)

    ham = np.array([r["subkey_bit_hamming_fraction"] for r in records])
    spre = np.array([r["sign_pre_flip_fraction"] for r in records])
    spost = np.array([r["sign_post_flip_fraction"] for r in records])
    perm = np.array([r["perm_change_fraction"] for r in records])
    cos_mean = np.array([r["ciphertext_cosine_abs_mean"] for r in records])
    cos_max = np.array([r["ciphertext_cosine_abs_max"] for r in records])

    # figure: avalanche histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(ham, bins=24, color="#4c72b0", alpha=0.85)
    ax1.axvline(0.5, color="k", ls="--", lw=1, label="ideal 0.5")
    ax1.set_title("Derived sign-subkey Hamming fraction\n(256 single-bit key flips)")
    ax1.set_xlabel("fraction of subkey bits flipped"); ax1.set_ylabel("count"); ax1.legend()
    ax2.hist(cos_mean, bins=24, color="#c44e52", alpha=0.85)
    ax2.axvline(0.0, color="k", ls="--", lw=1, label="ideal 0")
    ax2.set_title("Ciphertext |cosine| (orig vs 1-bit-flipped key)\n"
                  f"d={DIM}, {N_LATENT} latents, rounds={ROUNDS}, perm={USE_PERMUTATION}")
    ax2.set_xlabel("|cosine| between ciphertexts"); ax2.set_ylabel("count"); ax2.legend()
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180); plt.close(fig)

    summary = {
        "schema_version": 1, "task_id": "Q10.5", "seed": SEED,
        "fixture": "d7_cpu_security.FIXTURE_KEY (public reproducibility fixture; not key material)",
        "dimension": DIM, "rounds": ROUNDS, "permutation": USE_PERMUTATION,
        "n_bit_flips": n_bits, "n_latents": N_LATENT,
        "subkey_hamming_fraction": {"mean": float(ham.mean()), "min": float(ham.min()), "max": float(ham.max())},
        "sign_pre_flip_fraction": {"mean": float(spre.mean()), "min": float(spre.min()), "max": float(spre.max())},
        "sign_post_flip_fraction": {"mean": float(spost.mean()), "min": float(spost.min()), "max": float(spost.max())},
        "perm_change_fraction": {"mean": float(perm.mean()), "min": float(perm.min()), "max": float(perm.max())},
        "ciphertext_cosine_abs": {"mean": float(cos_mean.mean()), "max_over_flips": float(cos_max.max())},
        "positive_control_same_key_cosine_min": float(min(ctrl_cos)),
        "note": (
            "DIAGNOSTIC of KDF diffusion, NOT a security proof; key-space size is not a "
            "security argument (skill sec. 8). A one-bit key change flips ~50% of the sign "
            "subkey bits and yields near-zero ciphertext correlation; the permutation changes "
            "at almost every position. Same-key positive control gives cosine 1.0."
        ),
    }
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {_rel(args.out)} ({n_bits} bit flips); "
          f"subkey Hamming mean={ham.mean():.4f}, sign-pre flip mean={spre.mean():.4f}, "
          f"perm change mean={perm.mean():.4f}, ciphertext |cos| mean={cos_mean.mean():.2e} "
          f"(max {cos_max.max():.2e}); same-key control cos_min={min(ctrl_cos):.6f}; fig {_rel(args.figure)}")


if __name__ == "__main__":
    main()
