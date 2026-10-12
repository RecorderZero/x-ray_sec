#!/usr/bin/env python3
"""A4.3 known-plaintext attack (KPA) on the predecessor's S0 / S1 latent keys.

Threat model T3: the attacker holds ONE (z, z_ano) pair produced under the
deployment key and wants the key (or an equivalent decryptor) to open other
ciphertexts made with the same key.

  S0  z_ano = k (.) z                  ->  k_hat = sign(z_ano) * sign(z)
  S1  z_ano[p] = k[pi[p]] * z[pi[p]]   ->  pi_hat by sorting |z_ano| against |z|
                                           (equal magnitudes pair up by rank),
                                           k_hat from the signs

Real float32 latents (d = 65536) contain tied magnitudes, so a single pair does
not identify pi inside a tie group; extra pairs under the same key split the
groups (the composite key (|z_1|, ..., |z_m|) is almost always unique).  The
d = 4096 synthetic control has all-distinct magnitudes, where S1 is exact from
one pair.  Raw key material is never written; only booleans / agreement rates.

CPU only, seed 1911, deterministic.

    scripts/run_cfg_ddim.sh python -m scripts.a4_3_kpa
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# CPU only: hide every GPU before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.env_guard import enforce_active_prefix  # noqa: E402
from scripts.e1_ddim_runner import git_value, sha256_file  # noqa: E402
from scripts.e2_legacy_wrapper import apply_legacy_key, build_legacy_components  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

enforce_active_prefix({"numpy": np, "torch": torch})
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "A4.3 must run on CPU only"

SOURCE_ROOT = ROOT / "past/SourceCode"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from guided_diffusion.anonymization import (  # noqa: E402
    DiffusionAnonymizer,
    RademacherKey,
    SignedPermutationKey,
)

SEED = 1911
TASK_ID = "A4.3"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.a4_3_kpa"
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/A4.3.json"
TABLE = ROOT / "paper_assets/tables/table_KPA.csv"
ANON_PATH = SOURCE_ROOT / "guided_diffusion/anonymization.py"
LATENT_CACHE = ROOT / "artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a/latents.npy"

REAL_TARGETS = (0, 1, 2)  # latents the attacker has the plaintext of
PAIR_STRIDE = 3  # extra pairs for target t are latents t+3, t+6, t+9
HELDOUT_OFFSET = 20  # held-out victim latent for target t is t+20 (different image, same key)
MAX_PAIRS = 4
SYN_SHAPE = (1, 64, 64)  # d = 4096
SYN_KEY_SEED = 1911  # synthetic-control key (S1 permutation seed becomes 1911+1000000 inside the class)


# ---------------------------------------------------------------------------
# attack primitives (flat float32 numpy arrays)
# ---------------------------------------------------------------------------
def tied_magnitude_coords(z_flat: np.ndarray) -> int:
    """Number of coordinates whose |z| value is shared with at least one other coordinate."""
    _, counts = np.unique(np.abs(z_flat), return_counts=True)
    return int(counts[counts > 1].sum())


def s0_kpa(z_flat: np.ndarray, z_ano_flat: np.ndarray) -> np.ndarray:
    """k_hat = sign(z_ano) * sign(z); undefined where z == 0, which the caller must exclude."""
    assert not np.any(z_flat == 0), "zero plaintext coordinate: sign not determined"
    return (np.sign(z_ano_flat) * np.sign(z_flat)).astype(np.float32)


def s1_kpa_multi(zs: list[np.ndarray], es: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Recover (k_hat, pi_hat) from m known pairs under one key.

    Position p of the ciphertext holds coordinate pi[p] of the plaintext, so the
    magnitude tuple (|e_1[p]|, ..., |e_m[p]|) equals (|z_1[pi[p]]|, ..., |z_m[pi[p]]|).
    Sorting both sides lexicographically and matching by rank recovers pi up to
    permutations inside groups of identical tuples.  Returns (k_hat, pi_hat,
    n_coords_in_ambiguous_groups).
    """
    d = zs[0].size
    za = np.stack([np.abs(z) for z in zs])  # (m, d)
    ea = np.stack([np.abs(e) for e in es])
    # np.lexsort sorts by the LAST key first; the order only has to be the same on both sides.
    order_z = np.lexsort(za[::-1])
    order_e = np.lexsort(ea[::-1])
    pi_hat = np.empty(d, dtype=np.int64)
    pi_hat[order_e] = order_z
    # coordinates sharing an identical composite magnitude tuple
    sorted_tuples = za[:, order_z]
    same_prev = np.all(sorted_tuples[:, 1:] == sorted_tuples[:, :-1], axis=0)
    in_group = np.zeros(d, dtype=bool)
    in_group[1:] |= same_prev
    in_group[:-1] |= same_prev
    k_hat = np.empty(d, dtype=np.float32)
    k_hat[pi_hat] = np.sign(es[0] * zs[0][pi_hat])
    return k_hat, pi_hat, int(in_group.sum())


def s0_decrypt(k_hat: np.ndarray, c_flat: np.ndarray) -> np.ndarray:
    return (k_hat * c_flat).astype(np.float32)


def s1_decrypt(k_hat: np.ndarray, pi_hat: np.ndarray, c_flat: np.ndarray) -> np.ndarray:
    out = np.empty_like(c_flat)
    out[pi_hat] = k_hat[pi_hat] * c_flat
    return out


def mismatch_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))


# ---------------------------------------------------------------------------
# key access (truth is used ONLY to compute booleans, never written out)
# ---------------------------------------------------------------------------
def legacy_truth(scheme: str) -> tuple[np.ndarray, np.ndarray | None]:
    _, key = build_legacy_components(SOURCE_ROOT, scheme, torch.device("cpu"))
    if scheme == "S0":
        return key.key.numpy().reshape(-1).copy(), None
    return (
        key.rademacher_key.key.numpy().reshape(-1).copy(),
        key.permutation_key.permutation.numpy().reshape(-1).copy(),
    )


def legacy_encrypt(z_flat: np.ndarray, scheme: str) -> np.ndarray:
    z = torch.from_numpy(z_flat.copy()).reshape(1, 1, 256, 256)
    return apply_legacy_key(z, SOURCE_ROOT, scheme, "anonymize").numpy().reshape(-1)


def legacy_decrypt(c_flat: np.ndarray, scheme: str) -> np.ndarray:
    c = torch.from_numpy(c_flat.copy()).reshape(1, 1, 256, 256)
    return apply_legacy_key(c, SOURCE_ROOT, scheme, "deanonymize").numpy().reshape(-1)


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def run_s0_real(cache, truth_k: np.ndarray) -> list[dict]:
    rows = []
    for t in REAL_TARGETS:
        z = np.array(cache[t], dtype=np.float32).reshape(-1)
        c = legacy_encrypt(z, "S0")
        k_hat = s0_kpa(z, c)
        exact = bool(np.array_equal(k_hat, truth_k))
        pair_reproduced = bool(np.array_equal(k_hat * z, c))
        held = np.array(cache[t + HELDOUT_OFFSET], dtype=np.float32).reshape(-1)
        c_held = legacy_encrypt(held, "S0")
        dec = s0_decrypt(k_hat, c_held)
        rows.append({
            "latent_index": t, "heldout_latent_index": t + HELDOUT_OFFSET,
            "pairs_used": 1, "d": int(z.size), "tied_magnitude_coords": tied_magnitude_coords(z),
            "zero_coords": int(np.sum(z == 0)),
            "key_exact_recovery": exact, "pair_reproduced_bit_exact": pair_reproduced,
            "heldout_decrypt_bit_exact": bool(np.array_equal(dec, held)),
            "heldout_cosine": cosine(dec, held), "heldout_mismatch_fraction": mismatch_fraction(dec, held),
            "heldout_decrypt_equals_legacy_deanonymize": bool(np.array_equal(dec, legacy_decrypt(c_held, "S0"))),
        })
    return rows


def run_s1_real(cache, truth_k: np.ndarray, truth_pi: np.ndarray) -> list[dict]:
    rows = []
    for t in REAL_TARGETS:
        idx = [t + PAIR_STRIDE * j for j in range(MAX_PAIRS)]
        zs = [np.array(cache[i], dtype=np.float32).reshape(-1) for i in idx]
        es = [legacy_encrypt(z, "S1") for z in zs]
        held = np.array(cache[t + HELDOUT_OFFSET], dtype=np.float32).reshape(-1)
        c_held = legacy_encrypt(held, "S1")
        series = []
        minimal = None
        for m in range(1, MAX_PAIRS + 1):
            k_hat, pi_hat, ambiguous = s1_kpa_multi(zs[:m], es[:m])
            exact = bool(np.array_equal(pi_hat, truth_pi) and np.array_equal(k_hat, truth_k))
            pair_ok = bool(np.array_equal(s1_decrypt(k_hat, pi_hat, es[0]), zs[0]))
            dec = s1_decrypt(k_hat, pi_hat, c_held)
            series.append({
                "pairs_used": m,
                "tied_magnitude_coords": tied_magnitude_coords(zs[0]) if m == 1 else ambiguous,
                "coords_in_ambiguous_groups": ambiguous,
                "key_exact_recovery": exact,
                "permutation_mismatch_fraction": mismatch_fraction(pi_hat, truth_pi),
                "sign_mismatch_fraction": mismatch_fraction(k_hat, truth_k),
                "known_pair_reproduced_bit_exact": pair_ok,
                "heldout_decrypt_bit_exact": bool(np.array_equal(dec, held)),
                "heldout_mismatch_fraction": mismatch_fraction(dec, held),
                "heldout_cosine": cosine(dec, held),
                "heldout_decrypt_equals_legacy_deanonymize": bool(
                    np.array_equal(dec, legacy_decrypt(c_held, "S1"))
                ),
            })
            if minimal is None and exact:
                minimal = m
        rows.append({
            "latent_index": t, "pair_latent_indices": idx[:MAX_PAIRS], "heldout_latent_index": t + HELDOUT_OFFSET,
            "d": int(zs[0].size), "distinct_magnitudes_single_latent": int(np.unique(np.abs(zs[0])).size),
            "zero_coords": int(np.sum(zs[0] == 0)),
            "series": series, "minimal_pairs_for_exact_key": minimal,
        })
    return rows


def distinct_magnitude_normal(rng: np.random.Generator, d: int) -> np.ndarray:
    """N(0,1) float32 vector whose |.| values are all distinct (colliding entries are redrawn)."""
    z = rng.standard_normal(d).astype(np.float32)
    while True:
        _, first, counts = np.unique(np.abs(z), return_index=True, return_counts=True)
        keep = np.zeros(d, dtype=bool)
        keep[first] = True
        bad = ~keep
        if not bad.any():
            return z
        z[bad] = rng.standard_normal(int(bad.sum())).astype(np.float32)


def synthetic_case(scheme: str, rng: np.random.Generator) -> dict:
    d = int(np.prod(SYN_SHAPE))
    z = distinct_magnitude_normal(rng, d)
    held = rng.standard_normal(d).astype(np.float32)
    distinct = int(np.unique(np.abs(z)).size)
    anonymizer = DiffusionAnonymizer(latent_shape=SYN_SHAPE, mask_type="full", margin=0,
                                     use_permutation=scheme == "S1")
    if scheme == "S0":
        key = RademacherKey(shape=SYN_SHAPE, seed=SYN_KEY_SEED)
    else:
        key = SignedPermutationKey(shape=SYN_SHAPE, seed=SYN_KEY_SEED)

    def enc(x: np.ndarray) -> np.ndarray:
        return anonymizer.anonymize_latent(torch.from_numpy(x.copy()).reshape(1, *SYN_SHAPE), key).numpy().reshape(-1)

    c, c_held = enc(z), enc(held)
    if scheme == "S0":
        k_hat = s0_kpa(z, c)
        exact = bool(np.array_equal(k_hat, key.key.numpy().reshape(-1)))
        dec = s0_decrypt(k_hat, c_held)
        ambiguous = 0
    else:
        k_hat, pi_hat, ambiguous = s1_kpa_multi([z], [c])
        exact = bool(np.array_equal(pi_hat, key.permutation_key.permutation.numpy())
                     and np.array_equal(k_hat, key.rademacher_key.key.numpy().reshape(-1)))
        dec = s1_decrypt(k_hat, pi_hat, c_held)
    return {
        "scheme": scheme, "d": d, "pairs_used": 1, "distinct_magnitudes": distinct,
        "tied_magnitude_coords": tied_magnitude_coords(z), "coords_in_ambiguous_groups": ambiguous,
        "key_exact_recovery": exact,
        "heldout_decrypt_bit_exact": bool(np.array_equal(dec, held)),
        "heldout_mismatch_fraction": mismatch_fraction(dec, held),
        "heldout_decrypt_equals_legacy_class_inverse": bool(np.array_equal(
            dec, anonymizer.deanonymize_latent(torch.from_numpy(c_held.copy()).reshape(1, *SYN_SHAPE), key).numpy().reshape(-1))),
    }


def wrong_key_control(cache, truth_k: np.ndarray) -> dict:
    """k_hat learned under key A must NOT open ciphertexts made under an unrelated key B."""
    z = np.array(cache[0], dtype=np.float32).reshape(-1)
    k_hat = s0_kpa(z, legacy_encrypt(z, "S0"))
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(8)[7])
    other = (2.0 * rng.integers(0, 2, size=z.size) - 1.0).astype(np.float32)
    held = np.array(cache[HELDOUT_OFFSET], dtype=np.float32).reshape(-1)
    dec = s0_decrypt(k_hat, other * held)
    return {
        "key_B_sign_agreement_with_recovered_key": float(np.mean(other == k_hat)),
        "heldout_decrypt_bit_exact_under_key_B": bool(np.array_equal(dec, held)),
        "heldout_cosine_under_key_B": cosine(dec, held),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--table", type=Path, default=TABLE)
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.output, args.table], args.force)

    cache = np.load(LATENT_CACHE, mmap_mode="r")
    assert cache.shape == (200, 1, 256, 256) and cache.dtype == np.float32
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(8)]

    truth_k0, _ = legacy_truth("S0")
    truth_k1, truth_pi1 = legacy_truth("S1")
    s0_real = run_s0_real(cache, truth_k0)
    s1_real = run_s1_real(cache, truth_k1, truth_pi1)
    s0_syn = synthetic_case("S0", streams[0])
    s1_syn = synthetic_case("S1", streams[1])
    wrong = wrong_key_control(cache, truth_k0)

    # model check: the attacker's S1 equation z_ano[p] = k[pi[p]] z[pi[p]] equals the predecessor's apply()
    z0 = np.array(cache[0], dtype=np.float32).reshape(-1)
    model_ok = bool(np.array_equal((truth_k1 * z0)[truth_pi1], legacy_encrypt(z0, "S1")))

    s1_single = [r["series"][0] for r in s1_real]
    s1_multi_min = [r["minimal_pairs_for_exact_key"] for r in s1_real]
    checks = {
        "S0_real_key_exact_all": all(r["key_exact_recovery"] for r in s0_real),
        "S0_real_heldout_decrypt_bit_exact_all": all(r["heldout_decrypt_bit_exact"] for r in s0_real),
        "S0_real_equals_legacy_deanonymize_all": all(r["heldout_decrypt_equals_legacy_deanonymize"] for r in s0_real),
        "S0_synthetic_exact_and_heldout_bit_exact": s0_syn["key_exact_recovery"] and s0_syn["heldout_decrypt_bit_exact"],
        "S1_synthetic_distinct_magnitudes_precondition": s1_syn["distinct_magnitudes"] == s1_syn["d"],
        "S1_synthetic_exact_and_heldout_bit_exact": s1_syn["key_exact_recovery"] and s1_syn["heldout_decrypt_bit_exact"],
        "S1_synthetic_equals_class_inverse": s1_syn["heldout_decrypt_equals_legacy_class_inverse"],
        "S1_attack_model_matches_predecessor_apply": model_ok,
        "S1_real_single_pair_reproduces_known_pair": all(s["known_pair_reproduced_bit_exact"] for s in s1_single),
        "S1_real_ties_present_so_single_pair_is_ambiguous": all(s["tied_magnitude_coords"] > 0 for s in s1_single)
        and not any(s["key_exact_recovery"] for s in s1_single),
        "S1_real_multi_pair_resolves_ties_exactly": all(m is not None for m in s1_multi_min),
        "S1_real_multi_pair_heldout_bit_exact": all(
            r["series"][r["minimal_pairs_for_exact_key"] - 1]["heldout_decrypt_bit_exact"] for r in s1_real
            if r["minimal_pairs_for_exact_key"] is not None),
        "S1_real_equals_legacy_deanonymize_when_exact": all(
            r["series"][r["minimal_pairs_for_exact_key"] - 1]["heldout_decrypt_equals_legacy_deanonymize"]
            for r in s1_real if r["minimal_pairs_for_exact_key"] is not None),
        "negative_control_wrong_key_not_decrypted": (not wrong["heldout_decrypt_bit_exact_under_key_B"])
        and abs(wrong["key_B_sign_agreement_with_recovered_key"] - 0.5) < 0.01,
        "no_zero_plaintext_coordinates": all(r["zero_coords"] == 0 for r in s0_real) and all(r["zero_coords"] == 0 for r in s1_real),
    }
    if not all(checks.values()):
        raise SystemExit(f"A4.3 checks failed: {[k for k, v in checks.items() if not v]}")

    tied = [r["tied_magnitude_coords"] for r in s0_real]
    mism1 = [s["heldout_mismatch_fraction"] for s in s1_single]
    conclusion = (
        "[事實] S0：僅需 1 對已知 (z, z_ano)，k_hat = sign(z_ano)·sign(z) 在 3 張真實 latent（d=65536）與 d=4096 合成對照上皆與金鑰逐位元相同，"
        "並可逐位元精確解開另一張未見影像的密文（held-out 解密 bit-exact，且與前作 deanonymize 路徑一致）。"
        f"[事實] S1 在 d=4096、量值互異的合成 latent 上，1 對即可由量值排序＋符號精確還原 (k, π)，held-out 解密 bit-exact。"
        f"[事實] 真實 float32 latent（d=65536）的量值有重複：每張有 {min(tied)}–{max(tied)} 個座標落在重複量值群組內；"
        f"單對時 π 在群組內無法唯一決定，3 張中 key_exact_recovery 皆為 False，held-out 解密不 bit-exact，錯位座標比例 "
        f"{min(mism1):.2e}–{max(mism1):.2e}（已知的那一對本身仍被精確重現）。"
        f"[事實] 於同一金鑰下再加已知對可拆開重複群組：此 3 個目標分別需 {s1_multi_min} 對即還原完整 (k, π)，之後 held-out 解密 bit-exact。"
        "[推定] 單對殘差只由浮點量值重複造成，非 S1 的安全性來源；A3.2 E4 已示範在縮小 seed 空間以種子搜尋可解決同樣歧義"
        "（還原 (k, π) 後逐一搜尋 sign/perm seed），且前作全部影像共用一把金鑰，攻擊者本就能取得多對。"
        "[假設] 此處『多對』假設攻擊者持有同一金鑰下的多組 (z, z_ano)；若僅有一對且不做 seed 搜尋，S1 在真實 latent 上只能得到近似而非精確的金鑰。"
    )

    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "created_at": created_at,
        "command": COMMAND,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "git_commit": git_value("rev-parse", "HEAD"),
        "seed": SEED,
        "seed_streams": "np.random.default_rng for each of np.random.SeedSequence(1911).spawn(8): stream 0 = S0 synthetic latents, 1 = S1 synthetic latents, 7 = wrong-key control",
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cpu (CUDA_VISIBLE_DEVICES='' set before importing torch)",
        "inputs": {
            "predecessor_source": "past/SourceCode/guided_diffusion/anonymization.py",
            "predecessor_source_sha256": sha256_file(ANON_PATH),
            "real_latent_cache": str(LATENT_CACHE.relative_to(ROOT)),
            "real_latent_cache_sha256": sha256_file(LATENT_CACHE),
            "real_target_latent_indices": list(REAL_TARGETS),
            "pair_latent_index_rule": f"target t uses latents t + {PAIR_STRIDE}*j (j=0..{MAX_PAIRS - 1}) as pairs",
            "heldout_latent_index_rule": f"t + {HELDOUT_OFFSET}",
            "real_key": "legacy S0/S1 full-mask key (build_legacy_components, d=65536); key material compared in memory only, never written",
            "synthetic_control": f"N(0,1) float32, shape {SYN_SHAPE} (d=4096), key seed {SYN_KEY_SEED} through the predecessor classes",
        },
        "premise": "one or more known (z, z_ano) pairs under ONE fixed key; held-out victim ciphertext made with the same key",
        "raw_key_material_recorded": False,
        "results": {
            "S0_real": s0_real,
            "S1_real": s1_real,
            "S0_synthetic_d4096": s0_syn,
            "S1_synthetic_d4096": s1_syn,
            "negative_control_wrong_key_S0": wrong,
        },
        "checks": checks,
        "conclusion": conclusion,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    header = ["scheme", "d", "latent_type", "pairs_used", "tied_magnitude_coords", "exact_recovery",
              "heldout_decrypt_bit_exact", "latent_index", "heldout_mismatch_fraction"]
    with args.table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for r in s0_real:
            writer.writerow(["S0", r["d"], "real", 1, r["tied_magnitude_coords"], r["key_exact_recovery"],
                             r["heldout_decrypt_bit_exact"], r["latent_index"], f"{r['heldout_mismatch_fraction']:.6g}"])
        writer.writerow(["S0", s0_syn["d"], "synthetic", 1, s0_syn["tied_magnitude_coords"], s0_syn["key_exact_recovery"],
                         s0_syn["heldout_decrypt_bit_exact"], "synthetic", f"{s0_syn['heldout_mismatch_fraction']:.6g}"])
        for r in s1_real:
            first = r["series"][0]
            writer.writerow(["S1", r["d"], "real", 1, first["tied_magnitude_coords"], first["key_exact_recovery"],
                             first["heldout_decrypt_bit_exact"], r["latent_index"], f"{first['heldout_mismatch_fraction']:.6g}"])
            m = r["minimal_pairs_for_exact_key"]
            best = r["series"][m - 1]
            writer.writerow(["S1", r["d"], "real", m, best["coords_in_ambiguous_groups"], best["key_exact_recovery"],
                             best["heldout_decrypt_bit_exact"], r["latent_index"], f"{best['heldout_mismatch_fraction']:.6g}"])
        writer.writerow(["S1", s1_syn["d"], "synthetic", 1, s1_syn["tied_magnitude_coords"], s1_syn["key_exact_recovery"],
                         s1_syn["heldout_decrypt_bit_exact"], "synthetic", f"{s1_syn['heldout_mismatch_fraction']:.6g}"])
    print(json.dumps({"checks": checks, "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
