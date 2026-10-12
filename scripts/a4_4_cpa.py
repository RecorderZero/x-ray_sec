#!/usr/bin/env python3
"""A4.4 chosen-plaintext attack (CPA) on the predecessor's S0 / S1 latent keys.

Premise (stated explicitly, it is what makes the attack work): the encryption
oracle uses ONE FIXED key for every query.  The map is then a fixed linear
(monomial) matrix M with z_ano = M z:

  S0  M = diag(k)                       querying e_i returns k_i e_i
  S1  M = P diag(k)                     querying e_i returns k_i e_{p_i}, pi(p_i) = i

so d basis-vector queries read M out column by column.  The decryptor is
M^{-1} = M^T (signed permutation matrices are orthogonal) and needs no key.
A fresh random plaintext is then encrypted with the oracle and decrypted with the
reconstructed map only; the error must be exactly 0.

Dimension: d = 4096 (latent shape (1,64,64)) for speed.  The construction is
dimension-agnostic (d queries, one column each), so d = 65536 only costs 16x
more queries; it is not run here.

Controls: (i) leaving one basis query out breaks the reconstruction; (ii) a map
learnt under key A does not decrypt ciphertexts made under key B; (iii) an
oracle that re-keys on every query (what a per-image nonce does) gives a matrix
that does not decrypt a fresh plaintext.  Raw key material is never written.

CPU only, seed 1911, deterministic.

    scripts/run_cfg_ddim.sh python -m scripts.a4_4_cpa
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
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

enforce_active_prefix({"numpy": np, "torch": torch})
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "A4.4 must run on CPU only"

SOURCE_ROOT = ROOT / "past/SourceCode"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from guided_diffusion.anonymization import (  # noqa: E402
    DiffusionAnonymizer,
    RademacherKey,
    SignedPermutationKey,
)

SEED = 1911
TASK_ID = "A4.4"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.a4_4_cpa"
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/A4.4.json"
TABLE = ROOT / "paper_assets/tables/table_CPA.csv"
ANON_PATH = SOURCE_ROOT / "guided_diffusion/anonymization.py"
LATENT_CACHE = ROOT / "artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a/latents.npy"

SHAPE = (1, 64, 64)  # d = 4096
D = int(np.prod(SHAPE))
KEY_A_SEED = 1911
KEY_B_SEED = 2026  # a different key, used only by the negative control


class Oracle:
    """Chosen-plaintext encryption oracle of the predecessor (full mask, fixed key unless told otherwise)."""

    def __init__(self, scheme: str, key_seed: int, rekey_every_query: bool = False):
        self.scheme = scheme
        self.key_seed = key_seed
        self.rekey = rekey_every_query
        self.queries = 0
        self.anonymizer = DiffusionAnonymizer(latent_shape=SHAPE, mask_type="full", margin=0,
                                              use_permutation=scheme == "S1")
        self._key = self._make_key(key_seed)

    def _make_key(self, seed: int):
        if self.scheme == "S0":
            return RademacherKey(shape=SHAPE, seed=seed)
        return SignedPermutationKey(shape=SHAPE, seed=seed)

    def __call__(self, x_flat: np.ndarray) -> np.ndarray:
        if self.rekey:
            self._key = self._make_key(self.key_seed + 1 + self.queries)
        self.queries += 1
        x = torch.from_numpy(np.ascontiguousarray(x_flat, dtype=np.float32)).reshape(1, *SHAPE)
        return self.anonymizer.anonymize_latent(x, self._key).numpy().reshape(-1)


def basis_queries(oracle: Oracle, skip: int | None = None) -> np.ndarray:
    """Columns R[:, i] = oracle(e_i).  ``skip`` leaves one basis vector unqueried (negative control)."""
    matrix = np.zeros((D, D), dtype=np.float32)
    for i in range(D):
        if i == skip:
            continue
        e = np.zeros(D, dtype=np.float32)
        e[i] = 1.0
        matrix[:, i] = oracle(e)
    return matrix


def monomial_decryptor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """From the columns of M read position p_i and sign s_i of every basis image.

    Returns (positions, signs, is_monomial_pm1).  Column i has exactly one non-zero
    entry s_i at row p_i if M is a signed permutation.
    """
    nonzero_per_col = (matrix != 0).sum(axis=0)
    positions = np.abs(matrix).argmax(axis=0)
    signs = matrix[positions, np.arange(D)]
    ok = bool(np.all(nonzero_per_col == 1) and np.all(np.abs(signs) == 1.0))
    return positions, signs, ok


def decrypt_structured(positions: np.ndarray, signs: np.ndarray, c: np.ndarray) -> np.ndarray:
    """x_hat[i] = s_i * c[p_i]  (M^T c for a signed permutation M)."""
    return (signs * c[positions]).astype(np.float32)


def decrypt_dense(matrix: np.ndarray, c: np.ndarray) -> np.ndarray:
    """x_hat = M^T c with the dense matrix read from the queries."""
    return (matrix.T @ c).astype(np.float32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def run_scheme(scheme: str, streams: list[np.random.Generator], cache) -> dict:
    # --- attack: d chosen-plaintext queries, fixed key A
    oracle = Oracle(scheme, KEY_A_SEED)
    matrix = basis_queries(oracle)
    query_count = oracle.queries
    positions, signs, monomial_ok = monomial_decryptor(matrix)
    is_permutation = bool(np.array_equal(np.sort(positions), np.arange(D)))

    # --- validation: fresh plaintexts (one random N(0,1), one real latent resized to d via the first d values)
    fresh = streams[0].standard_normal(D).astype(np.float32)
    real = np.array(cache[7], dtype=np.float32).reshape(-1)[:D].copy()
    out: dict = {}
    for name, x in (("fresh_gaussian", fresh), ("fresh_real_latent_slice", real)):
        c = oracle(x)  # ciphertext made by the legitimate encryptor
        out[name] = {
            "max_abs_error_structured": max_abs(decrypt_structured(positions, signs, c), x),
            "max_abs_error_dense_matrix": max_abs(decrypt_dense(matrix, c), x),
            "bit_exact": bool(np.array_equal(decrypt_structured(positions, signs, c), x)
                              and np.array_equal(decrypt_dense(matrix, c), x)),
        }
    validation_queries = oracle.queries - query_count

    # --- negative control 1: one basis query missing
    skip = 17
    partial = basis_queries(Oracle(scheme, KEY_A_SEED), skip=skip)
    c = Oracle(scheme, KEY_A_SEED)(fresh)
    err_partial = max_abs(decrypt_dense(partial, c), fresh)

    # --- negative control 2: map learnt under key A against ciphertexts under key B
    c_b = Oracle(scheme, KEY_B_SEED)(fresh)
    dec_b = decrypt_structured(positions, signs, c_b)
    wrong_key = {"max_abs_error": max_abs(dec_b, fresh), "coordinate_mismatch_fraction": float(np.mean(dec_b != fresh))}

    # --- negative control 3: the oracle re-keys on every query (what a per-image nonce does)
    rekeyed = Oracle(scheme, KEY_A_SEED, rekey_every_query=True)
    m_rk = basis_queries(rekeyed)
    pos_rk, sgn_rk, _ = monomial_decryptor(m_rk)
    c_rk = rekeyed(fresh)
    dec_rk = decrypt_structured(pos_rk, sgn_rk, c_rk)
    per_query_rekey = {
        "queries": D,
        "max_abs_error": max_abs(dec_rk, fresh),
        "coordinate_mismatch_fraction": float(np.mean(dec_rk != fresh)),
    }

    return {
        "scheme": scheme, "d": D, "shape": list(SHAPE),
        "query_count": query_count, "validation_queries_not_counted_as_attack": validation_queries,
        "reconstructed_map_is_signed_permutation": monomial_ok and is_permutation,
        "map_is_diagonal": bool(np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0),
        "reconstructs_decryption_map": all(v["bit_exact"] for v in out.values()),
        "fresh_plaintext_max_abs_error": max(v["max_abs_error_structured"] for v in out.values()),
        "validation": out,
        "fixed_key_premise": True,
        "negative_control_missing_one_query": {"skipped_basis_index": skip, "max_abs_error": err_partial},
        "negative_control_wrong_key": wrong_key,
        "negative_control_per_query_rekey": per_query_rekey,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--table", type=Path, default=TABLE)
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.output, args.table], args.force)

    cache = np.load(LATENT_CACHE, mmap_mode="r")
    spawn = np.random.SeedSequence(SEED).spawn(8)
    results = {
        scheme: run_scheme(scheme, [np.random.default_rng(s) for s in spawn[i * 2:(i + 1) * 2]], cache)
        for i, scheme in enumerate(("S0", "S1"))
    }

    checks: dict[str, bool] = {}
    for scheme, r in results.items():
        checks[f"{scheme}_query_count_equals_d"] = r["query_count"] == r["d"]
        checks[f"{scheme}_map_is_signed_permutation"] = r["reconstructed_map_is_signed_permutation"]
        checks[f"{scheme}_reconstructs_decryption_map"] = r["reconstructs_decryption_map"]
        checks[f"{scheme}_fresh_plaintext_max_abs_error_zero"] = r["fresh_plaintext_max_abs_error"] == 0.0
        checks[f"{scheme}_missing_query_breaks_reconstruction"] = r["negative_control_missing_one_query"]["max_abs_error"] > 0.0
        checks[f"{scheme}_wrong_key_not_decrypted"] = r["negative_control_wrong_key"]["max_abs_error"] > 0.0
        checks[f"{scheme}_per_query_rekey_defeats_cpa"] = r["negative_control_per_query_rekey"]["coordinate_mismatch_fraction"] > 0.2
    checks["S0_map_is_diagonal"] = results["S0"]["map_is_diagonal"]
    if not all(checks.values()):
        raise SystemExit(f"A4.4 checks failed: {[k for k, v in checks.items() if not v]}")

    conclusion = (
        f"[事實] 固定金鑰下 d 次選擇明文即可重建等價解密映射，不需金鑰：d={D}（latent 形狀 {SHAPE}），S0 與 S1 各以 {D} 次基底向量查詢 e_i "
        f"讀出整個映射（S0 為對角 diag(k)；S1 每個 e_i 回傳單一 ±1 於位置 π⁻¹(i)，同時得到 π 與 k），"
        f"重建的映射解密全新的隨機明文與全新的真實 latent 片段，最大絕對誤差為 {results['S0']['fresh_plaintext_max_abs_error']} 與 "
        f"{results['S1']['fresh_plaintext_max_abs_error']}（bit-exact）。"
        "[事實] 這是線性映射的通性（S0 對角、S1 簽名置換、S2 的 WHT 皆然）：任何固定的 d×d 線性映射都可由 d 個基底查詢讀出，"
        "並以 M⁻¹ 解密；此攻擊的維度擴展到 d=65536 只需 16 倍查詢（本腳本未執行，屬由線性性推得）。"
        "[事實] 前提是明列的：同一把固定金鑰跨所有查詢（fixed_key_premise=True）。對照實驗中，少一次基底查詢會使該座標無法解密（誤差 {results['S0']['negative_control_missing_one_query']['max_abs_error']:.3f}／{results['S1']['negative_control_missing_one_query']['max_abs_error']:.3f}）；"
        f"用金鑰 A 學到的映射解金鑰 B 的密文（S0 座標錯誤率 {results['S0']['negative_control_wrong_key']['coordinate_mismatch_fraction']:.3f}、"
        f"S1 {results['S1']['negative_control_wrong_key']['coordinate_mismatch_fraction']:.3f}）、"
        f"以及每次查詢皆換金鑰的 oracle（S0 錯誤率 {results['S0']['negative_control_per_query_rekey']['coordinate_mismatch_fraction']:.3f}、"
        f"S1 {results['S1']['negative_control_per_query_rekey']['coordinate_mismatch_fraction']:.3f}）皆無法解密。"
        "[推定] 阻止 CPA 跨影像的是 per-image nonce（S2），固定 Q 的前提在此明列；若 S2 對每張影像使用新的 nonce 衍生變換，"
        "d 次查詢所學到的映射不可跨影像重用，但這需由 S2 的實作與 nonce 唯一性檢查另行驗證，本表僅驗證前作（固定金鑰）基線。"
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
        "seed_streams": "np.random.default_rng for np.random.SeedSequence(1911).spawn(8): streams 0-1 S0 / 2-3 S1; fresh plaintext = stream 0 of the scheme's pair",
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cpu (CUDA_VISIBLE_DEVICES='' set before importing torch)",
        "inputs": {
            "predecessor_source": "past/SourceCode/guided_diffusion/anonymization.py",
            "predecessor_source_sha256": sha256_file(ANON_PATH),
            "real_latent_cache": str(LATENT_CACHE.relative_to(ROOT)),
            "real_latent_cache_sha256": sha256_file(LATENT_CACHE),
            "real_latent_index_for_fresh_slice": 7,
            "oracle": f"predecessor DiffusionAnonymizer.anonymize_latent, mask full, margin 0, shape {SHAPE}, key seeds {KEY_A_SEED} (attack) / {KEY_B_SEED} (wrong-key control); seeds never written to disk as key material",
        },
        "premise": "FIXED key across all chosen-plaintext queries (state of the predecessor: one shared key for all images)",
        "raw_key_material_recorded": False,
        "dimension_note": f"d={D} run; d=65536 extends by linearity at 16x the query count (not executed)",
        "results": results,
        "checks": checks,
        "conclusion": conclusion,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["scheme", "d", "query_count", "reconstructs_decryption_map",
                         "fresh_plaintext_max_abs_error", "fixed_key_premise"])
        for scheme, r in results.items():
            writer.writerow([scheme, r["d"], r["query_count"], r["reconstructs_decryption_map"],
                             r["fresh_plaintext_max_abs_error"], r["fixed_key_premise"]])
    print(json.dumps({"checks": checks, "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
