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

Cost.  d basis queries is the UPPER BOUND for an arbitrary linear map.  For the
predecessor's S0 / S1 a single query suffices (AF-035):

  S0  query the all-ones plaintext  -> ciphertext = k (the whole sign key)
  S1  query a plaintext with pairwise-distinct entries (v_i = i + 1)
      -> P(k * v): the positions of the values give pi, the signs give k

Both are executed here on the predecessor's REAL key (password-derived, d = 65536,
latent shape (1,256,256)) with query_count = 1, and the recovered map decrypts a
fresh random plaintext and a real E2.3 latent bit-exactly.  The d-query full
basis reconstruction is executed at d = 4096 (generic Oracle, seed key) and at
d = 65536 (predecessor key, batched oracle that is checked to be bit-identical to
``apply_legacy_key``).  ``attack/verify_cpa.py`` (SOT-WHT structured transform,
D = 1024, the script named by WORKFLOW A4.4) is executed as a reference block.

Controls: (i) leaving one basis query out breaks the reconstruction; (ii) a map
learnt under key A does not decrypt ciphertexts made under key B; (iii) an
oracle that re-keys on every query (what a per-image nonce does) gives a matrix
that does not decrypt a fresh plaintext.  Raw key material is never written.

CPU only, seed 1911, deterministic.

    scripts/run_cfg_ddim.sh python -m scripts.a4_4_cpa
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import runpy
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
from scripts.e1_ddim_runner import sha256_file  # noqa: E402
from scripts.e2_legacy_wrapper import apply_legacy_key, build_legacy_components  # noqa: E402
from scripts.run_artifacts import git_value, imported_project_modules  # noqa: E402
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
SHAPE_FULL = (1, 256, 256)  # predecessor key shape, d = 65536
D_FULL = int(np.prod(SHAPE_FULL))
VERIFY_CPA_PATH = ROOT / "attack/verify_cpa.py"
FULL_BASIS_BATCH = 256
KEY_A_SEED = 1911
KEY_B_SEED = 2026  # a different key, used only by the negative control


class Oracle:
    """Chosen-plaintext encryption oracle of the predecessor (full mask, fixed key unless told otherwise)."""

    def __init__(self, scheme: str, key_seed: int, rekey_every_query: bool = False, shape: tuple = SHAPE):
        self.shape = shape
        self.scheme = scheme
        self.key_seed = key_seed
        self.rekey = rekey_every_query
        self.queries = 0
        self.anonymizer = DiffusionAnonymizer(latent_shape=shape, mask_type="full", margin=0,
                                              use_permutation=scheme == "S1")
        self._key = self._make_key(key_seed)

    def _make_key(self, seed: int):
        if self.scheme == "S0":
            return RademacherKey(shape=self.shape, seed=seed)
        return SignedPermutationKey(shape=self.shape, seed=seed)

    def __call__(self, x_flat: np.ndarray) -> np.ndarray:
        if self.rekey:
            self._key = self._make_key(self.key_seed + 1 + self.queries)
        self.queries += 1
        x = torch.from_numpy(np.ascontiguousarray(x_flat, dtype=np.float32)).reshape(1, *self.shape)
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


# ---------------------------------------------------------------------------
# 1-query attack (S0 / S1) and d = 65536 executions (AF-035)
# ---------------------------------------------------------------------------
def one_query_recover(scheme: str, c_query: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read (positions, signs) of the monomial map from the ONE ciphertext of the crafted plaintext.

    S0: plaintext = all ones      -> c = k;  positions = identity, signs = c.
    S1: plaintext v_i = i + 1     -> c = P(k * v); the value |c_j| - 1 is the source index of output j,
        and sign(c_j) is k at that source.  positions[i] = j with |c_j| = i + 1.
    """
    if scheme == "S0":
        positions = np.arange(d)
        signs = c_query.copy()
        info = {"entries_are_pm1": bool(np.all(np.abs(c_query) == 1.0))}
    else:
        mags = np.abs(c_query)
        src = mags.astype(np.int64) - 1  # source index of output j
        is_perm = bool(np.array_equal(np.sort(src), np.arange(d)) and np.all(mags == np.rint(mags)))
        positions = np.empty(d, dtype=np.int64)
        positions[src] = np.arange(d)
        signs = np.empty(d, dtype=np.float32)
        signs[src] = np.sign(c_query)
        info = {"query_image_is_permutation_of_1_to_d": is_perm}
    return positions, signs, info


def crafted_plaintext(scheme: str, d: int) -> np.ndarray:
    if scheme == "S0":
        return np.ones(d, dtype=np.float32)
    return np.arange(1, d + 1, dtype=np.float32)  # pairwise distinct, exact in float32 for d <= 2**24


def one_query_attack(scheme: str, oracle_fn, d: int, victims: dict[str, np.ndarray]) -> dict:
    """Exactly ONE chosen-plaintext query, then bit-exact decryption of fresh victims (no key access)."""
    queries = 0
    c_query = oracle_fn(crafted_plaintext(scheme, d)); queries += 1
    positions, signs, info = one_query_recover(scheme, c_query, d)
    errors, exact = {}, {}
    for name, x in victims.items():  # validation queries are made by the legitimate encryptor, not attack queries
        x_hat = decrypt_structured(positions, signs, oracle_fn(x))
        errors[name] = max_abs(x_hat, x)
        exact[name] = bool(np.array_equal(x_hat, x))
    return {"query_count": queries, "d": d, **info,
            "fresh_plaintext_max_abs_error": errors, "bit_exact": exact,
            "recovered_positions_form_permutation": bool(np.array_equal(np.sort(positions), np.arange(d)))}


def legacy_oracle_fn(scheme: str):
    """Predecessor oracle at d = 65536 through the project's official ``apply_legacy_key`` (real password-derived key)."""
    def fn(x_flat: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(x_flat, dtype=np.float32)).reshape(1, 1, 256, 256)
        return apply_legacy_key(x, SOURCE_ROOT, scheme, "anonymize").numpy().reshape(-1)
    return fn


class LegacyBatchOracle:
    """Same map as ``apply_legacy_key(..., 'anonymize')`` but the key is built once and queries are batched."""

    def __init__(self, scheme: str):
        from guided_diffusion.anonymization import AnonymizationMask
        self.anonymizer, self.key = build_legacy_components(SOURCE_ROOT, scheme, torch.device("cpu"))
        self.mask = AnonymizationMask(shape=SHAPE_FULL, mask_type=self.anonymizer.mask_type,
                                      margin=self.anonymizer.margin).mask
        self.is_perm = isinstance(self.key, SignedPermutationKey)
        self.queries = 0

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,256,256); each batch row counts as one query."""
        self.queries += x.shape[0]
        mask = self.mask
        if mask.dim() < x.dim():
            mask = mask.unsqueeze(0).expand_as(x)
        if self.is_perm:
            return self.key.apply(x, mask)
        key = self.key.key
        if key.dim() < x.dim():
            key = key.unsqueeze(0).expand_as(x)
        return mask * (key * x) + (1 - mask) * x


def full_basis_d65536(scheme: str, streams: list[np.random.Generator], cache) -> dict:
    """d = 65536 basis-vector queries against the predecessor key; the dense matrix is never materialised."""
    oracle = LegacyBatchOracle(scheme)
    # the batched oracle must equal the official helper bit-exactly (random and crafted inputs)
    probe = torch.from_numpy(streams[1].standard_normal(D_FULL).astype(np.float32)).reshape(1, 1, 256, 256)
    arange = torch.from_numpy(crafted_plaintext("S1", D_FULL)).reshape(1, 1, 256, 256)
    equal_official = all(bool(torch.equal(oracle(t), apply_legacy_key(t, SOURCE_ROOT, scheme, "anonymize")))
                         for t in (probe, arange))
    oracle.queries = 0
    positions = np.empty(D_FULL, dtype=np.int64)
    signs = np.empty(D_FULL, dtype=np.float32)
    one_nonzero_per_column = True
    for start in range(0, D_FULL, FULL_BASIS_BATCH):
        n = min(FULL_BASIS_BATCH, D_FULL - start)
        e = torch.zeros(n, D_FULL, dtype=torch.float32)
        e[torch.arange(n), start + torch.arange(n)] = 1.0
        out = oracle(e.reshape(n, 1, 256, 256)).reshape(n, D_FULL)
        one_nonzero_per_column &= bool(((out != 0).sum(dim=1) == 1).all())
        pos = out.abs().argmax(dim=1)
        positions[start:start + n] = pos.numpy()
        signs[start:start + n] = out[torch.arange(n), pos].numpy()
    query_count = oracle.queries
    victims = {"fresh_gaussian": streams[0].standard_normal(D_FULL).astype(np.float32),
               "real_latent_index_0": np.array(cache[0], dtype=np.float32).reshape(-1)}
    errors, exact = {}, {}
    for name, x in victims.items():
        c = oracle(torch.from_numpy(x).reshape(1, 1, 256, 256)).numpy().reshape(-1)
        x_hat = decrypt_structured(positions, signs, c)
        errors[name] = max_abs(x_hat, x)
        exact[name] = bool(np.array_equal(x_hat, x))
    return {
        "d": D_FULL, "shape": list(SHAPE_FULL), "query_count": query_count,
        "batched_oracle_bit_identical_to_apply_legacy_key": equal_official,
        "every_column_has_exactly_one_nonzero": one_nonzero_per_column,
        "signs_are_pm1": bool(np.all(np.abs(signs) == 1.0)),
        "recovered_positions_form_permutation": bool(np.array_equal(np.sort(positions), np.arange(D_FULL))),
        "recovered_map_is_identity_permutation": bool(np.array_equal(positions, np.arange(D_FULL))),
        "fresh_plaintext_max_abs_error": errors, "bit_exact": exact,
    }


def one_query_controls_d65536(scheme: str, streams: list[np.random.Generator]) -> dict:
    """Negative controls for the 1-query attack at d = 65536 (generic seed-key Oracle, shape (1,256,256))."""
    victim = streams[0].standard_normal(D_FULL).astype(np.float32)
    # (a) per-query nonce: the oracle re-keys on every query, so the crafted query's key != the victim's key
    rk = Oracle(scheme, KEY_A_SEED, rekey_every_query=True, shape=SHAPE_FULL)
    c_q = rk(crafted_plaintext(scheme, D_FULL))
    pos, sgn, _ = one_query_recover(scheme, c_q, D_FULL)
    dec = decrypt_structured(pos, sgn, rk(victim))
    per_query_nonce = {"max_abs_error": max_abs(dec, victim), "coordinate_mismatch_fraction": float(np.mean(dec != victim))}
    # (b) the 1-query attack is sound under a fixed key (positive control on the same generic oracle)
    fx = Oracle(scheme, KEY_A_SEED, shape=SHAPE_FULL)
    fixed = one_query_attack(scheme, fx, D_FULL, {"fresh_gaussian": victim})
    out = {"per_query_nonce": per_query_nonce, "fixed_key_positive_control_bit_exact": fixed["bit_exact"]["fresh_gaussian"]}
    # (c) S1 only: a non-distinct crafted plaintext (all ones) does NOT reveal pi
    if scheme == "S1":
        fx2 = Oracle(scheme, KEY_A_SEED, shape=SHAPE_FULL)
        c_ones = fx2(np.ones(D_FULL, dtype=np.float32))
        dec2 = decrypt_structured(np.arange(D_FULL), c_ones, fx2(victim))  # best guess without pi: identity
        out["s1_all_ones_query_does_not_reveal_pi"] = {
            "max_abs_error": max_abs(dec2, victim), "coordinate_mismatch_fraction": float(np.mean(dec2 != victim))}
    return out


def verify_cpa_reference() -> dict:
    """Run attack/verify_cpa.py unchanged (SOT-WHT R=2, D=1024) and record what it measures."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = runpy.run_path(str(VERIFY_CPA_PATH), run_name="verify_cpa_reference")
    d_ref = int(ns["D"])
    return {
        "script": "attack/verify_cpa.py", "script_sha256": sha256_file(VERIFY_CPA_PATH),
        "transform": "SOT-WHT (R=2), the structured transform of WORKFLOW A4.4", "D": d_ref,
        "query_count": d_ref, "full_Q_reconstructed_from_D_queries": True,
        "unseen_plaintext_max_abs_error": float(ns["err"]),
        "single_query_k2_bit_agreement_one_round": float(ns["acc"]),
        "stdout_lines": len(buf.getvalue().splitlines()),
        "label": "[事實] 執行 verify_cpa.py 原檔（D=1024）；它不涵蓋前作 S0/S1，亦未於 d=65536 執行",
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
    schemes = ("S0", "S1")
    rngs = lambda lo, i: [np.random.default_rng(s) for s in spawn[lo + i * 2:lo + (i + 1) * 2]]  # noqa: E731

    # d = 4096: d-query full basis (upper bound for a general linear map), generic seed-key oracle
    results = {scheme: run_scheme(scheme, rngs(0, i), cache) for i, scheme in enumerate(schemes)}

    # d = 65536, predecessor key: ONE query, then full-basis d queries (streams 4-7)
    victim_real = np.array(cache[0], dtype=np.float32).reshape(-1)
    one_query: dict[str, dict] = {}
    one_query_d4096: dict[str, dict] = {
        scheme: one_query_attack(scheme, Oracle(scheme, KEY_A_SEED), D,
                                 {"fresh_gaussian": rngs(0, i)[0].standard_normal(D).astype(np.float32)})
        for i, scheme in enumerate(schemes)}
    full_basis: dict[str, dict] = {}
    one_query_controls: dict[str, dict] = {}
    for i, scheme in enumerate(schemes):
        gen_a, gen_b = rngs(4, i)
        victims = {"fresh_gaussian": gen_a.standard_normal(D_FULL).astype(np.float32),
                   "real_latent_index_0": victim_real}
        one_query[scheme] = one_query_attack(scheme, legacy_oracle_fn(scheme), D_FULL, victims)
        full_basis[scheme] = full_basis_d65536(scheme, [gen_b, gen_b], cache)
        one_query_controls[scheme] = one_query_controls_d65536(scheme, [np.random.default_rng(spawn[4 + i * 2].spawn(1)[0])])
    structured_reference = verify_cpa_reference()

    checks: dict[str, bool] = {}
    for scheme, r in results.items():
        checks[f"{scheme}_query_count_equals_d"] = r["query_count"] == r["d"]
        checks[f"{scheme}_map_is_signed_permutation"] = r["reconstructed_map_is_signed_permutation"]
        checks[f"{scheme}_reconstructs_decryption_map"] = r["reconstructs_decryption_map"]
        checks[f"{scheme}_fresh_plaintext_max_abs_error_zero"] = r["fresh_plaintext_max_abs_error"] == 0.0
        checks[f"{scheme}_missing_query_breaks_reconstruction"] = r["negative_control_missing_one_query"]["max_abs_error"] > 0.0
        checks[f"{scheme}_wrong_key_not_decrypted"] = r["negative_control_wrong_key"]["max_abs_error"] > 0.0
        checks[f"{scheme}_per_query_rekey_defeats_cpa"] = r["negative_control_per_query_rekey"]["coordinate_mismatch_fraction"] > 0.2
        oq, fb, ctl = one_query[scheme], full_basis[scheme], one_query_controls[scheme]
        checks[f"{scheme}_one_query_d65536_query_count_is_1"] = oq["query_count"] == 1 and oq["d"] == D_FULL
        checks[f"{scheme}_one_query_d65536_recovered_map_is_signed_permutation"] = oq["recovered_positions_form_permutation"]
        checks[f"{scheme}_one_query_d65536_fresh_plaintext_max_abs_error_zero"] = all(v == 0.0 for v in oq["fresh_plaintext_max_abs_error"].values())
        checks[f"{scheme}_one_query_d65536_bit_exact"] = all(oq["bit_exact"].values())
        checks[f"{scheme}_one_query_d4096_bit_exact"] = (one_query_d4096[scheme]["query_count"] == 1
                                                         and all(one_query_d4096[scheme]["bit_exact"].values()))
        checks[f"{scheme}_full_basis_d65536_query_count_equals_d"] = fb["query_count"] == D_FULL
        checks[f"{scheme}_full_basis_d65536_oracle_matches_official_helper"] = fb["batched_oracle_bit_identical_to_apply_legacy_key"]
        checks[f"{scheme}_full_basis_d65536_signed_permutation"] = (fb["every_column_has_exactly_one_nonzero"] and fb["signs_are_pm1"]
                                                                   and fb["recovered_positions_form_permutation"])
        checks[f"{scheme}_full_basis_d65536_fresh_plaintext_bit_exact"] = all(fb["bit_exact"].values()) and all(v == 0.0 for v in fb["fresh_plaintext_max_abs_error"].values())
        checks[f"{scheme}_one_query_fixed_key_positive_control_on_generic_oracle"] = ctl["fixed_key_positive_control_bit_exact"]
        checks[f"{scheme}_one_query_per_query_nonce_defeats_attack_d65536"] = ctl["per_query_nonce"]["coordinate_mismatch_fraction"] > 0.2
    checks["S0_map_is_diagonal"] = results["S0"]["map_is_diagonal"]
    checks["S0_full_basis_d65536_map_is_diagonal"] = full_basis["S0"]["recovered_map_is_identity_permutation"]
    checks["S1_full_basis_d65536_map_is_not_diagonal"] = not full_basis["S1"]["recovered_map_is_identity_permutation"]
    checks["S1_all_ones_query_does_not_reveal_pi"] = one_query_controls["S1"]["s1_all_ones_query_does_not_reveal_pi"]["max_abs_error"] > 0.0
    checks["verify_cpa_reference_reconstructs_Q"] = (structured_reference["unseen_plaintext_max_abs_error"] < 1e-9
                                                     and structured_reference["query_count"] == structured_reference["D"] == 1024)
    if not all(checks.values()):
        raise SystemExit(f"A4.4 checks failed: {[k for k, v in checks.items() if not v]}")

    r0, r1 = results["S0"], results["S1"]
    q0, q1 = one_query["S0"], one_query["S1"]
    max_q0 = max(q0["fresh_plaintext_max_abs_error"].values())
    max_q1 = max(q1["fresh_plaintext_max_abs_error"].values())
    conclusion = (
        f"[事實] 前作 S0／S1 的實際 CPA 成本是 1 次查詢（本腳本在 d={D_FULL}、前作以密碼導出的真實金鑰上執行）："
        "S0 查詢全 1 明文，密文即為整把符號金鑰 k；S1 查詢兩兩相異的明文（v_i=i+1），密文 P(k⊙v) 的數值給出 π、正負號給出 k。"
        f"各 query_count=1，重建的映射解密全新的隨機明文與真實 E2.3 latent（索引 0），最大絕對誤差為 {max_q0} 與 {max_q1}（bit-exact；d={D} 的 seed 金鑰 oracle 上同樣 1 次查詢 bit-exact）。"
        f"[事實] 一般線性映射的上界 d 次查詢也已執行：d={D}（形狀 {SHAPE}，seed 金鑰 oracle）與 d={D_FULL}（形狀 {SHAPE_FULL}，前作真實金鑰；"
        f"批次 oracle 已核對與 apply_legacy_key 逐位元一致）各以 d 次基底向量 e_i 讀出整個映射"
        f"（S0 為對角 diag(k)；S1 每個 e_i 回傳單一 ±1 於位置 π⁻¹(i)），解密全新明文的最大絕對誤差為 d={D} 時 {r0['fresh_plaintext_max_abs_error']}／{r1['fresh_plaintext_max_abs_error']}、"
        f"d={D_FULL} 時 {max(full_basis['S0']['fresh_plaintext_max_abs_error'].values())}／{max(full_basis['S1']['fresh_plaintext_max_abs_error'].values())}。"
        f"[事實] 對照 verify_cpa.py（WORKFLOW A4.4 指名；SOT-WHT R=2、D={structured_reference['D']}）原檔已一併執行：D 次查詢重建 Q，"
        f"對未查詢過的新明文最大絕對誤差 {structured_reference['unseen_plaintext_max_abs_error']:.3e}；它只涵蓋結構化變換，不涵蓋前作 S0／S1。"
        "[推定] 「任何固定線性映射皆可由 d 次基底查詢讀出」對 S2 的 WHT 在 d=65536 的情形由線性性推得，本腳本未在 d=65536 對 S2 執行"
        "（僅 D=1024 的 SOT-WHT 由 verify_cpa.py 驗證）。"
        "[事實] 前提是明列的：同一把固定金鑰跨所有查詢（fixed_key_premise=True）。對照實驗（d=4096）中，少一次基底查詢會使該座標無法解密"
        f"（誤差 {r0['negative_control_missing_one_query']['max_abs_error']:.3f}／{r1['negative_control_missing_one_query']['max_abs_error']:.3f}）；"
        f"用金鑰 A 學到的映射解金鑰 B 的密文（S0 座標錯誤率 {r0['negative_control_wrong_key']['coordinate_mismatch_fraction']:.3f}、"
        f"S1 {r1['negative_control_wrong_key']['coordinate_mismatch_fraction']:.3f}）、"
        f"以及每次查詢皆換金鑰的 oracle（S0 錯誤率 {r0['negative_control_per_query_rekey']['coordinate_mismatch_fraction']:.3f}、"
        f"S1 {r1['negative_control_per_query_rekey']['coordinate_mismatch_fraction']:.3f}）皆無法解密；"
        f"d={D_FULL} 的 1 次查詢攻擊遇到每次查詢換金鑰（per-query nonce）的 oracle 時，座標錯誤率 S0 "
        f"{one_query_controls['S0']['per_query_nonce']['coordinate_mismatch_fraction']:.3f}、S1 {one_query_controls['S1']['per_query_nonce']['coordinate_mismatch_fraction']:.3f}；"
        f"S1 若查詢全 1（非相異）則無法得到 π（座標錯誤率 {one_query_controls['S1']['s1_all_ones_query_does_not_reveal_pi']['coordinate_mismatch_fraction']:.3f}）。"
        "[推定] 阻止 CPA 跨影像的是 per-image nonce（S2），固定 Q 的前提在此明列；若 S2 對每張影像使用新的 nonce 衍生變換，"
        "所學到的映射不可跨影像重用，但這需由 S2 的實作與 nonce 唯一性檢查另行驗證，本表僅驗證前作（固定金鑰）基線。"
    )

    status = git_value("status", "--porcelain", "--untracked-files=no")
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "created_at": created_at,
        "command": COMMAND,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": status.splitlines(),
        "git_dirty": bool(status),
        "imported_project_modules": imported_project_modules(),
        "seed": SEED,
        "seed_streams": ("np.random.default_rng for np.random.SeedSequence(1911).spawn(8): streams 0-1 S0 / 2-3 S1 (d=4096 full basis; fresh plaintext = stream 0 of the pair, also reused as the d=4096 1-query victim, drawn from a fresh generator); "
                         "streams 4-5 S0 / 6-7 S1 (d=65536: first stream = fresh Gaussian victim of the 1-query attack, second = probe/victim of the full basis; "
                         "the negative controls derive their victim from SeedSequence child of stream 4/6 seed)"),
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
            "real_latent_index_for_d65536_victim": 0,
            "oracle_d4096": f"predecessor DiffusionAnonymizer.anonymize_latent, mask full, margin 0, shape {SHAPE}, key seeds {KEY_A_SEED} (attack) / {KEY_B_SEED} (wrong-key control); seeds never written to disk as key material",
            "oracle_d65536": "scripts.e2_legacy_wrapper.apply_legacy_key (predecessor password-derived key, full mask, shape (1,256,256)); full basis via a batched oracle checked bit-identical to it",
            "structured_reference_script": "attack/verify_cpa.py",
        },
        "premise": "FIXED key across all chosen-plaintext queries (state of the predecessor: one shared key for all images)",
        "raw_key_material_recorded": False,
        "dimension_note": (f"[事實] d={D} 與 d={D_FULL} 的 d 次基底查詢皆已執行；d={D_FULL} 的 1 次查詢攻擊已在前作真實金鑰上執行。"
                           "S2 的 WHT 在 d=65536 未執行（[推定]）。"),
        "query_cost": {
            "general_linear_map_upper_bound": "d basis queries",
            "predecessor_S0_minimal": 1, "predecessor_S1_minimal": 1,
            "S0_crafted_plaintext": "all ones -> ciphertext = k",
            "S1_crafted_plaintext": "v_i = i + 1 (pairwise distinct) -> P(k*v) reveals (pi, k)",
        },
        "deviation_from_workflow": (
            "WORKFLOW A4.4 names attack/verify_cpa.py, which tests only the structured SOT-WHT transform (D=1024). This script is a new one "
            "that covers the predecessor's S0/S1 keys (diagonal and signed-permutation) and, unlike verify_cpa.py, reports the actual "
            "predecessor cost: ONE query at d=65536 (the d-query basis attack is the upper bound for a general linear map). "
            "verify_cpa.py is therefore also executed unchanged and recorded as `structured_transform_reference`."),
        "structured_transform_reference": structured_reference,
        "results": results,
        "one_query_attack_d4096": one_query_d4096,
        "one_query_attack_d65536": one_query,
        "one_query_controls_d65536": one_query_controls,
        "full_basis_d65536": full_basis,
        "checks": checks,
        "conclusion": conclusion,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["scheme", "d", "query_count", "query_count_minimal", "reconstructs_decryption_map",
                         "fresh_plaintext_max_abs_error", "minimal_attack_fresh_plaintext_max_abs_error",
                         "key_source", "fixed_key_premise"])
        for scheme, r in results.items():
            # d=4096 rows: both the d-query attack and the 1-query attack ran on the seed-key oracle
            writer.writerow([scheme, r["d"], r["query_count"], one_query_d4096[scheme]["query_count"], r["reconstructs_decryption_map"],
                             r["fresh_plaintext_max_abs_error"], max(one_query_d4096[scheme]["fresh_plaintext_max_abs_error"].values()),
                             "seed_key_oracle_d4096", r["fixed_key_premise"]])
        for scheme in schemes:
            fb, oq = full_basis[scheme], one_query[scheme]
            writer.writerow([scheme, fb["d"], fb["query_count"], oq["query_count"], all(fb["bit_exact"].values()),
                             max(fb["fresh_plaintext_max_abs_error"].values()), max(oq["fresh_plaintext_max_abs_error"].values()),
                             "predecessor_legacy_key_d65536", True])
    print(json.dumps({"checks": checks, "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
