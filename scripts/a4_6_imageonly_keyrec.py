#!/usr/bin/env python3
"""A4.6 image-only key recovery: how far can an attacker who holds only the
ORIGINAL image and the ANONYMOUS image of some known subjects get?

This bridges A4.1 (T2-WB re-inversion) and A4.3 (KPA).  A4.3 assumed the
attacker already had exact (z, z_ano) latent pairs; here the ciphertext latent
is the *re-inverted* anonymous image ``ẑ_ano`` from A4.1 (approximate), while the
plaintext latent z is still exact (inverting the original image is deterministic
and reproduces the E2.3 cache).  The recovered key is then used to decrypt
held-out victims whose anonymous images the attacker also re-inverts.

  (a) single known pair  -> baseline harm of one (original, anonymous) pair
  (b) K = 1,2,4,8 pairs  -> how attacker capability grows with more pairs
        S0: magnitude-weighted sign voting across pairs
        S1: nearest-neighbour matching of per-coordinate magnitude signatures
            (exact lexsort matching, as in A4.3, needs bit-exact magnitudes and
            collapses on re-inverted latents; NN is the robust version)

Both storage protocols (legacy_png = predecessor deployment; range_preserving
= reference).  Everything is compared against the legitimate key-holder's
decrypt of the SAME ẑ_ano, so the two differ only where the recovered key is
wrong.  Raw key material is never written -- only accuracy / similarity.

CPU only, deterministic.

    scripts/run_cfg_ddim.sh python -m scripts.a4_6_imageonly_keyrec
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.env_guard import enforce_active_prefix  # noqa: E402
from scripts.e1_ddim_runner import sha256_file  # noqa: E402
from scripts.run_artifacts import git_value, imported_project_modules  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from scripts.a4_3_kpa import (  # noqa: E402
    SOURCE_ROOT, cosine, legacy_truth, legacy_decrypt, mismatch_fraction,
    s0_decrypt,
)


def s1_decrypt_safe(k_hat: np.ndarray, pi_hat: np.ndarray, c_flat: np.ndarray) -> np.ndarray:
    """S1 decrypt tolerant of a non-bijective recovered map: unfilled source
    coordinates stay 0 (a wrong recovery is penalised, never left undefined).
    A4.3's s1_decrypt assumes a true permutation and uses np.empty."""
    out = np.zeros_like(c_flat)
    out[pi_hat] = k_hat[pi_hat] * c_flat
    return out

enforce_active_prefix({"numpy": np, "torch": torch})
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "A4.6 must run on CPU only"

SEED = 1911
TASK_ID = "A4.6"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.a4_6_imageonly_keyrec"
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/A4.6.json"
TABLE = ROOT / "paper_assets/tables/table_A4.6_imageonly_keyrec.csv"
ANON_PATH = SOURCE_ROOT / "guided_diffusion/anonymization.py"

LATENT_CACHE = ROOT / "artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a/latents.npy"
A41_RUN = ROOT / "artifacts/runs/A4.1/A4.1_20261012T083425685874Z_3132fa49"

KNOWN_POOL = (0, 1, 2, 3, 4, 5, 6, 7)   # attacker's known (original, anonymous) subjects
PAIR_COUNTS = (1, 2, 4, 8)              # a) = 1 pair; b) = up to 8
VICTIMS = tuple(range(100, 150))        # 50 held-out victims, disjoint from the pool
STORAGES = ("legacy_png", "range_preserving_png")
SCHEMES = ("S0", "S1")


# ---------------------------------------------------------------------------
# re-inverted ciphertext loader (A4.1 query cache)
# ---------------------------------------------------------------------------
def load_reinverted():
    qlat = np.load(A41_RUN / "query_latents.npy")  # (800,1,256,256)
    index = json.loads((A41_RUN / "query_index.json").read_text())
    pos = {r["sample_id"]: r["query_index"] for r in index}
    return qlat, pos


def c_hat(qlat, pos, i, scheme, storage):
    """Attacker's re-inverted ciphertext latent for image i (flat float64)."""
    return qlat[pos[f"security_v1_{i:04d}|{scheme}|{storage}"]].reshape(-1).astype(np.float64)


# ---------------------------------------------------------------------------
# multi-pair recovery from approximate ciphertexts
# ---------------------------------------------------------------------------
def s0_recover(zs, chats):
    """Magnitude-weighted sign vote: each pair estimates sign(z)·sign(ẑ_ano);
    weight by |z| (a near-zero plaintext coordinate gives an unreliable sign)."""
    votes = np.zeros(zs[0].size, dtype=np.float64)
    for z, c in zip(zs, chats):
        votes += np.abs(z) * np.sign(z) * np.sign(c)
    k_hat = np.sign(votes)
    k_hat[k_hat == 0] = 1.0  # unseen tie -> arbitrary +1
    return k_hat.astype(np.float32)


def s1_recover(zs, chats):
    """Recover (k_hat, pi_hat) by matching per-coordinate magnitude signatures.

    Source signature S[j] = (|z_1[j]|,...,|z_K[j]|); ciphertext signature
    C[p] = (|ẑ_1[p]|,...,|ẑ_K[p]|).  Position p holds the source coordinate whose
    signature is nearest to C[p] (cKDTree).  Signs from the highest-|z| pair per
    coordinate.  pi_hat follows A4.3's convention (pi_hat[p] = source coord at p).
    """
    S = np.stack([np.abs(z) for z in zs], axis=1)   # (d, K)
    C = np.stack([np.abs(c) for c in chats], axis=1)
    tree = cKDTree(S)
    _, pi_hat = tree.query(C, k=1)                   # pi_hat[p] = nearest source coord
    pi_hat = pi_hat.astype(np.int64)
    # signs: for each source coord j, use the pair whose |z_p[j]| is largest
    zstack = np.stack(zs)      # (K,d)
    cstack = np.stack(chats)
    best_pair = np.argmax(np.abs(zstack), axis=0)    # (d,) per source coord
    k_hat = np.empty(zs[0].size, dtype=np.float32)
    # ciphertext value feeding source coord j sits at the p with pi_hat[p]==j;
    # build inverse mapping (last wins for collisions, which NN may produce)
    inv = np.full(zs[0].size, -1, dtype=np.int64)
    inv[pi_hat] = np.arange(pi_hat.size)
    for j in range(zs[0].size):
        p = inv[j]
        bp = best_pair[j]
        c_at = cstack[bp, p] if p >= 0 else 0.0
        k_hat[j] = np.sign(zstack[bp, j]) * np.sign(c_at) or 1.0
    return k_hat, pi_hat


# ---------------------------------------------------------------------------
# per (scheme, storage, K) evaluation
# ---------------------------------------------------------------------------
def evaluate(cache, qlat, pos, scheme, storage, K, truth_k, truth_pi):
    idx = KNOWN_POOL[:K]
    zs = [np.array(cache[i], dtype=np.float64).reshape(-1) for i in idx]
    chats = [c_hat(qlat, pos, i, scheme, storage) for i in idx]

    if scheme == "S0":
        k_hat = s0_recover(zs, chats)
        pi_hat = None
        sign_acc = float(np.mean(k_hat == truth_k))
        perm_acc = None
    else:
        k_hat, pi_hat = s1_recover(zs, chats)
        sign_acc = float(np.mean(k_hat == truth_k))
        perm_acc = float(np.mean(pi_hat == truth_pi))

    vcos_l, vrmse_l, vlinf_l, vsign_l = [], [], [], []   # attacker vs legit decrypt
    vcos_z, vrmse_z = [], []                              # attacker decrypt vs true z
    bit_exact = 0
    for v in VICTIMS:
        cv = c_hat(qlat, pos, v, scheme, storage)
        legit = legacy_decrypt(cv.astype(np.float32), scheme).astype(np.float64)
        zt = np.array(cache[v], dtype=np.float64).reshape(-1)
        if scheme == "S0":
            atk = s0_decrypt(k_hat, cv.astype(np.float32)).astype(np.float64)
        else:
            atk = s1_decrypt_safe(k_hat, pi_hat, cv.astype(np.float32)).astype(np.float64)
        vcos_l.append(cosine(atk, legit)); vrmse_l.append(float(np.sqrt(np.mean((atk - legit) ** 2))))
        vlinf_l.append(float(np.max(np.abs(atk - legit)))); vsign_l.append(mismatch_fraction(np.sign(atk), np.sign(legit)))
        vcos_z.append(cosine(atk, zt)); vrmse_z.append(float(np.sqrt(np.mean((atk - zt) ** 2))))
        if np.array_equal(atk.astype(np.float32), legit.astype(np.float32)):
            bit_exact += 1

    return {
        "scheme": scheme, "storage": storage, "pairs_used": K,
        "known_indices": list(idx), "n_victims": len(VICTIMS),
        "key_sign_accuracy": sign_acc,
        "key_perm_accuracy": perm_acc,
        "victim_vs_legit_cosine_median": float(np.median(vcos_l)),
        "victim_vs_legit_cosine_min": float(np.min(vcos_l)),
        "victim_vs_legit_rmse_median": float(np.median(vrmse_l)),
        "victim_vs_legit_linf_median": float(np.median(vlinf_l)),
        "victim_vs_legit_sign_mismatch_median": float(np.median(vsign_l)),
        "victim_vs_truez_cosine_median": float(np.median(vcos_z)),
        "victim_vs_truez_rmse_median": float(np.median(vrmse_z)),
        "victim_decrypt_bit_exact_vs_legit": bit_exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([OUTPUT, TABLE], args.force)

    np.random.seed(SEED)
    cache = np.load(LATENT_CACHE)
    qlat, pos = load_reinverted()

    rows = []
    truth = {s: legacy_truth(s) for s in SCHEMES}
    for storage in STORAGES:
        for scheme in SCHEMES:
            tk, tpi = truth[scheme]
            for K in PAIR_COUNTS:
                rows.append(evaluate(cache, qlat, pos, scheme, storage, K, tk, tpi))
                r = rows[-1]
                print(f"{storage:22s} {scheme} K={K} | sign_acc={r['key_sign_accuracy']:.4f} "
                      f"perm_acc={r['key_perm_accuracy']} | victim vs legit cos_med="
                      f"{r['victim_vs_legit_cosine_median']:.4f} bitexact={r['victim_decrypt_bit_exact_vs_legit']}/{r['n_victims']}")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})

    result = {
        "schema_version": 1, "task_id": TASK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": COMMAND, "script_sha256": sha256_file(SCRIPT_PATH),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": git_value("status", "--porcelain", "--untracked-files=no") != "",
        "imported_project_modules": imported_project_modules(),
        "seed": SEED, "numpy_version": np.__version__, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "device": "cpu",
        "inputs": {
            "predecessor_source": str(ANON_PATH.relative_to(ROOT)),
            "predecessor_source_sha256": sha256_file(ANON_PATH),
            "exact_plaintext_latent_cache": str(LATENT_CACHE.relative_to(ROOT)),
            "exact_plaintext_latent_cache_sha256": sha256_file(LATENT_CACHE),
            "reinverted_ciphertext_run": A41_RUN.name,
            "reinverted_ciphertext_sha256": sha256_file(A41_RUN / "query_latents.npy"),
            "known_pool": list(KNOWN_POOL), "pair_counts": list(PAIR_COUNTS),
            "victims": [VICTIMS[0], VICTIMS[-1]], "n_victims": len(VICTIMS),
        },
        "premise": (
            "attacker holds exact plaintext latents (original image inverts "
            "deterministically to the E2.3 cache) and re-inverted approximate "
            "ciphertext latents (A4.1) for the known pool; recovers one fixed key "
            "and decrypts held-out victims' re-inverted ciphertexts"
        ),
        "raw_key_material_recorded": False,
        "results": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} and {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
