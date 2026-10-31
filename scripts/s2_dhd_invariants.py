#!/usr/bin/env python3
"""Re-verify the S2 cipher invariants for the D.H.D deployment (R=1, P off):

  1. reversibility   decrypt(encrypt(z)) == z           -> l1_max, l2_max (must be ~0)
  2. fixed-norm      ||protected|| == C for every image -> max |.|-C
  3. Gaussian-preserving   protected latent first four moments ~ N(0, C^2/d)
  4. orthogonality   Q^T Q == I   (explicit matrix at small d=256)

CPU only, seed 1911.  Real latents from artifacts/t2bb_latent_cache (no inversion).

  scripts/run_cfg_ddim.sh python -m scripts.s2_dhd_invariants
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crypto.sot import transform, inverse_transform  # noqa: E402
from src.crypto.s2_fixed_norm import (  # noqa: E402
    FIXED_NORM, normalize_to_fixed_norm, restore_original_norm,
)
from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

SEED = 1911
ROUNDS = 1                    # deployment rounds; PERM set via --permutation


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/t2bb_latent_cache/latents")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--permutation", action="store_true", help="P on (D.P.H.D); default off (D.H.D)")
    ap.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_s2_dhd_invariants.json")
    add_force_flag(ap)
    args = ap.parse_args()
    PERM = args.permutation
    refuse_overwrite([args.out], args.force)
    torch.manual_seed(SEED)

    files = sorted(glob.glob(str(args.cache_dir / "*.npy")))[: args.n]
    l1m = l2m = normdev = 0.0
    ks_p, moments = [], []
    for i, f in enumerate(files):
        z = torch.from_numpy(np.load(f).reshape(1, 1, 256, 256).astype(np.float64))
        nc = nonce_for(f"invariants::{i}")
        u, s = normalize_to_fixed_norm(z)
        y = transform(u, FIXTURE_KEY, nc, rounds=ROUNDS, use_permutation=PERM)
        rec = restore_original_norm(inverse_transform(y, FIXTURE_KEY, nc, rounds=ROUNDS, use_permutation=PERM), s)
        err = (rec - z).abs()
        l1m = max(l1m, float(err.sum() / err.numel()))
        l2m = max(l2m, float(err.max()))
        normdev = max(normdev, abs(float(torch.linalg.vector_norm(y)) - float(FIXED_NORM)))
        a = y.reshape(-1).numpy()
        sd = float(FIXED_NORM) / np.sqrt(a.size)             # expected per-coord std
        ks = stats.kstest(a / sd, "norm")
        ks_p.append(float(ks.pvalue))
        moments.append([float(a.mean()), float(a.std()), float(stats.skew(a)), float(stats.kurtosis(a))])

    # explicit orthogonality at small d: build Q by applying transform to basis vectors
    d = 256
    nc = nonce_for("ortho::d256")
    Q = torch.stack([transform(torch.eye(d, dtype=torch.float64)[j], FIXTURE_KEY, nc,
                               rounds=ROUNDS, use_permutation=PERM) for j in range(d)], dim=1)
    QtQ = Q.T @ Q
    ortho_err = float((QtQ - torch.eye(d, dtype=torch.float64)).abs().max())

    m = np.array(moments)
    result = {
        "schema_version": 1, "task_id": "S2-DHD-invariants", "seed": SEED,
        "transform": f"{'D.P.H.D' if PERM else 'D.H.D'} (rounds={ROUNDS}, permutation={PERM})",
        "fixed_norm_C": float(FIXED_NORM),
        "n_latents": len(files), "d": 65536,
        "reversibility": {"l1_mean_max": l1m, "l2_maxabs_max": l2m, "pass": l2m < 1e-5},
        "fixed_norm": {"max_abs_dev_from_C": normdev, "pass": normdev < 1e-3},
        "gaussian_preserving": {
            "ks_pvalue_min": min(ks_p), "mean_meanabs": float(np.abs(m[:, 0]).mean()),
            "mean_std": float(m[:, 1].mean()), "expected_std": float(FIXED_NORM) / np.sqrt(65536),
            "mean_skew": float(m[:, 2].mean()), "mean_excess_kurtosis": float(m[:, 3].mean()),
        },
        "orthogonality_d256": {"QtQ_minus_I_maxabs": ortho_err, "pass": ortho_err < 1e-10},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("reversibility", "fixed_norm", "orthogonality_d256")}, indent=2))
    print(f"gaussian: ks_pmin={min(ks_p):.3f} std={result['gaussian_preserving']['mean_std']:.4f} "
          f"(exp {result['gaussian_preserving']['expected_std']:.4f}) "
          f"skew={result['gaussian_preserving']['mean_skew']:.4f} "
          f"kurt={result['gaussian_preserving']['mean_excess_kurtosis']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
