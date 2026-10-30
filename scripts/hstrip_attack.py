#!/usr/bin/env python3
"""H-strip attack: the public, self-inverse Walsh-Hadamard H can be re-applied by
any attacker.  If nothing keyed sits AFTER H, applying H to the published transform
output recovers D_pre*z, whose per-coordinate magnitudes equal |z| -- reopening the
sorted-magnitude linkage attack.

This quantifies, on real cached latents (d=65536), how much of the magnitude
fingerprint |u| a one-shot H-strip recovers for:

  bare HD    = H . D_pre              (SORF-style single block; nothing after H)
  D.H.D      = D_post . H . D_pre     (deployment R=1, P off)
  D.P.H.D    = D_post . P . H . D_pre (historical R=1 / R=2, P on)

Metric: sort-mag retention = ||sort(|strip(y)|) - sort(|u|)|| / ||u||.
  ~0  => |z| magnitudes fully recovered (VULNERABLE)
  >>0 => attacker gains nothing (SAFE).

CPU only, seed 1911, deterministic.  Reuses artifacts/t2bb_latent_cache (no inversion).

  scripts/run_cfg_ddim.sh python -m scripts.hstrip_attack
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crypto.sot import fwht, transform  # noqa: E402
from src.crypto.prng import sign_vector  # noqa: E402
from src.crypto.s2_fixed_norm import normalize_to_fixed_norm  # noqa: E402
from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

SEED = 1911
D = 65536


def bare_hd(u: torch.Tensor, nonce: bytes) -> torch.Tensor:
    s = torch.as_tensor(sign_vector(FIXTURE_KEY, nonce, 0, "sign-pre", D), dtype=u.dtype)
    return fwht((u.reshape(1, -1) * s)).reshape(-1)


def sortmag_retention(strip: torch.Tensor, u: torch.Tensor) -> float:
    a = torch.sort(strip.reshape(-1).abs()).values
    b = torch.sort(u.reshape(-1).abs()).values
    return float((a - b).norm() / b.norm())


def naive_sorted_abs(y: torch.Tensor, u: torch.Tensor) -> float:
    """Retention WITHOUT stripping H (what a naive attacker sees on the stored y)."""
    return sortmag_retention(y, u)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/t2bb_latent_cache/latents")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_hstrip.csv")
    add_force_flag(ap)
    args = ap.parse_args()
    json_out = args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)
    torch.manual_seed(SEED)

    files = sorted(glob.glob(str(args.cache_dir / "*.npy")))[: args.n]
    us, nonces = [], []
    for i, f in enumerate(files):
        z = torch.from_numpy(np.load(f).reshape(1, 1, 256, 256).astype(np.float64))
        u, _ = normalize_to_fixed_norm(z)
        us.append(u)
        nonces.append(nonce_for(f"hstrip::{i}"))

    variants = {
        "bare_HD":   dict(bare=True),
        "D.H.D":     dict(rounds=1, perm=False),
        "D.P.H.D":   dict(rounds=1, perm=True),
    }
    rows = []
    for name, cfg in variants.items():
        strip_ret, naive_ret = [], []
        for u, nc in zip(us, nonces):
            if cfg.get("bare"):
                y = bare_hd(u, nc)
            else:
                y = transform(u, FIXTURE_KEY, nc, rounds=cfg["rounds"], use_permutation=cfg["perm"])
            strip = fwht(y.reshape(1, -1)).reshape(-1)          # attacker re-applies public H
            strip_ret.append(sortmag_retention(strip, u))
            naive_ret.append(naive_sorted_abs(y, u))
        row = {
            "variant": name,
            "n": len(us),
            "hstrip_sortmag_retention_mean": round(float(np.mean(strip_ret)), 6),
            "naive_sortmag_retention_mean": round(float(np.mean(naive_ret)), 6),
            "vulnerable": bool(np.mean(strip_ret) < 0.01),
            "note": "retention~0 => |z| magnitudes recovered after H-strip => sorted-mag linkage reopens",
        }
        rows.append(row)
        print(f"{name:10s} H-strip retention={row['hstrip_sortmag_retention_mean']:.4f} "
              f"naive={row['naive_sortmag_retention_mean']:.4f} vulnerable={row['vulnerable']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "n", "hstrip_sortmag_retention_mean", "naive_sortmag_retention_mean", "vulnerable", "note"]
    with args.out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    json_out.write_text(json.dumps({
        "schema_version": 1, "task_id": "H-strip", "seed": SEED, "d": D,
        "key_source": "d7 FIXTURE_KEY", "cache": str(args.cache_dir),
        "finding": ("bare HD (H.D) is H-strip-vulnerable: the public self-inverse H is removed "
                    "and |z| recovered; D.H.D and D.P.H.D keep a keyed op after H and stay at random."),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
