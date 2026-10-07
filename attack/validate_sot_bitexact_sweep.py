#!/usr/bin/env python3
"""Rebuild SOT-WHT float32 bit-exact counts over a declared seed sweep."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from defense_design_check import check_bit_exact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=65536)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=100)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for rounds in args.rounds:
        flags = [bool(check_bit_exact(args.dimension, rounds, seed)) for seed in range(args.seed_start, args.seed_stop)]
        results[str(rounds)] = {
            "passed": sum(flags), "total": len(flags), "rate": float(np.mean(flags)),
            "failed_seeds": [seed for seed, passed in zip(range(args.seed_start, args.seed_stop), flags) if not passed],
        }
        print(f"d={args.dimension} R={rounds}: {sum(flags)}/{len(flags)}")
    payload = {
        "schema_version": 1, "generator": "attack/validate_sot_bitexact_sweep.py",
        "numpy_version": np.__version__, "python": platform.python_version(),
        "dimension": args.dimension, "seed_start": args.seed_start,
        "seed_stop_exclusive": args.seed_stop, "rounds": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
