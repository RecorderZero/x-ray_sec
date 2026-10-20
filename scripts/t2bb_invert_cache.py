#!/usr/bin/env python3
"""Incremental DDIM-inversion latent cache for the T2-BB linkage experiments.

Inversion is deterministic (DDIM eta=0, fixed checkpoint, null conditioning), so a
latent is cached per image keyed by file_sha256 and reused bit-identically. This
makes the shadow training set prefix-extensible: invert 200 patients now, 1000
later, re-inverting nothing (reports/t2bb_inner_linkage_protocol.md 11).

Cache layout:  <cache-dir>/manifest.json         config (checkpoint SHA, noise level)
               <cache-dir>/latents/<sha>.npy     one (256,256) float32 latent per image
               <cache-dir>/index.csv             sample_id,patient_id,pair_role,file_sha256,split

A run refuses to write into a cache built with a different checkpoint/noise level.

  scripts/run_cfg_ddim.sh python -m scripts.t2bb_invert_cache \
    --split splits/t2bb_shadow_v1.csv --max-patients 200 --checkpoint <ckpt>
  scripts/run_cfg_ddim.sh python -m scripts.t2bb_invert_cache \
    --split splits/t2bb_inner_v1.csv --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PAST = ROOT / "past" / "SourceCode"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PAST))

from scripts.q11_localization_parity import build_model, make_model_fn, MODEL_PATH  # noqa: E402
from scripts.e2_legacy_wrapper import legacy_invert  # noqa: E402
from create_dev_split import sha256_file  # noqa: E402

NOISE_LEVEL = 500
IMG = 256
SAMPLER = "ddim_anonymization_forward/null"


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_or_init_manifest(cache_dir: Path, ckpt_sha: str) -> dict:
    path = cache_dir / "manifest.json"
    if path.exists():
        man = json.loads(path.read_text())
        if man["checkpoint_sha256"] != ckpt_sha or man["noise_level"] != NOISE_LEVEL:
            raise SystemExit(f"cache config mismatch: manifest={man['checkpoint_sha256'][:8]}/"
                             f"{man['noise_level']} vs run={ckpt_sha[:8]}/{NOISE_LEVEL}")
        return man
    man = {"schema_version": 1, "checkpoint_sha256": ckpt_sha, "noise_level": NOISE_LEVEL,
           "sampler": SAMPLER, "image_size": IMG, "seed": 1911}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return man


def append_index(cache_dir: Path, rows: list[dict]) -> None:
    path = cache_dir / "index.csv"
    seen: set[str] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            seen = {r["file_sha256"] for r in csv.DictReader(handle)}
    new = [r for r in rows if r["file_sha256"] not in seen]
    if not new:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["sample_id", "patient_id", "pair_role", "file_sha256", "split"],
                           lineterminator="\n")
        if write_header:
            w.writeheader()
        w.writerows(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/t2bb_latent_cache")
    parser.add_argument("--max-patients", type=int, default=None, help="invert only the first N patients (prefix)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from scripts.e1_ddim_runner import preprocess
    with args.split.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_patients is not None:
        rows = rows[: args.max_patients * 2]   # splits are written in pair order

    ckpt_sha = sha256_file(args.checkpoint)
    load_or_init_manifest(args.cache_dir, ckpt_sha)
    latents_dir = args.cache_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    todo = [r for r in rows if not (latents_dir / f"{r['file_sha256']}.npy").exists()]
    print(f"{rel(args.split)}: {len(rows)} imgs, {len(rows) - len(todo)} cached, {len(todo)} to invert")

    if todo:
        model, diffusion, _ = build_model(args.device)
        model_fn = make_model_fn(model)
        t0 = time.time()
        for i, r in enumerate(todo, 1):
            x0 = preprocess(Path(r["local_path"]))[0].unsqueeze(0).to(args.device)
            z = legacy_invert(diffusion, model_fn, x0, NOISE_LEVEL)
            arr = z.reshape(IMG, IMG).cpu().numpy().astype(np.float32)
            np.save(latents_dir / f"{r['file_sha256']}.npy", arr)
            if i % 20 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"  {i}/{len(todo)}  {dt:.0f}s  ({dt/i:.2f}s/img)")

    append_index(args.cache_dir, [
        {"sample_id": r["sample_id"], "patient_id": r["patient_id"],
         "pair_role": r.get("pair_role", ""), "file_sha256": r["file_sha256"],
         "split": args.split.stem}
        for r in rows
    ])
    n_cached = len(list(latents_dir.glob("*.npy")))
    print(f"cache now holds {n_cached} latents at {rel(args.cache_dir)}")


if __name__ == "__main__":
    main()
