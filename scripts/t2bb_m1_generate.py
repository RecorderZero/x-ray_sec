#!/usr/bin/env python3
"""Generate the *actual released M1 PNGs* for the T2-BB linkage pilot.

This is the target-side half of the experiment.  It may read the target latent
cache and checkpoint, but it writes only attacker-visible grayscale PNGs plus a
provenance manifest.  The separate scorer (``t2bb_m1_frozen_linkage.py``) reads
only that exported directory and therefore cannot accidentally use latents,
keys, the checkpoint, or target-model activations.

Deployment cells are frozen to the current end-to-end comparison:
P/S0/S1 use the predecessor's range-discarding legacy PNG; S2 uses fixed norm,
per-image nonce, SOT R=1 without permutation, and the public q/255 portion of
the private-range PNG.  A fresh random S2 key is used for every pilot run and
is never persisted; only its SHA-256 fingerprint is recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# The predecessor instantiates Visdom clients at module import even though the
# inference path does not require a server.  Keep this batch experiment fully
# headless and prevent connection retries/noisy tracebacks.
import visdom as _visdom


class _NullVisdom:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


_visdom.Visdom = _NullVisdom

ROOT = Path(__file__).resolve().parents[1]
PAST = ROOT / "past" / "SourceCode"
sys.path[:0] = [str(ROOT), str(PAST)]

from scripts.e2_legacy_wrapper import apply_legacy_key, legacy_generate  # noqa: E402
from scripts.create_dev_split import stable_split_hash  # noqa: E402
from scripts.m1_storage import encode_private_range_png  # noqa: E402
from scripts.q11_localization_parity import MODEL_PATH, build_model, make_model_fn  # noqa: E402
from scripts.q8_pilot_runner import encode_legacy  # noqa: E402
from scripts.run_artifacts import sha256_file  # noqa: E402
from src.crypto.s2_fixed_norm import normalize_to_fixed_norm  # noqa: E402
from src.crypto.sot import transform  # noqa: E402

SEED = 1911
NOISE_LEVEL = 500
METHODS = ("P", "S0", "S1", "S2")
ROUNDS = 1
USE_PERMUTATION = False
IMAGE_SIZE = 256
SCHEMA_VERSION = 1


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _contained_path(base: Path, relative: str) -> Path:
    """Resolve a manifest path without permitting absolute paths or traversal."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RuntimeError(f"absolute artifact path is forbidden: {relative}")
    resolved_base = base.resolve()
    resolved = (base / candidate).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise RuntimeError(f"artifact path escapes export directory: {relative}")
    return resolved


def sanitize_v1_export(export_dir: Path, eval_map_path: Path) -> tuple[Path, Path]:
    """Migrate an already-completed v1 export to truth-isolated v2 in place.

    This exists so an interrupted/older generator run does not need expensive
    DDIM regeneration.  PNG bytes are only renamed; their hashes stay fixed.
    """
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "t2bb_m1_public/v1":
        raise RuntimeError("sanitize_v1_export requires a completed v1 export")
    if eval_map_path.exists():
        raise RuntimeError(f"refusing to overwrite private evaluation map: {eval_map_path}")
    split_path = ROOT / manifest["split"]
    split_rows = {row["sample_id"]: row for row in _load_rows(split_path, None)}
    latent_dir = ROOT / "artifacts" / "t2bb_latent_cache" / "latents"
    public, private, latent_hashes = [], [], []
    used: set[str] = set()
    for item in manifest["artifacts"]:
        while True:
            public_id = secrets.token_hex(16)
            relative = str(Path(item["method"]) / f"{public_id}.png")
            if relative not in used:
                used.add(relative); break
        old_path = _contained_path(export_dir, item["relative_path"])
        new_path = _contained_path(export_dir, relative)
        old_path.replace(new_path)
        public.append({key: item[key] for key in (
            "method", "storage", "png_sha256", "png_bytes", "nonce_hex"
        )} | {"public_id": public_id, "relative_path": relative})
        split_row = split_rows[item["sample_id"]]
        latent_sha = sha256_file(latent_dir / f"{split_row['file_sha256']}.npy")
        latent_hashes.append(latent_sha)
        private.append({
            "public_id": public_id, "relative_path": relative, "method": item["method"],
            "sample_id": item["sample_id"], "patient_id": item["patient_id"],
            "pair_id": item["pair_id"], "pair_role": item["pair_role"],
            "source_file_sha256": split_row["file_sha256"], "latent_sha256": latent_sha,
        })
    secrets.SystemRandom().shuffle(public)
    manifest["schema"] = "t2bb_m1_public/v2"
    manifest["schema_version"] = 2
    manifest["truth_isolation"] = "no sample/patient/pair/role labels; opaque randomized filenames and manifest order"
    manifest["consumed_latents_sha256"] = hashlib.sha256(
        "\n".join(sorted(latent_hashes)).encode("ascii")
    ).hexdigest()
    manifest["artifacts"] = public
    _atomic_json(manifest_path, manifest)
    evaluation = {
        "schema": "t2bb_m1_private_eval/v1",
        "source_export_manifest_sha256": sha256_file(manifest_path),
        "source_split_sha256": manifest["provenance"]["split_sha256"],
        "entries": private,
    }
    _atomic_json(eval_map_path, evaluation)
    progress_path = export_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["manifest_sha256"] = sha256_file(manifest_path)
        progress["truth_isolated"] = True
        _atomic_json(progress_path, progress)
    return manifest_path, eval_map_path


def _load_rows(split: Path, limit_patients: int | None) -> list[dict[str, str]]:
    with split.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) % 2:
        raise RuntimeError("T2-BB split must contain complete gallery/query pairs")
    by_pair: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    for pair, members in by_pair.items():
        if len(members) != 2 or {r["pair_role"] for r in members} != {"gallery", "query"}:
            raise RuntimeError(f"invalid pair {pair}: expected one gallery and one query")
        if len({r["patient_id"] for r in members}) != 1:
            raise RuntimeError(f"invalid pair {pair}: patient mismatch")
        if len({r["study_id"] for r in members}) != 2:
            raise RuntimeError(f"invalid pair {pair}: studies are not distinct")
    if limit_patients is not None:
        keep = set(list(by_pair)[:limit_patients])
        rows = [row for row in rows if row["pair_id"] in keep]
    return rows


def _validate_inputs(split: Path, rows: list[dict[str, str]], cache_dir: Path, checkpoint: Path) -> dict[str, Any]:
    split_file_sha = sha256_file(split)
    split_sha = stable_split_hash(rows)
    declared = {row["split_sha256"] for row in rows}
    if declared != {split_sha}:
        raise RuntimeError(f"split SHA mismatch: file={split_sha}, rows={declared}")
    split_manifest_path = split.with_name(f"{split.stem}_manifest.json")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest["split_sha256"] != split_sha:
        raise RuntimeError("split manifest SHA mismatch")

    for row in rows:
        image_path = Path(row["local_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if sha256_file(image_path) != row["file_sha256"]:
            raise RuntimeError(f"source image SHA mismatch: {row['sample_id']}")
        latent_path = cache_dir / "latents" / f"{row['file_sha256']}.npy"
        if not latent_path.exists():
            raise RuntimeError(f"missing latent cache: {row['sample_id']} ({latent_path})")

    cache_manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint)
    if cache_manifest["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("latent cache/checkpoint SHA mismatch")
    if cache_manifest["noise_level"] != NOISE_LEVEL or cache_manifest["image_size"] != IMAGE_SIZE:
        raise RuntimeError("latent cache sampling configuration mismatch")
    return {
        "split_sha256": split_sha,
        "split_file_sha256": split_file_sha,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "checkpoint_sha256": checkpoint_sha,
        "latent_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
    }


def _protected_latent(z: torch.Tensor, method: str, key: bytes, nonce: bytes) -> torch.Tensor:
    if method in ("P", "S0", "S1"):
        return apply_legacy_key(z, PAST, method, "anonymize")
    normalized, _original_norm = normalize_to_fixed_norm(z)
    return transform(normalized, key, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION)


def _public_png(anonymous: torch.Tensor, method: str) -> tuple[bytes, str]:
    if method in ("P", "S0", "S1"):
        payload, _ = encode_legacy(anonymous)
        return payload, "legacy_png/range_discarded"
    payload, _private_record = encode_private_range_png(
        anonymous[0, 0].detach().float().cpu().numpy()
    )
    return payload, "private_range_png/public_q_only"


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--split", type=Path, default=ROOT / "splits/t2bb_inner_v1.csv")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/t2bb_latent_cache")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/t2bb_m1_public_v1")
    parser.add_argument("--eval-map", type=Path, default=None,
                        help="private truth map; defaults beside (not inside) the public export")
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit_patients is not None and args.limit_patients <= 0:
        raise SystemExit("--limit-patients must be positive")
    if len(args.methods) != len(set(args.methods)):
        raise SystemExit("--methods must not contain duplicates")
    eval_map_path = args.eval_map or args.out_dir.with_name(f"{args.out_dir.name}_private_eval.json")
    resolved_output = args.out_dir.resolve()
    resolved_eval = eval_map_path.resolve()
    if resolved_eval == resolved_output or resolved_output in resolved_eval.parents:
        raise SystemExit("--eval-map must be outside the attacker-visible --out-dir")

    manifest_path = args.out_dir / "manifest.json"
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"refusing non-empty output directory without --force: {args.out_dir}")
    if args.force and args.out_dir.exists():
        # Remove only files created by this script, never an arbitrary directory tree.
        old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if old and old.get("schema") in {"t2bb_m1_public/v1", "t2bb_m1_public/v2"}:
            for item in old.get("artifacts", []):
                path = _contained_path(args.out_dir, item["relative_path"])
                if path.is_file():
                    path.unlink()
            for method in METHODS:
                directory = args.out_dir / method
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            manifest_path.unlink(missing_ok=True)
            (args.out_dir / "progress.json").unlink(missing_ok=True)
            if any(args.out_dir.iterdir()):
                raise SystemExit("--force refused: unrecognized files remain in output directory")
        elif any(args.out_dir.iterdir()):
            raise SystemExit("--force refused: directory is not a recognized T2-BB M1 export")
    if eval_map_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite private evaluation map without --force: {eval_map_path}")
    if args.force:
        eval_map_path.unlink(missing_ok=True)

    # The declared split hash covers the complete frozen split.  Validate that
    # complete object first, then take a prefix only for a smoke run.
    full_rows = _load_rows(args.split, None)
    provenance = _validate_inputs(args.split, full_rows, args.cache_dir, args.checkpoint)
    rows = _load_rows(args.split, args.limit_patients)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        (args.out_dir / method).mkdir(exist_ok=True)

    torch.manual_seed(SEED)
    key = secrets.token_bytes(32)
    nonces: set[bytes] = set()
    started = time.time()
    total = len(rows) * len(args.methods)
    run_state = {"completed": 0}

    def record_uncaught(exc_type, exc_value, traceback):
        _atomic_json(args.out_dir / "progress.json", {
            "state": "failed", "completed": run_state["completed"], "total": total,
            "elapsed_seconds": time.time() - started,
            "error_type": exc_type.__name__, "error": str(exc_value)[:1000],
        })
        sys.__excepthook__(exc_type, exc_value, traceback)

    sys.excepthook = record_uncaught
    model, diffusion, _ = build_model(args.device, args.checkpoint)
    model_fn = make_model_fn(model)
    artifacts: list[dict[str, Any]] = []
    evaluation_entries: list[dict[str, Any]] = []
    consumed_latent_hashes: list[str] = []
    public_paths: set[str] = set()
    completed = 0

    for row in rows:
        latent_path = args.cache_dir / "latents" / f"{row['file_sha256']}.npy"
        latent_sha = sha256_file(latent_path)
        consumed_latent_hashes.append(latent_sha)
        array = np.load(latent_path)
        if array.shape != (IMAGE_SIZE, IMAGE_SIZE) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise RuntimeError(f"invalid latent cache array: {row['sample_id']} {array.shape} {array.dtype}")
        z = torch.from_numpy(array).reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(args.device)
        for method in args.methods:
            while True:
                nonce = secrets.token_bytes(16)
                if nonce not in nonces:
                    nonces.add(nonce)
                    break
            protected = _protected_latent(z, method, key, nonce)
            anonymous = legacy_generate(diffusion, model_fn, protected, NOISE_LEVEL)
            if not bool(torch.isfinite(anonymous).all()):
                raise RuntimeError(f"non-finite generated image: {method}/{row['sample_id']}")
            payload, storage = _public_png(anonymous, method)
            while True:
                public_id = secrets.token_hex(16)
                relative = Path(method) / f"{public_id}.png"
                if str(relative) not in public_paths:
                    public_paths.add(str(relative)); break
            output = args.out_dir / relative
            output.write_bytes(payload)
            artifacts.append({
                "public_id": public_id, "method": method, "storage": storage, "relative_path": str(relative),
                "png_sha256": hashlib.sha256(payload).hexdigest(), "png_bytes": len(payload),
                # A nonce is public metadata in the deployed construction.  It is
                # recorded for auditability, while the random master key is not.
                "nonce_hex": nonce.hex() if method == "S2" else "not_applicable",
            })
            evaluation_entries.append({
                "public_id": public_id, "relative_path": str(relative), "method": method,
                "sample_id": row["sample_id"], "patient_id": row["patient_id"],
                "pair_id": row["pair_id"], "pair_role": row["pair_role"],
                "source_file_sha256": row["file_sha256"], "latent_sha256": latent_sha,
            })
            completed += 1
            run_state["completed"] = completed
            progress = {
                "state": "running", "completed": completed, "total": total,
                "last_artifact": str(relative), "elapsed_seconds": time.time() - started,
            }
            _atomic_json(args.out_dir / "progress.json", progress)
            print(f"[{completed:03d}/{total:03d}] {relative} ({time.time() - started:.1f}s)", flush=True)

    manifest = {
        "schema": "t2bb_m1_public/v2", "schema_version": 2,
        "evidence_role": "pilot" if args.limit_patients is None else "smoke",
        "threat_model": "T2-BB/no-model/no-key/no-plaintext/anonymous-PNG-only",
        "split": (str(args.split.resolve().relative_to(ROOT.resolve()))
                  if args.split.resolve().is_relative_to(ROOT.resolve()) else str(args.split.resolve())),
        "patients": len(rows) // 2,
        "images": len(rows), "methods": list(args.methods), "seed": SEED,
        "noise_level": NOISE_LEVEL, "rounds": ROUNDS, "use_permutation": USE_PERMUTATION,
        "s2_key_policy": "fresh random 256-bit per run; raw key not persisted",
        "key_repetitions": 1, "single_key_exploratory": True,
        "s2_key_sha256": hashlib.sha256(key).hexdigest(),
        "s2_nonce_policy": "fresh unique random 128-bit per image; public",
        "truth_isolation": "no sample/patient/pair/role labels; opaque randomized filenames and manifest order",
        "consumed_latents_sha256": hashlib.sha256(
            "\n".join(sorted(consumed_latent_hashes)).encode("ascii")
        ).hexdigest(),
        "storage_policy": {"P": "legacy", "S0": "legacy", "S1": "legacy", "S2": "private_range_public_q"},
        "provenance": provenance, "elapsed_seconds": time.time() - started,
        "artifacts": artifacts,
    }
    secrets.SystemRandom().shuffle(manifest["artifacts"])
    _atomic_json(manifest_path, manifest)
    _atomic_json(eval_map_path, {
        "schema": "t2bb_m1_private_eval/v1",
        "source_export_manifest_sha256": sha256_file(manifest_path),
        "source_split_sha256": provenance["split_sha256"],
        "entries": evaluation_entries,
    })
    _atomic_json(args.out_dir / "progress.json", {
        "state": "completed", "completed": total, "total": total,
        "manifest_sha256": sha256_file(manifest_path), "elapsed_seconds": time.time() - started,
    })
    print(f"completed {total} public PNGs in {time.time() - started:.1f}s; "
          f"manifest={manifest_path}; private_eval={eval_map_path}")


if __name__ == "__main__":
    main()
