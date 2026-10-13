#!/usr/bin/env python3
"""D7 CPU security evidence: T1, cross-nonce, geometry, probes, repeat, tamper.

This script consumes the frozen security_v1 latent cache and never loads the
diffusion checkpoint.  T2-WB image generation/inversion is deliberately kept
in ``d7_t2wb_inversion.py`` because it is a long GPU job.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.d7_metrics import SCORERS, feature, rank_metrics, ranks_l2
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import sha256_file
from src.crypto.errors import AuthenticationError
from src.crypto.nonce import NonceRegistry
from src.crypto.s2 import open_latent, seal_latent
from src.crypto.sot import inverse_transform, transform, transform_vectors


SEED = 1911
ROUNDS = 1
USE_PERMUTATION = False
FIXTURE_KEY = hashlib.sha256(b"cxr-s2/D7/public-reproducibility-fixture/v1").digest()


def nonce_for(label: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/D7/{label}".encode()).digest()[:16]


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_cache(meta_path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    path = Path(meta["cache_file"])
    if sha256_file(path) != meta["cache_file_sha256"]:
        raise RuntimeError("latent cache hash mismatch")
    latents = np.load(path).astype(np.float32, copy=False)
    if latents.shape != (200, 1, 256, 256) or not np.isfinite(latents).all():
        raise RuntimeError(f"unexpected formal cache: {latents.shape} {latents.dtype}")
    ids = list(meta["sample_ids"])
    if len(ids) != len(latents):
        raise RuntimeError("sample ID/cache length mismatch")
    return latents, ids, meta


def protect_formal(latents: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    output = []
    for sample_id, latent in zip(sample_ids, latents):
        protected = transform(
            torch.from_numpy(latent), FIXTURE_KEY, nonce_for(f"formal/{sample_id}"),
            rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        )
        output.append(protected.numpy())
    return np.stack(output).astype(np.float32, copy=False)


def linkage_rows(latents: np.ndarray, protected: np.ndarray) -> list[dict[str, Any]]:
    truth = np.arange(len(latents))
    shuffled = np.random.default_rng(SEED).permutation(len(latents))
    rows = []
    for scorer in SCORERS:
        gallery = feature(latents, scorer)
        query = feature(protected, scorer)
        attack = rank_metrics(ranks_l2(query, gallery, truth), len(latents))
        negative = rank_metrics(ranks_l2(query, gallery, shuffled), len(latents))
        rows.append({
            "task": "D7.2" if scorer != "norm" else "D7.3", "threat_model": "T1",
            "method": "S2a/S2", "storage": "exact_latent", "scorer": scorer,
            "n": len(latents), "gallery_size": len(latents),
            "top1": attack["top1"], "top1_ci_low": attack["top1_ci95_wilson"][0],
            "top1_ci_high": attack["top1_ci95_wilson"][1], "top5": attack["top5"],
            "mAP": attack["mAP"], "median_rank": attack["median_rank"],
            "negative_top1": negative["top1"], "random_top1": attack["random_top1"],
            "tie_break": "pessimistic", "rounds": ROUNDS, "permutation": USE_PERMUTATION,
        })
    return rows


def cross_nonce_rows() -> list[dict[str, Any]]:
    dimension = 1024
    generator = torch.Generator().manual_seed(SEED)
    heldout = torch.randn(32, dimension, dtype=torch.float64, generator=generator)
    nonce_a, nonce_b = nonce_for("cross/A"), nonce_for("cross/B")
    identity = torch.eye(dimension, dtype=torch.float64)
    # CPA: chosen basis rows. KPA: a full-rank orthonormal set of known random rows.
    known_random = torch.linalg.qr(
        torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    ).Q
    rows = []
    for attack, known in (("CPA_basis", identity), ("KPA_full_rank", known_random)):
        cipher_a = transform_vectors(
            known, FIXTURE_KEY, nonce_a, rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        )
        # Row-vector convention: C = Z Q^T, so Q^T = Z^T C for orthonormal Z.
        learned_q_t = known.T @ cipher_a
        for target_name, target_nonce in (("same_nonce", nonce_a), ("cross_nonce", nonce_b)):
            target_cipher = transform_vectors(
                heldout, FIXTURE_KEY, target_nonce, rounds=ROUNDS,
                use_permutation=USE_PERMUTATION,
            )
            recovered = target_cipher @ learned_q_t.T
            error = recovered - heldout
            cos = torch.nn.functional.cosine_similarity(recovered, heldout, dim=1)
            rows.append({
                "attack": attack, "known_pairs": dimension, "dimension": dimension,
                "target": target_name, "samples": len(heldout),
                "rmse": float(torch.sqrt(torch.mean(error.square()))),
                "max_abs": float(torch.max(torch.abs(error))),
                "cosine_mean": float(cos.mean()),
                "positive_control": target_name == "same_nonce",
                "rounds": ROUNDS, "permutation": USE_PERMUTATION,
            })
    return rows


def pairwise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    gram = x @ x.T
    norms = np.linalg.norm(x, axis=1)
    cosine = gram / np.maximum(norms[:, None] * norms[None, :], 1e-30)
    distance = np.sqrt(np.maximum(norms[:, None] ** 2 + norms[None, :] ** 2 - 2 * gram, 0.0))
    return cosine, distance


def geometry_rows(latents: np.ndarray, per_image: np.ndarray) -> list[dict[str, Any]]:
    flat = latents.reshape(len(latents), -1)
    fixed = transform_vectors(
        torch.from_numpy(flat), FIXTURE_KEY, nonce_for("geometry/fixed"),
        rounds=ROUNDS, use_permutation=USE_PERMUTATION,
    ).numpy()
    base_cos, base_dist = pairwise(flat)
    truth = np.arange(len(flat))
    shuffled = np.random.default_rng(SEED).permutation(len(flat))
    rows = []
    for mode, protected in (("fixed_Q", fixed), ("per_image_Q", per_image.reshape(len(flat), -1))):
        cos, dist = pairwise(protected)
        # Geometry signature excludes self distance; sorting avoids identity leakage from column order.
        qsig = np.sort(dist, axis=1)[:, 1:]
        gsig = np.sort(base_dist, axis=1)[:, 1:]
        attack = rank_metrics(ranks_l2(qsig, gsig, truth), len(flat))
        negative = rank_metrics(ranks_l2(qsig, gsig, shuffled), len(flat))
        rows.append({
            "mode": mode, "n": len(flat),
            "pairwise_cosine_max_abs_error": float(np.max(np.abs(cos - base_cos))),
            "pairwise_distance_max_abs_error": float(np.max(np.abs(dist - base_dist))),
            "distance_signature_top1": attack["top1"],
            "distance_signature_top5": attack["top5"],
            "negative_top1": negative["top1"], "random_top1": 1 / len(flat),
            "rounds": ROUNDS, "permutation": USE_PERMUTATION,
        })
    return rows


def structured_probe_rows() -> list[dict[str, Any]]:
    dimension = 4096
    probes: dict[str, torch.Tensor] = {
        "basis_0": torch.nn.functional.one_hot(torch.tensor(0), dimension).double(),
        "basis_123": torch.nn.functional.one_hot(torch.tensor(123), dimension).double(),
        "constant": torch.ones(dimension, dtype=torch.float64),
        "sparse_16": torch.cat((torch.ones(16, dtype=torch.float64), torch.zeros(dimension - 16, dtype=torch.float64))),
    }
    rows = []
    for name, probe in probes.items():
        protected = transform(
            probe, FIXTURE_KEY, nonce_for(f"probe/{name}"), rounds=ROUNDS,
            use_permutation=USE_PERMUTATION,
        )
        rows.append({
            "probe": name, "dimension": dimension,
            "input_nonzero_fraction": float(torch.mean((probe != 0).double())),
            "output_nonzero_fraction": float(torch.mean((protected != 0).double())),
            "output_mean": float(protected.mean()), "output_variance": float(protected.var(unbiased=False)),
            "output_max_abs": float(torch.max(torch.abs(protected))),
            "norm_ratio": float(torch.linalg.vector_norm(protected) / torch.linalg.vector_norm(probe)),
            "interpretation": "diagnostic only; not a security proof",
        })
    return rows


def repeat_rows(latent: np.ndarray) -> list[dict[str, Any]]:
    source = torch.from_numpy(latent)
    protected, rows = [], []
    for index in range(100):
        nonce = nonce_for(f"repeat/{index}")
        ciphertext = transform(
            source, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION,
        ).numpy()
        protected.append(ciphertext.reshape(-1))
        rows.append({
            "repeat": index, "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
            "payload_sha256": hashlib.sha256(np.ascontiguousarray(ciphertext).tobytes()).hexdigest(),
        })
    matrix = np.stack(protected).astype(np.float64)
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine = normalized @ normalized.T
    offdiag = cosine[~np.eye(len(cosine), dtype=bool)]
    unique_nonces = len({row["nonce_sha256"] for row in rows})
    unique_payloads = len({row["payload_sha256"] for row in rows})
    for row in rows:
        row.update({
            "unique_nonces_total": unique_nonces, "unique_payloads_total": unique_payloads,
            "offdiag_cosine_mean": float(offdiag.mean()), "offdiag_cosine_max": float(offdiag.max()),
            "all_unique": unique_nonces == unique_payloads == 100,
        })
    return rows


def _serialize(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def tamper_rows() -> list[dict[str, Any]]:
    source = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) / 64
    nonce = nonce_for("tamper/base")
    aad = {
        "study_instance_uid": "research:tamper", "sop_instance_uid": "research:tamper:0",
        "model_sha256": "ab" * 32, "sampler_version": "legacy-ddim-anonymization/v1",
    }
    with tempfile.TemporaryDirectory(prefix="d7_nonce_") as temporary:
        with NonceRegistry(Path(temporary) / "nonces.sqlite3") as registry:
            sealed = seal_latent(
                source, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION,
                aad=aad, registry=registry,
            )
    base = json.loads(sealed)
    cases: list[tuple[str, dict[str, Any]]] = []
    for name in ("tag", "nonce", "ciphertext", "version", "parameters", "aad", "storage"):
        changed = json.loads(json.dumps(base))
        if name == "tag":
            changed["tag"] = ("00" if changed["tag"][:2] != "00" else "01") + changed["tag"][2:]
        elif name == "nonce":
            changed["nonce"] = ("00" if changed["nonce"][:2] != "00" else "01") + changed["nonce"][2:]
        elif name == "ciphertext":
            raw = bytearray(base64.b64decode(changed["ciphertext"])); raw[0] ^= 1
            changed["ciphertext"] = base64.b64encode(raw).decode()
        elif name == "version":
            changed["version"] = "cxr-auth-container/v999"
        elif name == "parameters":
            changed["parameters"]["rounds"] = 2
        elif name == "aad":
            changed["aad"]["sop_instance_uid"] = "research:tamper:changed"
        elif name == "storage":
            changed["storage"]["payload_sha256"] = "00" * 32
        cases.append((name, changed))
    rows = []
    for name, document in cases:
        rejected = False
        try:
            open_latent(_serialize(document), FIXTURE_KEY)
        except AuthenticationError:
            rejected = True
        rows.append({"method": "S2", "case": name, "tampered": True, "rejected": rejected})
    protected = transform(source, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION)
    tampered = protected.clone(); tampered.reshape(-1)[0] += 0.25
    recovered = inverse_transform(
        tampered, FIXTURE_KEY, nonce, rounds=ROUNDS, use_permutation=USE_PERMUTATION,
    )
    rows.append({
        "method": "S2a", "case": "latent_value", "tampered": True, "rejected": False,
        "output_max_abs_change": float(torch.max(torch.abs(recovered - source))),
    })
    if not all(row["rejected"] for row in rows if row["method"] == "S2"):
        raise RuntimeError("S2 tamper suite did not reject every case")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-meta", type=Path, default=Path("results/E2.3_DONE.json"))
    parser.add_argument("--attack-table", type=Path, default=Path("paper_assets/tables/table_attack_original.csv"))
    parser.add_argument("--norm-table", type=Path, default=Path("paper_assets/tables/table_norm_leakage.csv"))
    parser.add_argument("--cross-nonce-table", type=Path, default=Path("paper_assets/tables/table_cross_nonce.csv"))
    parser.add_argument("--geometry-table", type=Path, default=Path("paper_assets/tables/table_geometry.csv"))
    parser.add_argument("--probe-table", type=Path, default=Path("paper_assets/tables/table_structured_probe.csv"))
    parser.add_argument("--repeat-table", type=Path, default=Path("paper_assets/tables/table_repeat.csv"))
    parser.add_argument("--tamper-table", type=Path, default=Path("paper_assets/tables/table_tamper.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/D7_cpu_security.json"))
    add_force_flag(parser)
    args = parser.parse_args()
    outputs = [args.attack_table, args.norm_table, args.cross_nonce_table, args.geometry_table,
               args.probe_table, args.repeat_table, args.tamper_table, args.json_output]
    refuse_overwrite(outputs, args.force)
    latents, sample_ids, cache_meta = load_cache(args.cache_meta)
    protected = protect_formal(latents, sample_ids)
    linkage = linkage_rows(latents, protected)
    attack = [row for row in linkage if row["task"] == "D7.2"]
    norm = [row for row in linkage if row["task"] == "D7.3"]
    cross_nonce = cross_nonce_rows()
    geometry = geometry_rows(latents, protected)
    probes = structured_probe_rows()
    repeat = repeat_rows(latents[0])
    tamper = tamper_rows()
    write_csv(args.attack_table, attack); write_csv(args.norm_table, norm)
    write_csv(args.cross_nonce_table, cross_nonce); write_csv(args.geometry_table, geometry)
    write_csv(args.probe_table, probes); write_csv(args.repeat_table, repeat)
    write_csv(args.tamper_table, tamper)
    payload = {
        "schema_version": 1, "task_id": "D7.CPU", "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)), "cache_run_id": cache_meta["run_id"],
        "cache_sha256": cache_meta["cache_file_sha256"], "split_sha256": cache_meta["split_sha256"],
        "sample_count": len(latents), "seed": SEED, "rounds": ROUNDS,
        "permutation": USE_PERMUTATION,
        "fixture_key_sha256": hashlib.sha256(FIXTURE_KEY).hexdigest(),
        "raw_key_material_recorded": False,
        "tables": {name: str(path) for name, path in {
            "D7.2": args.attack_table, "D7.3": args.norm_table, "D7.4": args.cross_nonce_table,
            "D7.5": args.geometry_table, "D7.6": args.probe_table, "D7.7": args.repeat_table,
            "D7.8": args.tamper_table,
        }.items()},
        "disclaimer": "attack outcomes are measurements, not engineering pass gates; probes are not a security proof",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote D7 CPU evidence for {len(latents)} formal samples")


if __name__ == "__main__":
    main()
