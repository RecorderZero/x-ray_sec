#!/usr/bin/env python3
"""D6.1-D6.6 CPU evidence for the structured orthogonal transform.

The deterministic key/nonce material in this script is a published experiment
fixture, never a deployment secret.  It makes correctness and ablation results
independently reproducible; none of the diagnostics is presented as a general
confidentiality proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy import stats

from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.run_artifacts import sha256_file
from src.crypto.prng import permutation, sign_vector
from src.crypto.sot import fwht, inverse_transform, transform, transform_vectors


SEED = 1911
ROUNDS = (1, 2, 3, 4)
PERMUTATIONS = (False, True)
FIXTURE_KEY = hashlib.sha256(b"cxr-s2/D6/public-reproducibility-fixture/v1").digest()
DISCLAIMER = "Structural diagnostics and p-values are diagnostics, not a security proof."


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def nonce_for(label: str) -> bytes:
    return hashlib.sha256(f"cxr-s2/D6/{label}".encode()).digest()[:16]


def describe(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "variance": float(flat.var()),
        "skew": float(stats.skew(flat)),
        "excess_kurtosis": float(stats.kurtosis(flat)),
    }


def topk_accuracy(queries: np.ndarray, gallery: np.ndarray, k: int) -> float:
    q = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    g = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-30)
    scores = q @ g.T
    order = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([index in order[index] for index in range(len(order))]))


def norm_top1(queries: np.ndarray, gallery: np.ndarray) -> float:
    distances = np.abs(queries[:, None] - gallery[None, :])
    return float(np.mean(np.argmin(distances, axis=1) == np.arange(len(queries))))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-meta", type=Path, default=Path("results/E2.3_dev_v1.1_cache.json"))
    parser.add_argument("--d61", type=Path, default=Path("results/D6.1.json"))
    parser.add_argument("--d62", type=Path, default=Path("results/D6.2.json"))
    parser.add_argument("--d63", type=Path, default=Path("results/D6.3.json"))
    parser.add_argument("--d63-figure", type=Path, default=Path("image/D6.3_gaussian_distribution.png"))
    parser.add_argument("--d64", type=Path, default=Path("results/D6.4.csv"))
    parser.add_argument(
        "--d65", type=Path, default=Path("paper_assets/tables/table_rounds_permutation.csv")
    )
    parser.add_argument("--d66", type=Path, default=Path("paper_assets/tables/table_structure.csv"))
    add_force_flag(parser)
    args = parser.parse_args()
    outputs = [args.d61, args.d62, args.d63, args.d63_figure, args.d64, args.d65, args.d66]
    refuse_overwrite(outputs, args.force)

    meta = json.loads(args.cache_meta.read_text(encoding="utf-8"))
    cache_path = Path(meta["cache_file"])
    if sha256_file(cache_path) != meta["cache_file_sha256"]:
        raise RuntimeError("true-latent cache hash mismatch")
    latents = np.load(cache_path)
    if tuple(latents.shape[1:]) != (1, 256, 256) or latents.dtype != np.float32:
        raise RuntimeError(f"unexpected latent cache contract: {latents.shape} {latents.dtype}")
    sample_ids = list(meta["sample_ids"])
    if len(sample_ids) != len(latents):
        raise RuntimeError("cache sample IDs and rows differ")

    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git_value("status", "--porcelain=v1", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seed": SEED,
        "key_mode": "public deterministic reproducibility fixture; not a deployment secret",
        "key_sha256": hashlib.sha256(FIXTURE_KEY).hexdigest(),
        "nonce_mode": "unique deterministic per sample/config fixture",
    }

    generator = torch.Generator().manual_seed(SEED)
    base64 = torch.randn(65536, dtype=torch.float64, generator=generator)
    base32 = base64.float()
    d61_rows = []
    for use_permutation in PERMUTATIONS:
        for rounds in ROUNDS:
            nonce = nonce_for(f"D6.1/R{rounds}/P{int(use_permutation)}")
            protected64 = transform(
                base64, FIXTURE_KEY, nonce, rounds=rounds, use_permutation=use_permutation,
            )
            restored64 = inverse_transform(
                protected64, FIXTURE_KEY, nonce, rounds=rounds,
                use_permutation=use_permutation,
            )
            protected32 = transform(
                base32, FIXTURE_KEY, nonce, rounds=rounds, use_permutation=use_permutation,
            )
            restored32 = inverse_transform(
                protected32, FIXTURE_KEY, nonce, rounds=rounds,
                use_permutation=use_permutation,
            )
            d61_rows.append({
                "rounds": rounds,
                "permutation": use_permutation,
                "dimension": 65536,
                "float64_max_abs": float(torch.max(torch.abs(restored64 - base64))),
                "float64_norm_relative_error": float(
                    torch.abs(torch.linalg.vector_norm(protected64) - torch.linalg.vector_norm(base64))
                    / torch.linalg.vector_norm(base64)
                ),
                "float32_max_abs": float(torch.max(torch.abs(restored32 - base32))),
                "float32_bit_exact_rate": float(torch.mean((restored32 == base32).float())),
            })
    d61_pass = all(row["float64_max_abs"] <= 1e-12 for row in d61_rows)

    dimension = 1024
    identity = torch.eye(dimension, dtype=torch.float64)
    matrix_nonce = nonce_for("D6.2/explicit-Q")
    q1 = transform_vectors(
        identity, FIXTURE_KEY, matrix_nonce, rounds=1, use_permutation=True,
    ).T
    q2 = transform_vectors(
        identity, FIXTURE_KEY, matrix_nonce, rounds=2, use_permutation=True,
    ).T
    hadamard = fwht(identity).T
    pre = torch.from_numpy(sign_vector(FIXTURE_KEY, matrix_nonce, 1, "sign-pre", dimension)).double()
    post = torch.from_numpy(sign_vector(FIXTURE_KEY, matrix_nonce, 1, "sign-post", dimension)).double()
    indices = torch.from_numpy(permutation(FIXTURE_KEY, matrix_nonce, 1, dimension))
    permutation_matrix = identity[indices]
    q_round_two = torch.diag(post) @ permutation_matrix @ hadamard @ torch.diag(pre)
    d62 = {
        **common,
        "task_id": "D6.2",
        "dimension": dimension,
        "rounds": 2,
        "permutation": True,
        "orthogonality_max_abs": float(torch.max(torch.abs(q2.T @ q2 - identity))),
        "norm_error_spectral": float(abs(torch.linalg.matrix_norm(q2, ord=2) - 1.0)),
        "round_formula_max_abs": float(torch.max(torch.abs(q2 - q_round_two @ q1))),
        "formula": "Q_total = Q_2 Q_1; generalized implementation loops Q_R ... Q_1",
    }
    d62["passed"] = max(
        d62["orthogonality_max_abs"], d62["norm_error_spectral"], d62["round_formula_max_abs"]
    ) <= 1e-12

    synthetic = torch.randn(128, 4096, dtype=torch.float64, generator=generator)
    synthetic_after = transform_vectors(
        synthetic, FIXTURE_KEY, nonce_for("D6.3/gaussian"), rounds=2, use_permutation=True,
    )
    before_np, after_np = synthetic.numpy(), synthetic_after.numpy()
    # Standardize against the declared N(0,1), and report the full test rather than
    # treating a thresholded p-value as a security decision.
    before_ks = stats.kstest(before_np.reshape(-1), "norm")
    after_ks = stats.kstest(after_np.reshape(-1), "norm")
    d63 = {
        **common,
        "task_id": "D6.3",
        "vectors": 128,
        "dimension": 4096,
        "before": {**describe(before_np), "ks_statistic": float(before_ks.statistic), "ks_pvalue": float(before_ks.pvalue)},
        "after": {**describe(after_np), "ks_statistic": float(after_ks.statistic), "ks_pvalue": float(after_ks.pvalue)},
        "interpretation": DISCLAIMER,
    }
    quantiles = np.linspace(0.001, 0.999, 999)
    theoretical = stats.norm.ppf(quantiles)
    before_q = np.quantile(before_np, quantiles)
    after_q = np.quantile(after_np, quantiles)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(before_np.reshape(-1), bins=100, density=True, alpha=0.5, label="z")
    axes[0].hist(after_np.reshape(-1), bins=100, density=True, alpha=0.5, label="Qz")
    xline = np.linspace(-4, 4, 300)
    axes[0].plot(xline, stats.norm.pdf(xline), color="black", linewidth=1, label="N(0,1)")
    axes[0].set(title="Gaussian histogram", xlabel="value", ylabel="density"); axes[0].legend()
    axes[1].plot(theoretical, before_q, label="z", linewidth=1)
    axes[1].plot(theoretical, after_q, label="Qz", linewidth=1)
    axes[1].plot([-3.2, 3.2], [-3.2, 3.2], color="black", linewidth=1, linestyle="--")
    axes[1].set(title="Normal Q-Q", xlabel="theoretical quantile", ylabel="observed quantile"); axes[1].legend()
    fig.suptitle("D6.3 distribution diagnostic (not a security proof)")
    fig.tight_layout(); args.d63_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.d63_figure, dpi=180); plt.close(fig)

    d64_rows: list[dict[str, Any]] = []
    d65_rows: list[dict[str, Any]] = []
    original_flat = latents.reshape(len(latents), -1)
    for use_permutation in PERMUTATIONS:
        for rounds in ROUNDS:
            protected_rows, restored_rows, runtimes = [], [], []
            for sample_id, latent_np in zip(sample_ids, latents):
                latent = torch.from_numpy(latent_np)
                nonce = nonce_for(f"D6.true/{sample_id}/R{rounds}/P{int(use_permutation)}")
                started = time.perf_counter()
                protected = transform(
                    latent, FIXTURE_KEY, nonce, rounds=rounds,
                    use_permutation=use_permutation,
                )
                runtimes.append(time.perf_counter() - started)
                restored = inverse_transform(
                    protected, FIXTURE_KEY, nonce, rounds=rounds,
                    use_permutation=use_permutation,
                )
                protected_np = protected.numpy()
                restored_np = restored.numpy()
                protected_rows.append(protected_np.reshape(-1))
                restored_rows.append(restored_np.reshape(-1))
                source_stats, protected_stats = describe(latent_np), describe(protected_np)
                d64_rows.append({
                    "sample_id": sample_id,
                    "rounds": rounds,
                    "permutation": use_permutation,
                    **{f"z_{name}": value for name, value in source_stats.items()},
                    **{f"qz_{name}": value for name, value in protected_stats.items()},
                    "mean_shift": protected_stats["mean"] - source_stats["mean"],
                    "variance_shift": protected_stats["variance"] - source_stats["variance"],
                    "skew_shift": protected_stats["skew"] - source_stats["skew"],
                    "excess_kurtosis_shift": protected_stats["excess_kurtosis"] - source_stats["excess_kurtosis"],
                    "norm_ratio": float(np.linalg.norm(protected_np) / np.linalg.norm(latent_np)),
                    "same_vector_cosine": float(
                        np.dot(latent_np.reshape(-1), protected_np.reshape(-1))
                        / (np.linalg.norm(latent_np) * np.linalg.norm(protected_np))
                    ),
                    "generation_quality_status": "measured separately by D6.7 dev integration",
                })
            protected_flat = np.stack(protected_rows)
            restored_flat = np.stack(restored_rows)
            sorted_original = np.sort(np.abs(original_flat), axis=1)
            sorted_protected = np.sort(np.abs(protected_flat), axis=1)
            norms_original = np.linalg.norm(original_flat, axis=1)[:, None]
            norms_protected = np.linalg.norm(protected_flat, axis=1)[:, None]
            d65_rows.append({
                "rounds": rounds,
                "permutation": use_permutation,
                "n": len(latents),
                "original_abs_top1": topk_accuracy(np.abs(protected_flat), np.abs(original_flat), 1),
                "original_abs_top5": topk_accuracy(np.abs(protected_flat), np.abs(original_flat), 5),
                "sorted_abs_top1": topk_accuracy(sorted_protected, sorted_original, 1),
                "sorted_abs_top5": topk_accuracy(sorted_protected, sorted_original, 5),
                "norm_top1": norm_top1(norms_protected[:, 0], norms_original[:, 0]),
                "runtime_ms_mean": 1000 * float(np.mean(runtimes)),
                "runtime_ms_median": 1000 * float(np.median(runtimes)),
                "float32_roundtrip_max_abs": float(np.max(np.abs(restored_flat - original_flat))),
                "float32_bit_exact_rate": float(np.mean(restored_flat == original_flat)),
                "note": "Attack outcomes select R; norm leakage cannot be removed by R.",
            })

    d66_rows = []
    diagnostic_identity = torch.eye(1024, dtype=torch.float64)
    for use_permutation in PERMUTATIONS:
        for rounds in ROUNDS:
            q = transform_vectors(
                diagnostic_identity, FIXTURE_KEY,
                nonce_for(f"D6.6/R{rounds}/P{int(use_permutation)}"),
                rounds=rounds, use_permutation=use_permutation,
            ).T
            qtq_error = float(torch.max(torch.abs(q.T @ q - diagnostic_identity)))
            absolute = torch.abs(q)
            d66_rows.append({
                "rounds": rounds,
                "permutation": use_permutation,
                "dimension": 1024,
                "trace": float(torch.trace(q)),
                "trace_over_d": float(torch.trace(q) / 1024),
                "same_vector_cosine_expectation": float(torch.trace(q) / 1024),
                "basis_probe_nonzero_fraction": float(torch.mean((absolute[:, 0] > 1e-15).double())),
                "matrix_zero_fraction_tol_1e15": float(torch.mean((absolute <= 1e-15).double())),
                "matrix_abs_mean": float(torch.mean(absolute)),
                "matrix_abs_max": float(torch.max(absolute)),
                "orthogonality_max_abs": qtq_error,
                "interpretation": DISCLAIMER,
            })

    args.d61.parent.mkdir(parents=True, exist_ok=True)
    args.d61.write_text(json.dumps({**common, "task_id": "D6.1", "passed": d61_pass, "rows": d61_rows}, indent=2) + "\n")
    args.d62.write_text(json.dumps(d62, indent=2) + "\n")
    args.d63.write_text(json.dumps(d63, indent=2) + "\n")
    write_csv(args.d64, d64_rows)
    write_csv(args.d65, d65_rows)
    write_csv(args.d66, d66_rows)
    if not d61_pass or not d62["passed"]:
        raise RuntimeError("D6 numerical correctness gate failed")
    print(json.dumps({
        "D6.1_passed": d61_pass,
        "D6.2_passed": d62["passed"],
        "D6.3_after": d63["after"],
        "D6.4_rows": len(d64_rows),
        "D6.5_rows": len(d65_rows),
        "D6.6_rows": len(d66_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
