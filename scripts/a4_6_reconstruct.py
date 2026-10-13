#!/usr/bin/env python3
"""A4.6c end-to-end image reconstruction: compare the legitimate key-holder's
decrypt with the attacker's decrypt of the SAME stored anonymous image, where
the attacker's key was recovered (A4.6a/b) from only (original, anonymous) image
pairs of OTHER subjects.

For each victim and (scheme, storage): take the re-inverted anonymous latent
ẑ_ano (A4.1 query cache), de-anonymise it with (i) the true key -> legitimate
reconstruction, (ii) the recovered key -> attacker reconstruction, generate both
with the target checkpoint, and report PSNR / SSIM / L-infinity between the two
reconstructions and against the preprocessed original.  Panels are written for
inspection.  No raw key material is written.

  scripts/run_cfg_ddim.sh python scripts/managed_run.py --task-id A4.6_RECON \
    --validate-artifacts -- python -m scripts.a4_6_reconstruct \
    --checkpoint <ckpt> --latent-cache artifacts/runs/E2.3/<run> \
    --query-run artifacts/runs/A4.1/<run>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from scripts.e1_ddim_runner import Tee, create_runtime, load_split, preprocess
from scripts.e2_legacy_wrapper import apply_legacy_key, build_legacy_components, legacy_generate
from scripts.run_artifacts import finalize_run, open_run, sha256_file

KNOWN_POOL = (0, 1, 2, 3, 4, 5, 6, 7)
SCHEMES = ("S0", "S1")
STORAGES = ("legacy_png", "range_preserving_png")


# ---- recovery (CPU numpy; replicated so the module needn't force CUDA off) ---
def s0_recover(zs: list[np.ndarray], chats: list[np.ndarray]) -> np.ndarray:
    votes = np.zeros(zs[0].size, dtype=np.float64)
    for z, c in zip(zs, chats):
        votes += np.abs(z) * np.sign(z) * np.sign(c)
    k_hat = np.sign(votes)
    k_hat[k_hat == 0] = 1.0
    return k_hat.astype(np.float32)


def s1_recover(zs: list[np.ndarray], chats: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    source = np.stack([np.abs(z) for z in zs], axis=1)
    cipher = np.stack([np.abs(c) for c in chats], axis=1)
    _, pi_hat = cKDTree(source).query(cipher, k=1)
    pi_hat = pi_hat.astype(np.int64)
    zstack, cstack = np.stack(zs), np.stack(chats)
    best = np.argmax(np.abs(zstack), axis=0)
    inv = np.full(zs[0].size, -1, dtype=np.int64)
    inv[pi_hat] = np.arange(pi_hat.size)
    k_hat = np.empty(zs[0].size, dtype=np.float32)
    for j in range(zs[0].size):
        p = inv[j]
        c_at = cstack[best[j], p] if p >= 0 else 0.0
        k_hat[j] = np.sign(zstack[best[j], j]) * np.sign(c_at) or 1.0
    return k_hat, pi_hat


def s0_decrypt(k_hat: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (k_hat * c).astype(np.float32)


def s1_decrypt(k_hat: np.ndarray, pi_hat: np.ndarray, c: np.ndarray) -> np.ndarray:
    out = np.zeros_like(c)
    out[pi_hat] = k_hat[pi_hat] * c
    return out


def flat(cache, i) -> np.ndarray:
    return np.array(cache[i], dtype=np.float64).reshape(-1)


def as_latent(vec: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(vec.reshape(1, 1, 256, 256).astype(np.float32)).to(device)


def image_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = a.astype(np.float64), b.astype(np.float64)
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    rng = max(hi - lo, 1e-8)
    mse = float(np.mean((a - b) ** 2))
    # identical images give PSNR = inf; cap at 100 dB so the value stays finite.
    psnr = 100.0 if mse == 0.0 else min(float(peak_signal_noise_ratio(a, b, data_range=rng)), 100.0)
    return {
        "psnr_db": psnr,
        "ssim": float(structural_similarity(a, b, data_range=rng)),
        "linf": float(np.max(np.abs(a - b))),
        "rmse": float(np.sqrt(mse)),
    }


def save_panel(path: Path, x0, anon, legit, attacker):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    diff = np.abs(legit - attacker)
    panels = [("original", x0), ("anonymous", anon), ("legit recon", legit),
              ("attacker recon", attacker), ("|legit-attacker|", diff)]
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
    for ax, (title, img) in zip(axes, panels):
        vmax = 1.0 if title != "|legit-attacker|" else max(float(diff.max()), 1e-6)
        vmin = 0.0 if title != "|legit-attacker|" else 0.0
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--query-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--noise-level", type=int, default=500)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--victims", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--table-output", type=Path,
                        default=Path("paper_assets/tables/table_A4.6_reconstruct.csv"))
    parser.add_argument("--panel-dir", type=Path, default=Path("image/A4.6_panels"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--task-id", default="A4.6_RECON")
    try:
        from scripts.output_guard import add_force_flag, refuse_overwrite
    except ModuleNotFoundError:
        from output_guard import add_force_flag, refuse_overwrite
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.table_output], args.force)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    stdout_cap, stderr_cap = io.StringIO(), io.StringIO()
    with redirect_stdout(Tee(sys.stdout, stdout_cap)), redirect_stderr(Tee(sys.stderr, stderr_cap)):
        rows_split = load_split(args.split)
        cache = np.load(args.latent_cache / "latents.npy")
        qlat = np.load(args.query_run / "query_latents.npy")
        qindex = json.loads((args.query_run / "query_index.json").read_text())
        qpos = {r["sample_id"]: r["query_index"] for r in qindex}

        def chat(i, scheme, storage):
            return qlat[qpos[f"security_v1_{i:04d}|{scheme}|{storage}"]].reshape(-1).astype(np.float64)

        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(known_pool=list(KNOWN_POOL[:args.pairs]), schemes=list(SCHEMES), storages=list(STORAGES))
        context = open_run(args.task_id, config, args.artifacts_root)
        device = torch.device("cuda:0")
        _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
        args.panel_dir.mkdir(parents=True, exist_ok=True)

        known = KNOWN_POOL[:args.pairs]
        records: list[dict[str, Any]] = []
        for scheme in SCHEMES:
            zs = [flat(cache, i) for i in known]
            for storage in STORAGES:
                chats = [chat(i, scheme, storage) for i in known]
                if scheme == "S0":
                    k_hat, pi_hat = s0_recover(zs, chats), None
                else:
                    k_hat, pi_hat = s1_recover(zs, chats)
                for v in args.victims:
                    cv = chat(v, scheme, storage)
                    z_legit = apply_legacy_key(as_latent(cv, device), args.source_root, scheme, "deanonymize")
                    if scheme == "S0":
                        z_atk = as_latent(s0_decrypt(k_hat, cv), device)
                    else:
                        z_atk = as_latent(s1_decrypt(k_hat, pi_hat, cv), device)
                    recon_legit = legacy_generate(diffusion, model_fn, z_legit, args.noise_level)[0, 0].float().cpu().numpy()
                    recon_atk = legacy_generate(diffusion, model_fn, z_atk, args.noise_level)[0, 0].float().cpu().numpy()
                    # original + anonymous image for display
                    x0 = preprocess(Path(rows_split[v]["local_path"]))[1]
                    z_ano_true = apply_legacy_key(as_latent(flat(cache, v), device), args.source_root, scheme, "anonymize")
                    anon = legacy_generate(diffusion, model_fn, z_ano_true, args.noise_level)[0, 0].float().cpu().numpy()

                    panel = args.panel_dir / f"A4.6_{rows_split[v]['sample_id']}_{scheme}_{storage}.png"
                    save_panel(panel, x0, anon, recon_legit, recon_atk)

                    m_la = image_metrics(recon_legit, recon_atk)
                    m_ax = image_metrics(recon_atk, x0)
                    m_lx = image_metrics(recon_legit, x0)
                    records.append({
                        "schema_version": 1,
                        "sample_id": f"{rows_split[v]['sample_id']}|{scheme}|{storage}",
                        "image_id": rows_split[v]["sample_id"], "scheme": scheme, "storage": storage,
                        "pairs_used": args.pairs,
                        "legit_vs_attacker_psnr_db": m_la["psnr_db"], "legit_vs_attacker_ssim": m_la["ssim"],
                        "legit_vs_attacker_linf": m_la["linf"], "legit_vs_attacker_rmse": m_la["rmse"],
                        "attacker_vs_original_psnr_db": m_ax["psnr_db"], "attacker_vs_original_ssim": m_ax["ssim"],
                        "legit_vs_original_psnr_db": m_lx["psnr_db"], "legit_vs_original_ssim": m_lx["ssim"],
                        "panel": str(panel),
                    })
                    print(f"{records[-1]['sample_id']}: legit~attacker PSNR={m_la['psnr_db']:.2f}dB "
                          f"SSIM={m_la['ssim']:.4f} Linf={m_la['linf']:.3f} | atk~orig PSNR={m_ax['psnr_db']:.2f}")

        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        with args.table_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(records)

        numeric = [k for k, v in records[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        manifest = {
            "dataset_split_hash": rows_split[0]["split_sha256"],
            "checkpoint_hash": sha256_file(args.checkpoint),
            "latent_cache_run_id": json.loads((args.latent_cache / "manifest.json").read_text())["run_id"],
            "query_run": args.query_run.name,
            "query_latents_sha256": sha256_file(args.query_run / "query_latents.npy"),
            "pairs_used": args.pairs, "known_pool": list(known), "victims": list(args.victims),
            "pipeline": "re-inverted ẑ_ano -> (true | recovered) key de-anon -> generation",
            "raw_key_material_recorded": False,
            # schema-required provenance fields (predecessor legacy scheme)
            "scheme": ",".join(SCHEMES), "rounds": 0, "nonce_mode": "none",
            "container_version": "none", "seed_list": [0] * len(records), "steps": args.noise_level,
        }
        finalize_run(context, Path(__file__), records, numeric, manifest,
                     stdout_cap.getvalue(), stderr_cap.getvalue())


if __name__ == "__main__":
    main()
