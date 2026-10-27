#!/usr/bin/env python3
"""No-GT parallel-latent localization smoke; not a lesion-accuracy benchmark.

Launch via scripts.managed_run (task-id LOC-SMOKE) to obtain durable logs,
heartbeat, signal handling, unique run directory and terminal status.
Recovery and pseudo-healthy generation branch from the same recovered latent;
the recovery image is NEVER re-inverted. Execution is sequential on one GPU,
but the two branches are independent in the computational graph.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import MODEL_CONFIG, create_runtime, load_split, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.exact_inv_sweep import CKPT, DPMSolverpp
from scripts.layerA_dpm_ablation import ddim_cfg_backward, dpm_cfg_backward
from scripts.respaced_ddim_sweep import SEED, Respaced, schedule
from scripts.run_artifacts import finalize_run, git_value, open_run, sha256_file


def finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"nonfinite tensor: {name}")
    return value


@torch.inference_mode()
def cfg_data_prediction(rs, x, timestep, gs):
    """Mix clipped branch x0 directly, preserving layerA's clip-before-mix.

    Algebraically equivalent to mixing branch eps, without an unnecessary
    float32 x0->eps->x0 roundtrip using differently rounded coefficients.
    This is NOT raw-epsilon CFG followed by clipping the mixed prediction.
    """
    t = torch.full((x.shape[0],), int(timestep), device=rs.dev, dtype=torch.long)
    cond_kwargs = rs._base_kwargs(x.shape[0])
    xc = rs.d.p_mean_variance(rs.m, x, t, clip_denoised=True,
                             model_kwargs=cond_kwargs)["pred_xstart"]
    if gs == 0:
        return xc, t
    uncond_kwargs = dict(cond_kwargs, null=True)
    xu = rs.d.p_mean_variance(rs.m, x, t, clip_denoised=True,
                             model_kwargs=uncond_kwargs)["pred_xstart"]
    return xc + gs * (xc - xu), t


@torch.inference_mode()
def canonical_cfg_backward(sampler, latent, boundaries, gs):
    """Canonical direct-x0 guided update; gs0 exactly follows backward math."""
    is_dpm = isinstance(sampler, DPMSolverpp)
    rs = sampler.rs if is_dpm else sampler
    x = latent
    previous, previous_h = None, None
    for k in range(len(boundaries)-1, 0, -1):
        s, dest = int(boundaries[k]), int(boundaries[k-1])
        px, t = cfg_data_prediction(rs, x, s, gs)
        if is_dpm:
            c1, c2, h = sampler._coef(s, dest, x.dtype)
            if previous is None:
                data = px
            else:
                r = previous_h / h
                data = (1 + 1/(2*r)) * px - (1/(2*r)) * previous
            x = c1*x + c2*data
            previous, previous_h = px, h
        else:
            eps = rs.d._predict_eps_from_xstart(x, t, px)
            alpha = rs.abar[dest].to(x.dtype)
            x = px * torch.sqrt(alpha) + torch.sqrt(1-alpha) * eps
    return x


@torch.inference_mode()
def unconditional_backward(sampler, z, boundaries):
    """Recovery backward matched to the (unconditional) inversion, i.e. null=True.

    The shipped `sampler.backward` conditions on y=0 (an asymmetry vs the null
    inversion inherited from the predecessor).  For localization the recovery
    must stay class-neutral so the lesion survives into |recovery - healthy|;
    recovery_condition_smoke.py showed null recovery gains +1.1 dB (DDIM) /
    +4.3 dB (DPM++) in fidelity and ~1.5x larger difference signal.  This helper
    does NOT mutate the shared `sampler.backward` used by the frozen RQ4/linkage
    results -- it only supplies the null variant for this localization branch.
    """
    rs = sampler.rs if isinstance(sampler, DPMSolverpp) else sampler
    mk = rs._base_kwargs(z.shape[0]); mk["null"] = True
    if isinstance(sampler, DPMSolverpp):
        order = [(int(boundaries[k]), int(boundaries[k - 1])) for k in range(len(boundaries) - 1, 0, -1)]
        return sampler._run(z, order, mk)
    x = z
    for k in range(len(boundaries) - 1, 0, -1):
        x = rs._step(x, boundaries[k], boundaries[k - 1], mk)
    return x


@torch.inference_mode()
def parallel_branches(sampler, recovery_backward, cfg_backward, latent, boundaries, guidance):
    """Shared decrypted latent -> null recovery + guided pseudo-healthy branches.

    recovery_backward is the (unconditional) recovery generator -- supplied by the
    caller rather than hardwired to sampler.backward (y=0), so the recovery stays
    class-neutral.  The recovery is NEVER re-inverted.  Inputs are cloned to
    protect against accidental in-place samplers.
    """
    finite("latent", latent)
    recovery = finite("recovery", recovery_backward(latent.clone(), boundaries))
    healthy = {}
    maps = {}
    for gs in guidance:
        healthy[gs] = finite("healthy", cfg_backward(latent.clone(), gs))
        maps[gs] = finite("difference", (recovery - healthy[gs]).abs())
    return recovery, healthy, maps


@torch.inference_mode()
def recovered_latent(sampler, x, boundaries, scheme, sample_id):
    """Original branch or unauthenticated S2 numerical/storage fixture only."""
    z = finite("inversion", sampler.forward(x, boundaries))
    if scheme == "original":
        return z
    if scheme != "S2fn-transform-only":
        raise ValueError(f"unknown scheme: {scheme}")
    from scripts.deploy_s2_sweep import encrypt, decrypt
    from scripts.m1_storage import range_preserving_png_handoff

    encrypted, norm = encrypt(z, "S2fn", sample_id)
    anonymous = finite("anonymous", sampler.backward(encrypted, boundaries))
    handoff, _ = range_preserving_png_handoff(anonymous)
    noisy = finite("anonymous inversion", sampler.forward(handoff, boundaries))
    return finite("decrypted", decrypt(noisy, "S2fn", sample_id, norm))


def map_statistics(value):
    a = finite("map", value).detach().double().flatten()
    return {"mean": float(a.mean()), "max": float(a.max()),
            "p95": float(torch.quantile(a, 0.95)),
            "fraction_gt_001": float((a > 0.01).double().mean())}


def map_agreement(a, b):
    """Raw-map MAE and centered correlation; undefined correlation stays null."""
    a = finite("map a", a).detach().double().flatten()
    b = finite("map b", b).detach().double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    denom = float(ac.norm() * bc.norm())
    return {"mae": float((a - b).abs().mean()),
            "linf": float((a - b).abs().max()),
            "pearson": float(ac.dot(bc)) / denom if denom > 1e-20 else None}


def gs0_sanity(y0_reference, healthy_gs0, tolerance=1e-4):
    """CFG gs=0 must reduce to a single y=0 conditional pass (== sampler.backward).

    Recovery is now the unconditional (null) backward, so it is no longer the gs0
    reference; the image-independent invariant checked here is the CFG-math
    identity (1+0)*x0_cond - 0*x0_uncond = x0_cond.
    """
    error = float((finite("y0_reference", y0_reference) - finite("gs0", healthy_gs0)).abs().max())
    if error > tolerance:
        raise ValueError(f"gs=0 conditional equivalence failed: {error} > {tolerance}")
    return error


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--source-root", type=Path, default=Path("past/SourceCode"))
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 10])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--include-s2", action="store_true")
    parser.add_argument("--diagnose-legacy", action="store_true",
                        help="record historical layerA float32 roundtrip difference; not a gate")
    args = parser.parse_args()
    if not os.environ.get("EXPERIMENT_RUN_DIR"):
        parser.error("launch via scripts.managed_run --task-id LOC-SMOKE (durable run status required)")
    if not 1 <= args.steps <= 499 or not 1 <= len(args.indices) <= 4:
        parser.error("smoke only: steps 1..499 and 1..4 dev images")
    if len(set(args.indices)) != len(args.indices) or min(args.indices) < 0:
        parser.error("indices must be unique and nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU tests do not load the checkpoint")
    rows = load_split(args.split)
    chosen = [rows[i] for i in args.indices]
    schemes = ["original"] + (["S2fn-transform-only"] if args.include_s2 else [])
    guidance = [0.0, 2.0]
    b = schedule(args.steps)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"seed": SEED, "guidance": guidance, "schemes": schemes,
                   "model_config": MODEL_CONFIG, "boundaries": b.tolist(),
                   "resumable": False, "gt_available": False,
                   "guidance_convention": "(1+gs)*eps_y0 - gs*eps_null; gs0=conditional",
                   "recovery_conditioning": "null=True (unconditional, matched to inversion)",
                   "inversion_conditioning": "y=0,p_uncond=-1,clf_free=True,null=True",
                   "difference_map": "abs(recovery-healthy), raw unclipped float, no normalization",
                   "cfg_update": "direct clipped-x0 mixture; clip-before-mix matching layerA convention",
                   "security_scope": "public test key and deterministic fixture nonce; norm held in RAM; no AEAD"})
    config["model_calls_per_sample_per_sampler"] = {
        scheme: ((3 if scheme == "S2fn-transform-only" else 1) + 1 + 1 + sum(1 if gs == 0 else 2 for gs in guidance)) * (len(b)-1)
        for scheme in schemes
    }
    config["pass_accounting"] = "prefix original=1 or S2=3; null recovery=1; y0 gs0-sanity reference=1; healthy gs0=1, gs2=2 model calls/step; optional legacy diagnostic adds 4N calls per original branch"
    context = open_run("LOC-SMOKE", config, Path("artifacts/runs"))
    started = time.perf_counter()
    checkpoint_hash = sha256_file(args.checkpoint)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(args.source_root, args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device)
    dpm = DPMSolverpp(rs)
    samplers = {"aligned-DDIM": (rs, lambda z, gs: canonical_cfg_backward(rs, z, b, gs)),
                "DPM++2M": (dpm, lambda z, gs: canonical_cfg_backward(dpm, z, b, gs))}
    details, metrics = [], []
    numeric_fields = ["recovery_psnr_db", "recovery_ssim", "gs0_linf", "elapsed_seconds"]
    cross = []
    for item_no, row in enumerate(chosen):
        sid = row["sample_id"]
        x = preprocess(Path(row["local_path"]))[0].unsqueeze(0).to(device)
        cache = {}
        for name, (sampler, cfg) in samplers.items():
            for scheme in schemes:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                z = recovered_latent(sampler, x, b, scheme, sid)
                recovery, healthy, maps = parallel_branches(
                    sampler, lambda zz, bb: unconditional_backward(sampler, zz, bb), cfg, z, b, guidance)
                sanity = gs0_sanity(sampler.backward(z.clone(), b), healthy[0.0])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                rec = image_pair_metrics("recovery", x, recovery)
                uid = f"{sid}/{name}/{scheme}"
                metrics.append({"sample_id": uid, "source_sample_id": sid, "sampler": name,
                                "scheme": scheme, "recovery_psnr_db": rec["recovery_psnr_db"],
                                "recovery_ssim": rec["recovery_ssim"], "gs0_linf": sanity,
                                "elapsed_seconds": elapsed})
                detail = {"sample_id": uid, "recovery_vs_original": rec, "guidance": {}}
                if args.diagnose_legacy and scheme == "original":
                    from scripts.layerA_dpm_ablation import _cfg_eps, _x0_from_eps
                    direct_px, _ = cfg_data_prediction(rs, z, b[-1], 0.0)
                    roundtrip_px = _x0_from_eps(rs, z, b[-1], _cfg_eps(rs, z, b[-1], 0.0))
                    legacy = ddim_cfg_backward if name == "aligned-DDIM" else dpm_cfg_backward
                    legacy0 = legacy(sampler, z, b, 0.0)
                    legacy2 = legacy(sampler, z, b, 2.0)
                    detail["legacy_layerA_diagnostic"] = {
                        "first_step_x0_roundtrip_linf": float((direct_px-roundtrip_px).abs().max()),
                        "gs0_vs_recovery_linf": float((legacy0-recovery).abs().max()),
                        "gs2_vs_canonical_linf": float((legacy2-healthy[2.0]).abs().max()),
                        "gs2_map_agreement": map_agreement((recovery-legacy2).abs(), maps[2.0]),
                        "cause": "redundant x0/eps roundtrip with differently rounded float32 coefficients",
                    }
                    print(json.dumps(detail["legacy_layerA_diagnostic"]), flush=True)
                arrays = {"original": x.cpu().numpy(), "recovery": recovery.cpu().numpy()}
                for gs in guidance:
                    detail["guidance"][str(gs)] = {
                        "healthy_vs_original": image_pair_metrics("healthy", x, healthy[gs]),
                        "map_stats": map_statistics(maps[gs])}
                    arrays[f"healthy_gs{gs:g}"] = healthy[gs].cpu().numpy()
                    arrays[f"map_gs{gs:g}"] = maps[gs].cpu().numpy()
                    cache[name, scheme, gs] = maps[gs].cpu()
                np.savez_compressed(context.run_dir / f"local_only_{item_no}_{name}_{scheme}.npz", **arrays)
                details.append(detail)
                print(json.dumps(metrics[-1]), flush=True)
        for scheme in schemes:
            for gs in guidance:
                cross.append({"sample_id": sid, "scheme": scheme, "gs": gs,
                              **map_agreement(cache["aligned-DDIM", scheme, gs], cache["DPM++2M", scheme, gs])})
    payload = {"claim_scope": "numerical smoke only; no ground-truth localization accuracy",
               "elapsed_seconds": time.perf_counter() - started, "per_sample": details,
               "cross_sampler_map_agreement": cross}
    (context.run_dir / "localization_detail.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    finalize_run(context, Path(__file__), metrics, numeric_fields, {
        "dataset_split_hash": rows[0]["split_sha256"], "split_file_sha256": sha256_file(args.split),
        "checkpoint_hash": checkpoint_hash, "seed_list": [SEED], "steps": args.steps,
        "scheme": schemes, "rounds": 2 if args.include_s2 else 0,
        "nonce_mode": "public deterministic fixture; nonproduction" if args.include_s2 else "none",
        "container_version": "none", "gt_available": False,
        "source_sample_count": len(chosen), "elapsed_seconds": time.perf_counter() - started,
        "git_status_at_finish": git_value("status", "--porcelain=v1"),
        "nfe_per_recovery_branch": len(b)-1,
        "nfe_per_healthy_branch": {str(gs): (1 if gs == 0 else 2)*(len(b)-1) for gs in guidance},
        "image_hashes": {r["sample_id"]: sha256_file(Path(r["local_path"])) for r in chosen},
    })


if __name__ == "__main__":
    main()
