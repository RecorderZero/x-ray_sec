#!/usr/bin/env python3
"""Layer-B: exact-reversible samplers vs respaced DDIM at matched NFE.

The predecessor's BDIA appendix (`scripts/bdia_backend.py`) was run at T=500 with
Delta t = 1 (hard-wired to `alphas_cumprod_prev/next`) and compared against the
*legacy* backward, which we have since shown carries a decode-grid off-by-one
(~4.5 dB loss).  This module re-does BDIA on the SAME *respaced* grid and aligned
decode math as `scripts.respaced_ddim_sweep.Respaced`, so BDIA and DDIM are
apples-to-apples at identical step boundaries and conditioning.

BDIA (gamma=1, Zhang et al., ECCV 2024) is algebraically exact-reversible with one
model evaluation per step -- same NFE as DDIM.  One eval at x_i yields both the
generation step G (lands at the scheduled neighbour toward data, b[i-1]) and the
reverse step R (lands at the scheduled neighbour toward noise, b[i+1]); the
leapfrog recurrence

  generate:  X[i-1] = X[i+1] + G(X[i]) - R(X[i])
  invert:    X[i+1] = X[i-1] + R(X[i]) - G(X[i])

reverses step-for-step (boundary step is pure DDIM, the only non-exact part).

We isolate the sampler by the KEY-FREE round-trip x0 -> encode -> z -> decode ->
x_rec (encryption is bit-exact and orthogonal to step count).  Validation gate:
at full density (N~499) both samplers must reach the legacy dense anchor (within
1 dB) AND degrade monotonically with fewer steps; otherwise the math is wrong.

  smoke (2 imgs):  scripts/run_cfg_ddim.sh python -m scripts.exact_inv_sweep --smoke
  full  (4 imgs):  scripts/run_cfg_ddim.sh python -m scripts.exact_inv_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_legacy_wrapper import legacy_invert, legacy_generate
from scripts.respaced_ddim_sweep import Respaced, schedule, psnr, SEED, TOP

CKPT = Path("past/SourceCode/results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt")
FULL_INDICES = (0, 1, 10, 11)
SMOKE_INDICES = (0, 1)
SWEEP_N = (499, 200, 100, 50, 25)


class BDIARespaced:
    """BDIA (gamma=1) leapfrog on the respaced grid, sharing Respaced's eval math.

    Conditioning mirrors Respaced exactly: inversion uses the forward kwargs
    (null=True), generation uses the backward kwargs (guidance -1, no null), so
    the only difference from DDIM is the leapfrog integration -- not the model
    inputs.  NFE == DDIM (len(boundaries)-1 evals per pass).
    """

    def __init__(self, rs: Respaced):
        self.rs = rs

    @torch.inference_mode()
    def _gr(self, x, t_cur, b_prev, b_next, mk):
        """One model eval at t_cur -> (G lands at abar[b_prev], R lands at abar[b_next])."""
        t = torch.full((x.shape[0],), int(t_cur), device=self.rs.dev, dtype=torch.long)
        px = self.rs.d.p_mean_variance(self.rs.m, x, t, clip_denoised=True, model_kwargs=mk)["pred_xstart"]
        eps = self.rs.d._predict_eps_from_xstart(x, t, px)
        ap = self.rs.abar[int(b_prev)].to(x.dtype)
        an = self.rs.abar[int(b_next)].to(x.dtype)
        g = px * torch.sqrt(ap) + torch.sqrt(1 - ap) * eps
        r = px * torch.sqrt(an) + torch.sqrt(1 - an) * eps
        return g, r

    @torch.inference_mode()
    def forward(self, x0, b):
        """Inversion x0 -> z (ascending grid b[0]=0 .. b[M]=TOP)."""
        mk = self.rs._base_kwargs(x0.shape[0]); mk["null"] = True
        M = len(b) - 1
        prev, curr = None, x0  # curr at grid index i
        for i in range(0, M):
            if i == 0:  # boundary: pure DDIM step index 0 -> 1
                nxt = self.rs._step(curr, b[0], b[1], mk)
            else:
                g, r = self._gr(curr, b[i], b[i - 1], b[i + 1], mk)
                nxt = prev + r - g
            prev, curr = curr, nxt
        return curr

    @torch.inference_mode()
    def backward(self, z, b):
        """Generation z -> x_rec (descending over the same grid b)."""
        mk = self.rs._base_kwargs(z.shape[0])  # guidance -1, no null
        M = len(b) - 1
        prev, curr = None, z  # curr at grid index i
        for i in range(M, 0, -1):
            if i == M:  # boundary: pure DDIM step index M -> M-1
                nxt = self.rs._step(curr, b[M], b[M - 1], mk)
            else:
                g, r = self._gr(curr, b[i], b[i - 1], b[i + 1], mk)
                nxt = prev + g - r
            prev, curr = curr, nxt
        return curr


class DPMSolverpp:
    """DPM-Solver++(2M) data-prediction, descending generation / ascending inversion.

    1 model eval per step (NFE parity with DDIM/BDIA).  The exponential-integrator
    update X_t = (s_t/s_s) X_s - a_t (e^{-h}-1) D is exact to 2nd order with the
    multistep buffer D = (1+1/2r) x0_s - 1/2r x0_prev (r = h_prev/h); boundary step
    is first order (== DDIM).  This is NOT bit-exact-reversible (the 2M buffer runs
    opposite directions on the two passes, same obstruction BELM sidesteps and the
    reason we do not claim exactness here) -- it is the diffusers-style approximate
    inverse, compared fairly at matched NFE on the round-trip PSNR.  Conditioning
    mirrors Respaced (inversion null=True, generation no null).
    """

    def __init__(self, rs: Respaced):
        self.rs = rs
        ac = rs.abar
        self.alpha = ac.sqrt()                       # a_t
        self.sigma = (1 - ac).sqrt()                 # s_t
        self.lam = 0.5 * (ac.log() - (1 - ac).log())  # lambda_t = log(a/s)

    @torch.inference_mode()
    def _x0(self, x, t_cur, mk):
        t = torch.full((x.shape[0],), int(t_cur), device=self.rs.dev, dtype=torch.long)
        return self.rs.d.p_mean_variance(self.rs.m, x, t, clip_denoised=True, model_kwargs=mk)["pred_xstart"]

    def _coef(self, s, t, dt):
        """Return (c1, c2, h) for a step from timestep s to t (dtype dt)."""
        h = (self.lam[int(t)] - self.lam[int(s)]).item()
        c1 = (self.sigma[int(t)] / self.sigma[int(s)]).to(dt)
        import math as _m
        c2 = (-self.alpha[int(t)].item() * (_m.exp(-h) - 1.0))
        return c1, float(c2), h

    @torch.inference_mode()
    def _run(self, x, order, mk):
        """order = list of (s, t) timestep pairs in traversal order."""
        x0_prev, h_prev = None, None
        for k, (s, t) in enumerate(order):
            c1, c2, h = self._coef(s, t, x.dtype)
            x0_s = self._x0(x, s, mk)
            if k == 0:
                D = x0_s
            else:
                r = h_prev / h
                D = (1 + 1 / (2 * r)) * x0_s - (1 / (2 * r)) * x0_prev
            x = c1 * x + c2 * D
            x0_prev, h_prev = x0_s, h
        return x

    def forward(self, x0, b):
        mk = self.rs._base_kwargs(x0.shape[0]); mk["null"] = True
        order = [(int(b[i]), int(b[i + 1])) for i in range(len(b) - 1)]  # ascending
        return self._run(x0, order, mk)

    def backward(self, z, b):
        mk = self.rs._base_kwargs(z.shape[0])
        order = [(int(b[i]), int(b[i - 1])) for i in range(len(b) - 1, 0, -1)]  # descending
        return self._run(z, order, mk)


def _timed(fn, *a):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="2 images, fewer, for pipeline sanity")
    ap.add_argument("--split", type=Path, default=Path("splits/dev_v1.1.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/exact_inv_sweep"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices = list(SMOKE_INDICES if args.smoke else FULL_INDICES)

    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), CKPT, device)
    rs = Respaced(diffusion, model_fn, device)
    bdia = BDIARespaced(rs)
    dpm = DPMSolverpp(rs)

    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    imgs = [preprocess(Path(rows[i]["local_path"]))[0].unsqueeze(0).to(device) for i in indices]
    ids = [rows[i]["sample_id"] for i in indices]

    # warm-up
    rs.backward(rs.forward(imgs[0], schedule(10)), schedule(10))

    # legacy dense round-trip anchor (no crypto) on the same images
    anchor = float(np.mean([
        psnr(legacy_generate(diffusion, model_fn, legacy_invert(diffusion, model_fn, x0, 500), 500), x0)
        for x0 in imgs]))
    print(f"[anchor] legacy dense round-trip = {anchor:.2f} dB", flush=True)

    records = []
    for n in SWEEP_N:
        b = schedule(n)
        for x0, sid in zip(imgs, ids):
            zc, tf_d = _timed(rs.forward, x0, b)
            xr_d, tb_d = _timed(rs.backward, zc, b)
            zc_b, tf_b = _timed(bdia.forward, x0, b)
            xr_b, tb_b = _timed(bdia.backward, zc_b, b)
            zc_p, tf_p = _timed(dpm.forward, x0, b)
            xr_p, tb_p = _timed(dpm.backward, zc_p, b)
            records.append({
                "N": n, "evals_per_pass": len(b) - 1, "sample_id": sid,
                "ddim_psnr_db": psnr(xr_d, x0), "bdia_psnr_db": psnr(xr_b, x0),
                "dpm_psnr_db": psnr(xr_p, x0),
                "ddim_finite": bool(torch.isfinite(xr_d).all()),
                "bdia_finite": bool(torch.isfinite(xr_b).all()),
                "dpm_finite": bool(torch.isfinite(xr_p).all()),
                "ddim_pass_s": round((tf_d + tb_d) / 2, 4), "bdia_pass_s": round((tf_b + tb_b) / 2, 4),
                "dpm_pass_s": round((tf_p + tb_p) / 2, 4),
            })
        last = records[-len(imgs):]
        md = float(np.mean([r["ddim_psnr_db"] for r in last]))
        mb = float(np.mean([r["bdia_psnr_db"] for r in last]))
        mp = float(np.mean([r["dpm_psnr_db"] for r in last]))
        print(f"N={n:>4} evals={len(b)-1:>4}  DDIM={md:7.2f}  BDIA={mb:7.2f} ({mb-md:+5.2f})  "
              f"DPM++={mp:7.2f} ({mp-md:+5.2f})", flush=True)

    summary = {}
    for n in SWEEP_N:
        rr = [r for r in records if r["N"] == n]
        summary[n] = {
            "evals_per_pass": rr[0]["evals_per_pass"],
            "ddim_psnr_db": round(float(np.mean([r["ddim_psnr_db"] for r in rr])), 2),
            "bdia_psnr_db": round(float(np.mean([r["bdia_psnr_db"] for r in rr])), 2),
            "dpm_psnr_db": round(float(np.mean([r["dpm_psnr_db"] for r in rr])), 2),
            "ddim_pass_s": round(float(np.mean([r["ddim_pass_s"] for r in rr])), 4),
            "bdia_pass_s": round(float(np.mean([r["bdia_pass_s"] for r in rr])), 4),
            "dpm_pass_s": round(float(np.mean([r["dpm_pass_s"] for r in rr])), 4),
            "all_finite": all(r["ddim_finite"] and r["bdia_finite"] and r["dpm_finite"] for r in rr),
        }

    def mono(key):
        vals = [summary[n][key] for n in SWEEP_N]  # N descending
        return all(vals[i] >= vals[i + 1] - 0.05 for i in range(len(vals) - 1))

    full = summary[SWEEP_N[0]]
    gate = {
        "ddim_full_ge_anchor": full["ddim_psnr_db"] >= anchor - 1.0,
        "bdia_full_ge_anchor": full["bdia_psnr_db"] >= anchor - 1.0,
        "dpm_full_ge_anchor": full["dpm_psnr_db"] >= anchor - 1.0,
        "ddim_monotonic": mono("ddim_psnr_db"),
        "bdia_monotonic": mono("bdia_psnr_db"),
        "dpm_monotonic": mono("dpm_psnr_db"),
        "all_finite": all(summary[n]["all_finite"] for n in SWEEP_N),
    }
    gate["pass"] = all(gate.values())
    payload = {"seed": SEED, "top_timestep": TOP, "smoke": args.smoke, "indices": indices,
               "legacy_dense_anchor_db": round(anchor, 3), "gate": gate,
               "summary": summary, "per_sample": records}
    name = "smoke.json" if args.smoke else "sweep.json"
    (args.out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== exact-inv samplers vs DDIM (respaced, key-free round-trip; mean over {len(imgs)} imgs) ===")
    print(f"{'N':>5}{'evals':>7}{'DDIM':>9}{'BDIA':>9}{'DPM++':>9}{'s/pass':>8}")
    for n in SWEEP_N:
        s = summary[n]
        print(f"{n:>5}{s['evals_per_pass']:>7}{s['ddim_psnr_db']:>9.2f}{s['bdia_psnr_db']:>9.2f}"
              f"{s['dpm_psnr_db']:>9.2f}{s['ddim_pass_s']:>8.3f}")
    print(f"\nanchor={anchor:.2f} dB  gate={json.dumps(gate)}")
    print(f"wrote {args.out_dir}/{name}")


if __name__ == "__main__":
    main()
