#!/usr/bin/env python3
"""Q11 localization parity demo (GPU): run S0 / S1 / S2 -- all under the SAME
deployment envelope (fixed-norm + range-preserving, in-memory float) -- through
the predecessor's FULL pixel-space M1 pipeline and ask whether swapping the
cipher changes the downstream lesion-localization output.

Pipeline per (image, cipher), faithful to the predecessor's
``ddim_sample_loop_anomaly_detection_with_deanonymization``:

  encrypt : x0  --DDIM invert(L)-->  x_T  --cipher-->  v  --DDIM gen(L)-->  x_ano
  decrypt : x_ano --invert--> x_ano_T --cipher^-1--> u --gen--> x_rec
  healthy : x_rec --invert--> x_rec_T --CFG gen(gs=4)--> x_healthy  (pseudo-healthy)
  heatmap : |x_rec - x_healthy|

The cipher acts at x_T (flattened 65536 = 2^16).  All three share one envelope:
fixed-norm scaling to C (private per-image norm restored on decrypt, lossless) +
an orthogonal core that is the ONLY thing differing between methods:
  S0 = Rademacher sign flip ; S1 = signed permutation ;
  S2 = our structured orthogonal transform (src.crypto.sot, Hadamard-Rademacher).

This is NOT a localization-accuracy benchmark (no ground-truth masks; the medical
11-image set is unavailable).  It is a PARITY demo: if S2's restored and
pseudo-healthy images match S0/S1's, then replacing the encryption layer is
transparent to the downstream localization stage -- the by-construction claim,
checked end-to-end through the real sampler.  Expected to pass; the value is the
empirical confirmation, not a new capability.

Deterministic: seed 1911; eta=0 DDIM; cipher keys from a fixed public fixture
(d7_cpu_security.FIXTURE_KEY; not key material).  Only aggregate image metrics and
rendered images are written -- never key bytes.

  scripts/run_cfg_ddim.sh python -m scripts.q11_localization_parity --n 1   # smoke
  scripts/run_cfg_ddim.sh python -m scripts.q11_localization_parity --n 10  # full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[1]
PAST = ROOT / "past" / "SourceCode"
sys.path.insert(0, str(PAST))

from guided_diffusion.script_util import (  # noqa: E402
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for  # noqa: E402
from src.crypto.sot import transform, inverse_transform  # noqa: E402

try:
    from src.crypto.s2_fixed_norm import FIXED_NORM as C_NORM  # noqa: E402
except Exception:  # pragma: no cover - fall back to the documented constant
    C_NORM = 248.261

SEED = 1911
IMG = 256
DIM = IMG * IMG          # 65536 = 2^16 (power of two required by the Hadamard core)
NOISE_LEVEL = 500        # L, DDIM inversion/generation steps (thesis setting)
GUIDANCE = 4.0           # CFG scale for the pseudo-healthy stage (thesis setting)
MODEL_PATH = PAST / "results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt"
DATASET = ROOT / "dataset/chexpertchestxrays-u20210408"
TRAIN_BATCHES = [
    "CheXpert-v1.0 batch 2 (train 1)",
    "CheXpert-v1.0 batch 3 (train 2)",
    "CheXpert-v1.0 batch 4 (train 3)",
]

MODEL_OVERRIDES = dict(
    image_size=256, in_channels=1, num_channels=128, num_classes=2, class_cond=True,
    num_res_blocks=2, num_heads=1, learn_sigma=True, use_scale_shift_norm=False,
    attention_resolutions="16", unet_version="v1", diffusion_steps=1000,
    noise_schedule="linear", rescale_learned_sigmas=False, rescale_timesteps=False,
    timestep_respacing="ddim1000",
)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------- model
def build_model(device):
    defaults = model_and_diffusion_defaults()
    for k, v in MODEL_OVERRIDES.items():
        if k in defaults:
            defaults[k] = v
    model, diffusion = create_model_and_diffusion(**defaults)
    state = torch.load(str(MODEL_PATH), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, diffusion, n_params


def make_model_fn(model, class_cond=True):
    def model_fn(x, t, y=None, p_uncond=-1, null=False, clf_free=True):
        assert y is not None
        return model(x, t, y if class_cond else None, p_uncond, null, clf_free)
    return model_fn


# ----------------------------------------------------------------------------- data
def resolve_image(csv_rel_path: str):
    """Map a CSV path 'CheXpert-v1.0/train/patient.../viewZ.jpg' to a file on disk."""
    marker = "train/"
    idx = csv_rel_path.find(marker)
    suffix = csv_rel_path[idx + len(marker):] if idx >= 0 else csv_rel_path
    for batch in TRAIN_BATCHES:
        cand = DATASET / batch / suffix
        if cand.exists():
            return cand
    return None


def select_images(n: int):
    """First n frontal Pleural-Effusion-positive images that resolve on disk."""
    csv_path = DATASET / "train_visualCheXbert.csv"
    picks = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Frontal/Lateral") != "Frontal":
                continue
            if row.get("Pleural Effusion") != "1.0":
                continue
            path = resolve_image(row["Path"])
            if path is None:
                continue
            name = "_".join(path.parts[-3:]).replace(".jpg", "")
            picks.append((name, path))
            if len(picks) >= n:
                break
    return picks


def load_image(path: Path, device):
    arr = np.asarray(Image.open(path).convert("L").resize((IMG, IMG), Image.BICUBIC)).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)   # visualize() -> [0,1]
    return torch.from_numpy(arr).view(1, 1, IMG, IMG).to(device)


# ----------------------------------------------------------------------------- ciphers
class Envelope:
    """Shared fixed-norm (+ range-preserving, in-memory) wrapper around an orthogonal
    core.  encrypt stores the private per-image norm so decrypt is lossless."""

    def __init__(self, core_fwd, core_inv, device):
        self.core_fwd, self.core_inv = core_fwd, core_inv
        self.device = device
        self.s = None

    def encrypt(self, x_t: torch.Tensor) -> torch.Tensor:
        flat = x_t.reshape(-1)
        self.s = float(flat.norm())
        u = flat * (C_NORM / self.s)
        v = self.core_fwd(u)
        return v.reshape(x_t.shape)

    def decrypt(self, y_t: torch.Tensor) -> torch.Tensor:
        flat = y_t.reshape(-1)
        u = self.core_inv(flat)
        z = u * (self.s / C_NORM)
        return z.reshape(y_t.shape)


def make_cipher(method: str, device) -> Envelope:
    gen = torch.Generator().manual_seed(SEED)
    if method == "S0":
        k = (torch.randint(0, 2, (DIM,), generator=gen).float() * 2 - 1).to(device)
        return Envelope(lambda u: k * u, lambda y: k * y, device)
    if method == "S1":
        k = (torch.randint(0, 2, (DIM,), generator=gen).float() * 2 - 1).to(device)
        perm = torch.randperm(DIM, generator=gen).to(device)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(DIM, device=device)
        return Envelope(lambda u: (k * u)[perm], lambda y: k * y[inv], device)
    if method == "S2":
        nonce = nonce_for("q11-localization-parity")
        fwd = lambda u: transform(u, FIXTURE_KEY, nonce, rounds=1, use_permutation=True)
        inv = lambda y: inverse_transform(y, FIXTURE_KEY, nonce, rounds=1, use_permutation=True)
        return Envelope(fwd, inv, device)
    raise ValueError(method)


# ----------------------------------------------------------------------------- pipeline
def invert(diffusion, model_fn, x0, mk):
    x_t, _ = diffusion.ddim_anonymization_forward(
        model_fn, x0, NOISE_LEVEL, clip_denoised=True, model_kwargs=mk, progress=False)
    return x_t


def generate(diffusion, model_fn, x_t, mk, guidance):
    return diffusion.ddim_anonymization_backward(
        model_fn, x_t, NOISE_LEVEL, clip_denoised=True, model_kwargs=mk,
        guidance_scale=guidance, progress=False, intermediates=None)


def run_one(diffusion, model_fn, x0, cipher, mk):
    x_T = invert(diffusion, model_fn, x0, mk)
    x_ano = generate(diffusion, model_fn, cipher.encrypt(x_T), mk, -1)
    x_ano_T = invert(diffusion, model_fn, x_ano, mk)
    x_rec = generate(diffusion, model_fn, cipher.decrypt(x_ano_T), mk, -1)
    x_rec_T = invert(diffusion, model_fn, x_rec, mk)
    x_healthy = generate(diffusion, model_fn, x_rec_T, mk, GUIDANCE)
    heat = (x_rec - x_healthy).abs()
    return dict(encrypted=x_ano, restored=x_rec, healthy=x_healthy, heatmap=heat)


# ----------------------------------------------------------------------------- metrics / io
def to_img(t: torch.Tensor) -> np.ndarray:
    a = t.detach().float().cpu().numpy().reshape(IMG, IMG)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def try_ssim(a, b):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(a, b, data_range=1.0))
    except Exception:
        return ""


def save_png(arr01: np.ndarray, path: Path, colormap=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if colormap:
        rgba = cm.get_cmap("jet")(arr01)
        Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8)).save(path)
    else:
        Image.fromarray((arr01 * 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--n", type=int, default=1, help="number of images")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_assets/localization_parity")
    parser.add_argument("--table", type=Path, default=ROOT / "paper_assets/tables/table_localization_parity.csv")
    parser.add_argument("--figure", type=Path, default=ROOT / "image/localization_parity_grid.png")
    add_force_flag(parser)
    args = parser.parse_args()
    json_out = args.table.with_suffix(".json")
    refuse_overwrite([args.table, json_out, args.figure], args.force)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    t0 = time.time()
    model, diffusion, n_params = build_model(device)
    print(f"model built ({n_params} params) on {device} in {time.time()-t0:.1f}s")

    images = select_images(args.n)
    if len(images) < args.n:
        raise SystemExit(f"only resolved {len(images)}/{args.n} images on disk")
    methods = ["S0", "S1", "S2"]
    mk = dict(y=torch.zeros(1, device=device, dtype=torch.long), p_uncond=-1, clf_free=True)
    outputs = ("encrypted", "restored", "healthy", "heatmap")

    rows = []
    grid_imgs = []   # for the figure: per image, original + per-method restored/healthy/heatmap
    for name, path in images:
        x0 = load_image(path, device)
        t_img = time.time()
        res = {}
        for m in methods:
            cipher = make_cipher(m, device)
            res[m] = run_one(diffusion, model_fn=make_model_fn(model), x0=x0, cipher=cipher, mk=mk)
            for key in outputs:
                save_png(to_img(res[m][key]), args.out_dir / name / f"{m}_{key}.png",
                         colormap=(key == "heatmap"))
        save_png(to_img(x0), args.out_dir / name / "original.png")

        # recovery fidelity vs original, and cross-method parity (S2 vs S0, S2 vs S1, S0 vs S1)
        x0i = to_img(x0)
        for m in methods:
            ri = to_img(res[m]["restored"])
            rows.append(dict(image=name, kind="recovery_vs_x0", method=m, baseline="x0",
                             psnr_db=round(psnr(x0i, ri), 4), ssim=try_ssim(x0i, ri),
                             linf=round(float(np.max(np.abs(x0i - ri))), 6),
                             mae=round(float(np.mean(np.abs(x0i - ri))), 6)))
        for key in ("restored", "healthy", "heatmap"):
            imgs = {m: to_img(res[m][key]) for m in methods}
            for a, b in (("S2", "S0"), ("S2", "S1"), ("S0", "S1")):
                rows.append(dict(image=name, kind=f"parity_{key}", method=a, baseline=b,
                                 psnr_db=round(psnr(imgs[a], imgs[b]), 4),
                                 ssim=try_ssim(imgs[a], imgs[b]),
                                 linf=round(float(np.max(np.abs(imgs[a] - imgs[b]))), 6),
                                 mae=round(float(np.mean(np.abs(imgs[a] - imgs[b]))), 6)))
        grid_imgs.append((name, x0i, {m: {k: to_img(res[m][k]) for k in ("restored", "healthy", "heatmap")} for m in methods}))
        print(f"  {name}: done in {time.time()-t_img:.1f}s")

    # write table + json
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    def agg(kind, key):
        vals = [r["psnr_db"] for r in rows if r["kind"] == kind and r["method"] == key and r["baseline"] != "x0"]
        fin = [v for v in vals if math.isfinite(v)]
        return dict(n=len(vals), psnr_min=round(min(fin), 3) if fin else None,
                    psnr_mean=round(float(np.mean(fin)), 3) if fin else None,
                    n_inf=sum(1 for v in vals if not math.isfinite(v)))
    summary = {
        "schema_version": 1, "task_id": "Q11", "seed": SEED, "n_images": len(images),
        "noise_level": NOISE_LEVEL, "guidance_scale": GUIDANCE, "dim": DIM, "C_norm": C_NORM,
        "fixture": "d7_cpu_security.FIXTURE_KEY (public reproducibility fixture; not key material)",
        "parity_restored_S2_vs_S0": agg("parity_restored", "S2"),
        "parity_healthy_S2_vs_S0": agg("parity_healthy", "S2"),
        "note": (
            "Parity demo (NOT a localization-accuracy benchmark; no ground-truth masks). "
            "All three ciphers share a fixed-norm + range-preserving envelope; only the "
            "orthogonal core differs. High cross-method PSNR on restored/healthy/heatmap => "
            "swapping the encryption layer is transparent to the downstream localization stage."
        ),
    }
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # figure: rows = images; cols = original | {S0,S1,S2} x {restored, healthy, heatmap}
    ncol = 1 + 3 * 3
    fig, axes = plt.subplots(len(grid_imgs), ncol, figsize=(ncol * 1.5, len(grid_imgs) * 1.6))
    axes = np.atleast_2d(axes)
    col_titles = ["original"] + [f"{m}\n{k}" for k in ("restored", "healthy", "heatmap") for m in methods]
    for r, (name, x0i, per) in enumerate(grid_imgs):
        tiles = [x0i] + [per[m][k] for k in ("restored", "healthy", "heatmap") for m in methods]
        cmaps = ["gray"] + ["gray"] * 6 + ["jet"] * 3
        for c, (tile, cmap) in enumerate(zip(tiles, cmaps)):
            axes[r, c].imshow(tile, cmap=cmap); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=6)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170); plt.close(fig)

    print(f"wrote {_rel(args.table)} ({len(rows)} rows); "
          f"restored S2vsS0 {summary['parity_restored_S2_vs_S0']}; "
          f"healthy S2vsS0 {summary['parity_healthy_S2_vs_S0']}; "
          f"fig {_rel(args.figure)}; total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
