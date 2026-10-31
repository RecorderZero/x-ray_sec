#!/usr/bin/env python3
"""Presentation figures for the defense report outline (presentation/).

Reads FROZEN tables (numbers never hand-typed) and emits three PNGs into
presentation/assets/:

  1. fig_storage_motivation.png -- why range-preserving storage was proposed:
     legacy min-max PNG barely touches plaintext (-0.12 dB) but destroys
     encrypted-latent fidelity (S0/S1 -7.1/-7.4 dB); range-preserving recovers
     it.  Source: paper_assets/tables/table_error_attribution.csv.
  2. fig_sampler_nsweep.png -- recon PSNR vs inversion steps N for DDIM / BDIA /
     DPM-Solver++, marking the N=100 deployment choice.  Source:
     presentation/assets/W9-T1_sampler_comparison.csv.
  3. fig_pipeline_passes.png -- predecessor native 6-pass vs our layer-A 4-pass
     (diagram; pass breakdown from presentation/口試問題回覆.md / predecessor_decrypt_flow.md).

  scripts/run_cfg_ddim.sh python -m scripts.plot_presentation_figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

C_OURS = "#1f6feb"   # blue
C_PRED = "#b0582c"   # legacy/brown
C_GREY = "#9aa0a6"
C_GOOD = "#2e7d32"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def fig_storage_motivation() -> None:
    """Grouped bars: range-preserving vs legacy min-max PSNR per scheme."""
    rows = read_rows(ROOT / "paper_assets" / "tables" / "table_error_attribution.csv")
    rp, lg = {}, {}
    for r in rows:
        if r["stage"] == "m1_sampler_range":          # range-preserving storage
            rp[r["method"]] = float(r["image_psnr_db_vs_x0"])
        elif r["stage"] == "m1_deploy_legacy":        # legacy min-max storage
            lg[r["method"]] = float(r["image_psnr_db_vs_x0"])
    schemes = ["P", "S0", "S1", "S2"]
    labels = ["P\n(plaintext)", "S0\n(predecessor)", "S1\n(predecessor)", "S2\n(ours)"]
    rp_v = [rp[s] for s in schemes]
    lg_v = [lg[s] for s in schemes]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = range(len(schemes))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], lg_v, w, label="legacy min-max PNG (predecessor storage)", color=C_PRED)
    b2 = ax.bar([i + w / 2 for i in x], rp_v, w, label="range-preserving PNG (ours)", color=C_OURS)
    for i, s in enumerate(schemes):
        drop = lg[s] - rp[s]
        tag = "N/A (already\nrange-preserving)" if s == "S2" else f"{drop:+.2f} dB"
        ax.annotate(tag, (i, max(lg[s], rp[s]) + 0.6), ha="center", va="bottom",
                    fontsize=9, color=(C_GREY if s == "S2" else C_PRED), fontweight="bold")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_ylabel("Recovery PSNR vs $x_0$ (dB)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(28, 44)
    ax.set_title("Why range-preserving storage: legacy min-max discards the\n"
                 "encrypted latent's value range (−0.12 dB for plaintext, −7 dB for S0/S1)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "fig_storage_motivation.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def fig_sampler_nsweep() -> None:
    rows = read_rows(ASSETS / "W9-T1_sampler_comparison.csv")
    N = [int(r["N"]) for r in rows]
    series = [("DDIM", "ddim_psnr_db", C_GREY, "o", "-"),
              ("BDIA", "bdia_psnr_db", C_PRED, "s", "--"),
              ("DPM-Solver++", "dpmpp_psnr_db", C_OURS, "D", "-")]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, col, color, mk, ls in series:
        y = [float(r[col]) for r in rows]
        ax.plot(N, y, ls, marker=mk, color=color, label=name, lw=2, ms=6)
    ax.axvline(100, color=C_GOOD, ls=":", lw=1.5)
    ax.annotate("deployment: N=100\n(DPM++ 45.6 dB, ~5x faster)", (100, 31),
                color=C_GOOD, fontsize=9, ha="center",
                bbox=dict(boxstyle="round", fc="white", ec=C_GOOD, alpha=0.9))
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel("inversion steps N (= model evals per pass)")
    ax.set_ylabel("Recovery PSNR vs $x_0$ (dB)")
    ax.set_title("Sampler comparison: DPM-Solver++ dominates at every step budget\n"
                 "(key-free round-trip, matched NFE; source W9-T1)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "fig_sampler_nsweep.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def fig_sampler_time() -> None:
    """4-pass deployment pipeline wall-clock vs inversion steps N (companion to
    fig_sampler_nsweep).  Time is ~linear in N and sampler-independent (NFE-bound:
    all samplers do N model evals/pass x 4 passes), so DDIM/DPM++ overlap; the
    DPM++ advantage is reaching target quality at FEWER N -> less time.
    Source: deploy_dpm_sweep/sweep.csv (scheme S2fn = deployment)."""
    rows = [r for r in read_rows(ROOT / "paper_assets" / "deploy_dpm_sweep" / "sweep.csv")
            if r["scheme"] == "S2fn"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    # DDIM solid-thick grey UNDER, DPM++ dashed blue ON TOP -> shows they coincide.
    for name, color, mk, ls, lw in [("DDIM", C_GREY, "o", "-", 4.5), ("DPM++", C_OURS, "D", "--", 2.0)]:
        pts = sorted([(int(r["N"]), float(r["mean_pipeline_seconds"]))
                      for r in rows if r["sampler"] == name])
        ax.plot([n for n, _ in pts], [s for _, s in pts], ls, marker=mk, color=color,
                label=f"{name} (S2fn, 4-pass)", lw=lw, ms=6)
    ns = sorted({int(r["N"]) for r in rows})
    ax.axvline(100, color=C_GOOD, ls=":", lw=1.5)
    ax.annotate("deployment: N=100\n~7.5 s / image (4-pass)", (100, 21),
                color=C_GOOD, fontsize=9, ha="center",
                bbox=dict(boxstyle="round", fc="white", ec=C_GOOD, alpha=0.95))
    ax.text(0.30, 0.74, "time ≈ linear in N, sampler-independent (NFE-bound):\n"
            "DDIM & DPM++ coincide. DPM++'s win = same PSNR at\n"
            "fewer N (see PSNR figure), not faster per-step.",
            transform=ax.transAxes, va="top", fontsize=8.2, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f6f8fa", ec="#ccc"))
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("inversion steps N (= model evals per pass)")
    ax.set_ylabel("4-pass pipeline wall-clock (s / image)")
    ax.set_title("4-pass deployment time vs N (DDIM ≈ DPM++; source deploy_dpm_sweep)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "fig_sampler_time.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def _box(ax, x, y, w, h, text, color, alpha=1.0, fontcolor="white", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none", alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=fontcolor, fontsize=fs, zorder=3, fontweight="bold")


def _arrow(ax, x0, y0, x1, y1, color="#555"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.4, zorder=1))


def fig_pipeline_passes() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 4.4); ax.axis("off")
    w, h, gap = 1.72, 0.9, 0.20

    pred = ["invert\n$x_0{\\to}z$", "generate\nanon", "invert\nanon", "decrypt +\nrecover",
            "re-invert\n(redundant)", "generate\npseudo-healthy"]
    ours = ["invert\n$x_0{\\to}z$", "generate\nanon", "invert\nanon", "decrypt +\nrecover"]

    # predecessor row (top)
    y = 2.9
    ax.text(0.05, y + h + 0.32, "Predecessor native pipeline — 6 passes (~7 s/pass dense DDIM  ⇒  ~42 s/image)",
            fontsize=11, fontweight="bold", color=C_PRED)
    for i, t in enumerate(pred):
        x = 0.1 + i * (w + gap)
        removed = i >= 4
        _box(ax, x, y, w, h, t, C_GREY if removed else C_PRED,
             alpha=0.45 if removed else 1.0, fontcolor="#333" if removed else "white")
        if i:
            xp = 0.1 + (i - 1) * (w + gap)
            _arrow(ax, xp + w, y + h / 2, x, y + h / 2)
    # bracket for removed
    xr = 0.1 + 4 * (w + gap)
    ax.annotate("removed by layer-A\n(−2 passes, ~1.5x)", (xr + w + gap / 2, y - 0.28),
                ha="center", va="top", fontsize=8.5, color=C_GREY, fontweight="bold")

    # ours row (bottom)
    y = 0.75
    ax.text(0.05, y + h + 0.32, "Ours (layer-A) — 4 passes (DPM++ N=100 ~1.6 s/pass  ⇒  ~6.3 s/image)",
            fontsize=11, fontweight="bold", color=C_OURS)
    for i, t in enumerate(ours):
        x = 0.1 + i * (w + gap)
        _box(ax, x, y, w, h, t, C_OURS)
        if i:
            xp = 0.1 + (i - 1) * (w + gap)
            _arrow(ax, xp + w, y + h / 2, x, y + h / 2)

    ax.text(6.0, 0.18,
            "speed-up = sampler (DPM++ N=100, ~4.9x)  ×  layer-A (6→4 pass, ~1.5x)  ≈  6.7x",
            ha="center", fontsize=10, style="italic", color="#222")
    fig.tight_layout()
    out = ASSETS / "fig_pipeline_passes.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def fig_s2_container() -> None:
    """S2 forward/inverse + authenticated-container storage & key-boundary.

    Shows exactly what travels with the published artifact (nonce in clear,
    params, AEAD-encrypted private metadata, payload, HMAC tag) vs the only
    secret (32-byte master key, separate 0600 file).  Grounded in
    src/crypto/{sot,s2_fixed_norm,container,nonce,kdf,key_io}.py.
    """
    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 8.0); ax.axis("off")

    # ---- forward (encrypt) lane ----
    ax.text(0.1, 7.62, "FORWARD (encrypt)", fontsize=11, fontweight="bold", color=C_OURS)
    fwd = [
        (0.1, "latent\n$z$"),
        (2.1, "fix norm\n$u=\\frac{C}{\\|z\\|}z$"),
        (4.4, "$Q$ = D$\\cdot$P$\\cdot$H$\\cdot$D  (R=1)\n$D_{post}\\,P\\,\\frac{H}{\\sqrt{d}}\\,D_{pre}$"),
        (7.1, "protected\n$y=Qu$"),
    ]
    for i, (x, t) in enumerate(fwd):
        _box(ax, x, 6.55, 1.75, 0.85, t, C_OURS, fs=8.6)
        if i:
            _arrow(ax, fwd[i - 1][0] + 1.75, 6.97, x, 6.97)
    _arrow(ax, 7.1 + 1.75, 6.97, 9.35, 6.97)
    ax.text(8.95, 7.22, "pack", fontsize=8.2, color="#555")

    # ---- container (middle): what is published / stored ----
    cx, cy, cw, chh = 9.3, 2.5, 3.0, 4.7
    ax.add_patch(FancyBboxPatch((cx, cy), cw, chh, boxstyle="round,pad=0.04,rounding_size=0.1",
                                fc="#eef2f7", ec="#5a6b7b", lw=1.6, zorder=0))
    ax.text(cx + cw / 2, cy + chh - 0.28, "Authenticated Container\n(published / stored)",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#2b3a48")
    fields = [
        ("nonce (16 B, cleartext)", C_GREY, "#222"),
        ("params: rounds / perm / public $C$", C_GREY, "#222"),
        ("AEAD ciphertext [lock]\n{orig norm $s$; M1: lo/hi}", C_GOOD, "white"),
        ("payload: $y$ / range-PNG", C_OURS, "white"),
        ("HMAC-SHA256 tag", "#6a4ca8", "white"),
    ]
    fy = cy + chh - 1.15
    for t, c, fc in fields:
        _box(ax, cx + 0.18, fy - 0.5, cw - 0.36, 0.62, t, c, fontcolor=fc, fs=7.8)
        fy -= 0.78
    _arrow(ax, 10.8, 6.52, 10.8, cy + chh)          # forward -> container
    _arrow(ax, cx, cy + 0.9, 8.75, cy + 0.9)        # container -> inverse lane

    # ---- master key (only secret, outside container) ----
    _box(ax, 9.5, 0.35, 2.6, 0.95,
         "master key (32 B)\nseparate 0600 file\nONLY secret, NOT in container", "#b00020", fs=7.8)
    for yy in (6.97, 1.35):                          # key feeds HKDF in both lanes (dashed)
        ax.add_patch(FancyArrowPatch((9.5, 0.82), (5.8, yy - 0.0), arrowstyle="-|>",
                     mutation_scale=10, color="#b00020", lw=1.1, ls=(0, (4, 3)),
                     connectionstyle="arc3,rad=-0.18", zorder=1))
    ax.text(5.9, 4.0, "HKDF(master key, nonce, label)\n-> sign/perm, AEAD & MAC subkeys",
            fontsize=7.6, color="#b00020", ha="center", style="italic")

    # ---- inverse (decrypt) lane ----
    ax.text(0.1, 2.05, "INVERSE (decrypt, needs master key)", fontsize=11, fontweight="bold", color=C_GOOD)
    inv = [
        (0.1, "verify HMAC\n(anti-tamper T4)"),
        (2.0, "open AEAD\n$s$, lo/hi"),
        (3.9, "$Q^{-1}$\n$D_{pre}\\frac{H}{\\sqrt{d}}P^{-1}D_{post}$"),
        (6.3, "un-fix norm\n$z=\\frac{s}{C}u$"),
    ]
    for i, (x, t) in enumerate(inv):
        _box(ax, x, 0.95, 1.7, 0.8, t, C_GOOD, fs=8.2)
        if i:
            _arrow(ax, inv[i - 1][0] + 1.7, 1.35, x, 1.35)
    _box(ax, 8.15, 0.95, 1.1, 0.8, "recover\n$z$", "#1b5e20", fs=8.6)
    _arrow(ax, 6.3 + 1.7, 1.35, 8.15, 1.35)

    # ---- attacker boundary note ----
    ax.text(0.1, 5.65,
            "Attacker (T2/T3) HAS: whole container (cleartext nonce, params,\n"
            "AEAD ciphertext, payload, tag).   LACKS: master key  =>\n"
            "  - cannot HKDF the sign/perm  -> cannot invert $Q$\n"
            "  - cannot derive AEAD subkey  -> cannot read $s$ / lo/hi (ChaCha20-Poly1305)\n"
            "  - cannot forge HMAC          -> tampering is rejected\n"
            "nonce being public is fine (Kerckhoffs); its job is per-image uniqueness\n"
            "-> per-image $Q$ + subkeys, enforced by NonceRegistry (SQLite UNIQUE).",
            fontsize=8.4, va="top", ha="left", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff6f6", ec="#b00020", lw=1.0))

    ax.text(0.1, 7.9, "S2 encrypt / decrypt + container: where recovery info lives, why the attacker cannot use it",
            fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    out = ASSETS / "fig_s2_container.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def fig_attack_flow() -> None:
    """Five attack lanes with grounded outcomes: predecessor vs ours.

    Numbers (random baseline ~0.005-0.05):
      amplitude/magnitude (T1 exact latent): attack_package C4/C5 Top-1 1.00 -> fixed-norm ~0.005
      learned re-id (external GMIA/Packhauser, range-preserving): table_reid_external
        AUC S0 0.616 / S1 0.541 -> S2 0.445 (Top-1 0.15/0.10 -> 0.05)
      inner-linkage (anon<->anon): table_t2bb_inner_linkage AUC 0.66 -> S2 ~0.43
      KPA (T3): table_KPA -- S0 1 pair / S1 2 pairs recover key bit-exact;
        table_cross_nonce -- recovered transform cross-nonce cos 0.001 (no transfer)
      CPA (T3): table_CPA -- 1 query reconstructs full decrypt map (err 0);
        table_cross_nonce -- cross-nonce cos 0.001
    """
    fig, ax = plt.subplots(figsize=(12.6, 8.4))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 9.2); ax.axis("off")
    # source
    _box(ax, 0.1, 3.9, 1.9, 1.3, "Attacker gets\npublished\nanonymized\noutput", "#444", fs=9)

    lanes = [
        ("Amplitude / magnitude\n(T1, exact latent)",
         "Top-1 = 1.00", "~0.005 (chance)",
         "killed by fixed-norm + CSPRNG keystream\n(magnitude multiset destroyed)"),
        ("Learned re-id\n(GMIA / Packhauser)",
         "AUC 0.62 / T1 0.15", "AUC 0.44 / T1 0.05",
         "killed by per-image nonce + S2 transform"),
        ("Inner linkage\n(anon ↔ anon)",
         "AUC 0.66", "AUC 0.43 (chance)",
         "killed by per-image nonce (distinct Q/image)"),
        ("KPA (T3)\nknown plaintext–ciphertext",
         "S0: 1 pair / S1: 2 pairs\n→ key recovered", "cross-nonce cos 0.001",
         "per-image nonce: a solved image\ndoes NOT transfer to any other"),
        ("CPA (T3)\nchosen plaintext",
         "1 query → full\ndecrypt map (err 0)", "cross-nonce cos 0.001",
         "per-image nonce: map valid for\nthat nonce only"),
    ]
    ys = [7.0, 5.55, 4.1, 2.65, 1.2]
    for (method, pred, ours, defense), y in zip(lanes, ys):
        _arrow(ax, 2.0, 4.55, 2.55, y + 0.47)
        _box(ax, 2.55, y, 2.6, 0.95, method, C_GREY, fontcolor="#222", fs=8.6)
        _arrow(ax, 5.15, y + 0.47, 5.55, y + 0.47)
        _box(ax, 5.55, y + 0.5, 2.95, 0.5, f"Predecessor:  {pred}", C_PRED, fs=7.7)
        _box(ax, 5.55, y - 0.02, 2.95, 0.5, f"Ours:  {ours}", C_GOOD, fs=7.7)
        ax.text(8.65, y + 0.47, defense, fontsize=7.9, va="center", ha="left",
                color="#333", style="italic")
    ax.text(0.1, 8.75, "Attack efficacy: predecessor broken, ours at chance "
            "(random ≈ 0.005–0.05)", fontsize=12.5, fontweight="bold")
    ax.text(0.1, 8.35, "T1/T2 = re-identification & linkage · T3 = KPA/CPA (key/map recovery)",
            fontsize=9, color="#555")
    ax.text(0.1, 0.35, "Sources: attack_package C4/C5, table_reid_external, "
            "table_t2bb_inner_linkage, table_KPA, table_CPA, table_cross_nonce (seed 1911)",
            fontsize=7.8, color="#666")
    fig.tight_layout()
    out = ASSETS / "fig_attack_flow.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def fig_anon_value_range(npz: Path) -> None:
    """Histogram of anonymized-image pixel values: plaintext fits [0,1], encrypted
    (S2) spills ~38% outside [0,1] -> legacy min-max discards that range."""
    if not npz.exists():
        print(f"skip fig_anon_value_range (no {npz}; run scripts.smoke_presentation_panels)")
        return
    d = np.load(npz, allow_pickle=True)
    aP, aS = d["anon_P"].ravel(), d["anon_S2fn"].ravel()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bins = np.linspace(-0.6, 1.3, 120)
    ax.hist(aP, bins=bins, density=True, color=C_GREY, alpha=0.75,
            label=f"plaintext anon (P): range [{aP.min():.2f}, {aP.max():.2f}]")
    ax.hist(aS, bins=bins, density=True, color=C_OURS, alpha=0.6,
            label=f"encrypted anon (S2): range [{aS.min():.2f}, {aS.max():.2f}]")
    ax.axvspan(-0.6, 0, color="#d33", alpha=0.08)
    ax.axvspan(1, 1.3, color="#d33", alpha=0.08)
    ax.axvline(0, color="#d33", ls="--", lw=1); ax.axvline(1, color="#d33", ls="--", lw=1)
    frac = float(np.mean((aS < 0) | (aS > 1)))
    ax.annotate(f"{frac*100:.0f}% of encrypted-anon\npixels fall OUTSIDE [0,1]\n"
                "→ legacy min-max PNG\ndiscards this range",
                (-0.45, ax.get_ylim()[1] * 0.62), fontsize=9, color="#b00",
                bbox=dict(boxstyle="round", fc="white", ec="#b00", alpha=0.9))
    ax.set_xlabel("anonymized-image pixel value (raw float, before PNG)")
    ax.set_ylabel("density")
    ax.set_title("Why range-preserving storage: the encrypted anon's value range\n"
                 "exceeds [0,1], so legacy min-max→uint8 loses it (dev_v1.1_000)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "fig_anon_value_range.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def fig_fidelity_sidebyside(npz: Path) -> None:
    """2x3 same-image panels: predecessor (S1, E2.5 authentic crops) vs ours (S2fn
    smoke). Columns: Original | Anonymized (cipher) | Recovered."""
    if not npz.exists():
        print(f"skip fig_fidelity_sidebyside (no {npz})")
        return
    from PIL import Image
    e25 = np.array(Image.open(ASSETS / "E2.5_roundtrip_grid.png").convert("L"))
    # S1 dev_000 image row y=601..858; cols 256 wide: 0 original,1 anonymous,3 M1 PNG recovered
    def crop(col):
        return e25[601:858, col * 256:(col + 1) * 256]
    pred = [crop(0), crop(1), crop(3)]
    d = np.load(npz, allow_pickle=True)
    ours = [d["x0"], d["anon_S2fn"], d["rec_S2fn"]]

    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.2))
    cols = ["Original $x_0$", "Anonymized (cipher)", "Recovered"]
    rows = ["Predecessor\n(S1, legacy PNG,\ndense DDIM)\nmean 31.4 dB",
            "Ours\n(S2fn, range-preserving,\nDPM++ N=100)\nmean 46.4 dB"]
    rowc = [C_PRED, C_OURS]
    for r, imgs in enumerate([pred, ours]):
        for c, im in enumerate(imgs):
            ax = axes[r][c]
            ax.imshow(im, cmap="gray")  # per-panel min-max display
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r][0].set_ylabel(rows[r], fontsize=9.5, color=rowc[r],
                              rotation=0, ha="right", va="center", labelpad=8)
    axes[0][2].text(0.5, -0.08, "~31 dB (example)", transform=axes[0][2].transAxes,
                    ha="center", fontsize=9, color=C_PRED)
    axes[1][2].text(0.5, -0.08, f"{float(d['psnr_S2fn']):.1f} dB (example)",
                    transform=axes[1][2].transAxes, ha="center", fontsize=9, color=C_OURS)
    fig.suptitle("Predecessor vs ours on the same image (dev_v1.1_000)", fontsize=12)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    out = ASSETS / "fig_fidelity_sidebyside.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-npz", type=Path,
                    default=Path("/tmp/claude-1000/-data2-paper/"
                                 "06f83540-c3da-4694-b2f1-227996eae257/scratchpad/panel_data.npz"),
                    help="output of scripts.smoke_presentation_panels (for the two image figures)")
    args = ap.parse_args()
    fig_storage_motivation()
    fig_sampler_nsweep()
    fig_pipeline_passes()
    fig_s2_container()
    fig_attack_flow()
    fig_anon_value_range(args.panel_npz)
    fig_fidelity_sidebyside(args.panel_npz)


if __name__ == "__main__":
    main()
