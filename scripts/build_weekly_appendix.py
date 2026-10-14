#!/usr/bin/env python3
"""Build the weekly paper appendix (figures, tables, MANIFEST) from committed results.

Every figure and table in ``appendix/week<N>/`` is produced here from files that
are tracked in Git; numbers are never typed by hand.  Each output is listed in
``appendix/week<N>/MANIFEST.csv`` with its SHA-256, the SHA-256 of every source
it was built from, the run IDs behind it, a draft caption and its intended use
in the paper.  See ``appendix/README.md`` for the conventions.

  scripts/run_cfg_ddim.sh python -m scripts.build_weekly_appendix --week 1
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scripts.output_guard import add_force_flag, refuse_overwrite

ROOT = Path(__file__).resolve().parents[1]

# Chart tokens (light mode, print); categorical slots 1-3 validated all-pairs
# (dataviz validate_palette.js: CVD dE 9.2, normal-vision dE 24.0).
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Item:
    item_id: str
    kind: str  # figure | table
    filename: str
    paper_use: str  # 正文主表 / 正文 / 附錄 / 僅內部
    section: str
    caption: str
    supports: str
    sources: list[str]
    run_ids: list[str] = field(default_factory=list)
    build: Callable[[Path, list[Path]], None] | None = None  # None -> copy sources[0]


# ---------------------------------------------------------------- styling
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- week 1 figures
def fig_storage_reversibility(out: Path, src: list[Path]) -> None:
    plt = _style()
    rows = [r for r in _read_csv(src[0]) if r["metric"] == "vs_x0_psnr_db"]
    schemes = ("P", "S0", "S1")
    storages = (("legacy_png", "legacy PNG (predecessor)"),
                ("range_preserving_png", "range-preserving PNG (ours)"),
                ("float", "float (upper bound)"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width, gap = 0.26, 0.02
    for s_index, (storage, label) in enumerate(storages):
        xs, means, lows, highs = [], [], [], []
        for g_index, scheme in enumerate(schemes):
            row = next(r for r in rows if r["scheme"] == scheme and r["storage"] == storage)
            mean = float(row["mean"])
            xs.append(g_index + (s_index - 1) * (width + gap))
            means.append(mean)
            lows.append(mean - float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]) - mean)
        ax.bar(xs, means, width=width, color=SLOT[s_index], label=label, zorder=2)
        ax.errorbar(xs, means, yerr=[lows, highs], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
    ax.set_xticks(range(len(schemes)), ["P (identity key)", "S0 Rademacher", "S1 Signed Perm."])
    ax.set_ylabel("M1 recovery PSNR vs x0 (dB)")
    ax.set_ylim(0, 45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8, handlelength=1.2, columnspacing=1.2)
    ax.set_title("Storage protocol, not the cipher, drives M1 reversibility (dev, n=20, 95% CI)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_pixel_cosine_vs_psnr(out: Path, src: list[Path]) -> None:
    plt = _style()
    data = json.loads(src[0].read_text(encoding="utf-8"))
    table = data["e_cosine_vs_psnr_table_unclipped_noise"]["rows"]
    psnr = np.array([row["achieved_psnr_db_mean"] for row in table])
    gap = 1 - np.array([row["cosine_mean"] for row in table])
    patients = data["a_different_patient_pixel_cosine"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhspan(1 - patients["p95"], 1 - patients["p05"], color=GRID, alpha=0.7, zorder=1, linewidth=0)
    ax.axhline(1 - patients["median"], color=INK_MUTED, linewidth=1, zorder=2)
    ax.text(10.4, (1 - patients["median"]) * 1.12,
            f"two different patients: median cosine {patients['median']:.3f} (p05–p95 band)",
            fontsize=8, color=INK_2)
    ax.plot(psnr, gap, color=SLOT[0], linewidth=2, marker="o", markersize=5, zorder=3,
            label="same image + Gaussian noise")
    ax.scatter([30.23], [1 - 0.998978], s=60, color=SLOT[1], marker="s", zorder=4,
               edgecolors="white", linewidths=1.5)
    ax.annotate("predecessor's reported\n'Cosine Sim 0.9989' (30.23 dB)", xy=(30.23, 1 - 0.998978),
                xytext=(31.5, 6e-3), fontsize=8, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8})
    ax.set_yscale("log")
    ax.set_xlabel("PSNR of the compared image pair (dB)")
    ax.set_ylabel("1 − pixel-space cosine (log)")
    ax.set_title("Pixel cosine saturates: 0.9989 only means about 30 dB (dev_v1.1, n=20)",
                 fontsize=10, color=INK)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_latent_vs_image(out: Path, src: list[Path]) -> None:
    plt = _style()
    legacy = _read_csv(src[0])
    ranged = _read_csv(src[1])
    panels = (("legacy PNG (predecessor)", legacy, "m1png_decrypted_latent_cosine", "m1png_vs_x0_psnr_db"),
              ("range-preserving PNG (ours)", ranged, "m1range_decrypted_latent_cosine", "m1range_vs_x0_psnr_db"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (title, rows, x_key, y_key) in zip(axes, panels):
        for index, scheme in enumerate(("P", "S0", "S1")):
            selected = [r for r in rows if r["scheme"] == scheme]
            ax.scatter([float(r[x_key]) for r in selected], [float(r[y_key]) for r in selected],
                       s=34, color=SLOT[index], marker=MARKERS[index], edgecolors="white",
                       linewidths=1.2, zorder=3, label=scheme)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("decrypted-latent cosine to z")
        ax.set_xlim(-0.15, 1.05)
    axes[0].set_ylabel("M1 recovery PSNR vs x0 (dB)")
    axes[0].legend(title="scheme", fontsize=8.5, title_fontsize=8.5, loc="lower left")
    fig.suptitle("Latent cosine does not predict image reversibility (dev, n=20 per scheme)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------- week 1 tables
def tab_environment(out: Path, src: list[Path]) -> None:
    env = json.loads(src[0].read_text(encoding="utf-8"))
    inventory = _read_csv(src[1])[0]
    rows = [
        ("python", env["python"].split(" ")[0]), ("pytorch", env["torch"]),
        ("cuda_build", env["torch_cuda_build"]), ("cudnn", env["cudnn"]),
        ("gpu", env["gpu_name"]), ("numpy", env["numpy"]), ("scipy", env["scipy"]),
        ("pillow", env["pillow"]), ("opencv", env["opencv"]),
        ("checkpoint_sha256", inventory["checkpoint_sha256"]),
        ("checkpoint_strict_load", inventory["strict_load"]),
        ("missing_unexpected_keys", f"{inventory['missing_key_count']}/{inventory['unexpected_key_count']}"),
        ("model_parameters", inventory["parameter_count"]),
        ("diffusion_timesteps", inventory["diffusion_timesteps"]),
    ]
    rows += [(f"model_{k}", v) for k, v in json.loads(inventory["model_config_json"]).items()]
    rows += [
        ("noise_level_T", "500 (DDIM, ddim1000 respacing)"),
        ("inversion_conditioning", "null=True (unconditional)"),
        ("generation_conditioning", "y=0 healthy class, single call, no CFG mixing (guidance_scale=-1)"),
        ("pipeline", "legacy_anonymization:ddim_sample_loop_anonymization"),
        ("batch_size_formal", "1 (bit-exact recomputation; see W1-T9)"),
        ("preprocessing", "grayscale; histogram equalization; INTER_AREA 256x256; JPEG q100; per-image min-max [0,1]"),
    ]
    _write_csv(out, [{"item": k, "value": v} for k, v in rows])


def tab_splits(out: Path, src: list[Path]) -> None:
    dev, formal, manifest = _read_csv(src[0]), _read_csv(src[1]), json.loads(src[2].read_text(encoding="utf-8"))

    def describe(name, rows, selection, seed, notes):
        labels = [r["label"] for r in rows]
        return {
            "split": name, "images": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "healthy_label0": labels.count("0"), "effusion_label1": labels.count("1"),
            "selection": selection, "seed": seed, "split_sha256": rows[0]["split_sha256"], "notes": notes,
        }

    _write_csv(out, [
        describe("dev_v1.1", dev, "CheXpert valid; first 10 patients per class by patient_id", "deterministic",
                 "dev/smoke only; hash excludes local_path"),
        describe("security_v1", formal, f"AF-012(a): legacy sample_n={manifest['legacy_sample_n_per_class']}/class "
                 f"random_state={manifest['legacy_random_state']} reproduced; excluded patients",
                 manifest["seed"], f"{manifest['excluded_patient_count']} training-sample patients excluded; "
                 "patient-disjoint; formal N=200 (user decision)"),
    ])


def tab_legacy_regression(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    rows = []
    for scheme in data["schemes"]:
        for key, value in scheme.items():
            if key.endswith("_max_abs"):
                rows.append({"scheme": scheme["scheme"], "check": key, "max_abs": value})
    _write_csv(out, rows)


def tab_latent_cache(out: Path, src: list[Path]) -> None:
    rows = []
    for path in src:
        data = json.loads(path.read_text(encoding="utf-8"))
        checks = data["checks"]
        rows.append({
            "split": Path(data["split"]).stem, "run_id": data["run_id"], "images": data["sample_count"],
            "latent_shape": "x".join(map(str, data["latent_shape"])), "batch_size": data["batch_size"],
            "recomputed": len(checks["recompute_sample_ids"]), "recompute_max_abs": checks["recompute_max_abs"],
            "recompute_bit_exact": checks["recompute_bit_exact_count"], "tolerance": checks["recompute_tolerance"],
            "finite": checks["finite_ok"], "status": data["status"], "cache_sha256": data["cache_file_sha256"],
        })
    _write_csv(out, rows)


def tab_pixel_cosine(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    a, c = data["a_different_patient_pixel_cosine"], data["c_different_patient_mean_removed_cosine"]
    b = data["b_same_image_plus_gaussian_noise"]["primary_exact_scaled_unclipped"]
    scale = data["d_cosine_one_without_equality"]["rescaling_y_equals_a_times_x"]["0.5"]["over_20_images"]
    rows = [
        {"comparison": "two different patients (190 pairs)", "psnr_db": a["pair_psnr_db"]["median"], "cosine": a["median"], "statistic": "median"},
        {"comparison": "two different patients, mean removed", "psnr_db": "", "cosine": c["median"], "statistic": "median"},
        {"comparison": "same image + noise", "psnr_db": 30.6, "cosine": b["30.6"]["cosine_mean"], "statistic": "mean of 20"},
        {"comparison": "same image + noise", "psnr_db": 26.2, "cosine": b["26.2"]["cosine_mean"], "statistic": "mean of 20"},
        {"comparison": "y = 0.5 x (not equal)", "psnr_db": scale["psnr_db_mean"], "cosine": scale["cosine_mean"], "statistic": "mean of 20"},
        {"comparison": "predecessor reported (base vs decrypt output)", "psnr_db": 30.23, "cosine": 0.998978, "statistic": "thesis 05_experiments.tex:275,308 (literature value)"},
    ]
    _write_csv(out, rows)


def tab_compute(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    rows = []
    for r in data["results"]:
        f, g, g0 = (r["estimated_forward_half_cycle_seconds_per_image"],
                    r["estimated_generation_half_cycle_seconds_per_image"],
                    r["estimated_generation_guidance0_half_cycle_seconds_per_image"])
        rows.append({"batch_size": r["batch_size"], "forward_half_cycle_s": round(f, 3),
                     "generation_half_cycle_s_guidance_m1": round(g, 3), "generation_half_cycle_s_guidance_0": round(g0, 3),
                     "full_cycle_s_guidance_m1": round(f + g, 3), "peak_vram_bytes": r["peak_vram_bytes"]})
    _write_csv(out, rows)


def tab_inference_flow(out: Path, src: list[Path]) -> None:
    rows = _read_csv(src[0])
    keys = ("image_psnr_db", "image_ssim", "latent_repeat_max_abs",
            "latent_reinversion_shared_noise_cosine", "latent_reinversion_unknown_noise_cosine",
            "runtime_seconds_primary_cycle")
    _write_csv(out, [{"metric": key, "mean_of_4": float(np.mean([float(r[key]) for r in rows])),
                      "min": float(np.min([float(r[key]) for r in rows])),
                      "max": float(np.max([float(r[key]) for r in rows]))} for key in keys])


# ---------------------------------------------------------------- week 2 tables
def tab_invariants(out: Path, src: list[Path]) -> None:
    """A3.3: the key-independent magnitude/norm invariants, one row per
    scheme x input kind, with the worst-case difference over all inputs and a
    negative control (magnitudes of a different latent)."""
    data = json.loads(src[0].read_text(encoding="utf-8"))
    summary = data["summary"]
    neg = {(n["input_kind"], n["scheme"]): n for n in data["negative_controls"]}
    rows = []
    for kind in ("synthetic", "real"):
        for scheme in ("S0", "S1"):
            s = summary[f"{kind}_{scheme}"]
            invariant = "|z| elementwise" if scheme == "S0" else "sorted |z|"
            diff_key = "max_abs_elementwise_max_diff" if scheme == "S0" else "max_sorted_abs_max_diff"
            control = neg[(kind, scheme)]
            ctrl_diff = control["abs_elementwise_max_diff" if scheme == "S0" else "sorted_abs_max_diff"]
            rows.append({
                "input_kind": kind, "scheme": scheme, "n_inputs": s["n_inputs"],
                "invariant": invariant,
                "invariant_max_abs_diff": s[diff_key],
                "norm_rel_diff_float64": s["max_norm_rel_diff_float64"],
                "norm_rel_diff_float32_torch": s["max_norm_rel_diff_float32_torch"],
                "roundtrip_max_abs_error": s["max_roundtrip_max_abs_error"],
                "negctrl_invariant_max_abs_diff": ctrl_diff,
            })
    _write_csv(out, rows)


# ---------------------------------------------------------------- week 3 tables
def _module_int_constant(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, int):
                return value
    raise ValueError(f"missing integer constant {name} in {path}")


def tab_d5_pipeline(out: Path, src: list[Path]) -> None:
    kdf, prng, nonce, container, mac, key_io = src
    master_bytes = _module_int_constant(kdf, "MASTER_KEY_BYTES")
    nonce_bytes = _module_int_constant(kdf, "NONCE_BYTES")
    tag_bytes = _module_int_constant(container, "TAG_BYTES")
    texts = {path.name: path.read_text(encoding="utf-8") for path in src}
    required = {
        "kdf.py": ("secrets.token_bytes", "hashes.SHA256", "hashlib.scrypt"),
        "prng.py": ("algorithms.ChaCha20", "_uniform_below", "Fisher-Yates"),
        "nonce.py": ("PRIMARY KEY(scope, nonce)", "secrets.token_bytes"),
        "container.py": ("sort_keys=True", "lo_float32_bits", "ciphertext"),
        "mac.py": ("hmac.HMAC", "open_authenticated", "authentication failed"),
        "key_io.py": ("os.O_EXCL", "0o600", "32 raw bytes"),
    }
    for filename, needles in required.items():
        for needle in needles:
            if needle not in texts[filename]:
                raise ValueError(f"D5 pipeline source assertion failed: {filename}: {needle}")
    rows = [
        {"task": "D5.1", "component": "master key / password mode",
         "implementation": f"OS CSPRNG {master_bytes} bytes ({master_bytes * 8} bits); explicit scrypt password mode",
         "security_property": "no integer-seed API; password entropy remains explicit"},
        {"task": "D5.2", "component": "domain-separated KDF",
         "implementation": f"HKDF-SHA256; {master_bytes}-byte subkeys; nonce used as salt",
         "security_property": "round/purpose/mac labels are separated"},
        {"task": "D5.3", "component": "signs / permutation CSPRNG",
         "implementation": "full HKDF subkey drives ChaCha20; rejection-sampling Fisher-Yates",
         "security_property": "no Python/NumPy/Torch non-cryptographic seed path"},
        {"task": "D5.4", "component": "nonce registry",
         "implementation": f"OS CSPRNG {nonce_bytes} bytes ({nonce_bytes * 8} bits); SQLite PRIMARY KEY(scope, nonce)",
         "security_property": "persistent atomic reuse rejection; raw master key is not stored"},
        {"task": "D5.5", "component": "authenticated container",
         "implementation": f"strict canonical JSON; HMAC tag {tag_bytes} bytes; M1 lo/hi stored as float32 bits",
         "security_property": "version, parameters, AAD, storage metadata and ciphertext bytes are covered"},
        {"task": "D5.6", "component": "authentication",
         "implementation": "HMAC-SHA256; verify-before-open",
         "security_property": "failed authentication never calls the plaintext opener"},
        {"task": "D5.7", "component": "key file",
         "implementation": f"raw {master_bytes}-byte key; mode 0600; O_EXCL",
         "security_property": "reject overwrite/trailing bytes; no pickle-capable loader"},
    ]
    _write_csv(out, rows)


def tab_d5_nonce(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    result = data["result"]
    _write_csv(out, [{
        "requested": result["requested"], "unique": result["unique"],
        "collisions": result["collisions"],
        "explicit_reuse_rejected": result["explicit_reuse_rejected"],
        "same_nonce_other_master_scope_accepted": result["same_nonce_other_master_scope_accepted"],
        "registry_rows": result["registry_rows"], "nonce_bytes": result["nonce_bytes"],
        "raw_nonce_or_key_recorded": result["raw_nonce_or_key_recorded"],
        "evidence_git_commit": data["git_commit"],
        "evidence_clean_tracked": not data["git_dirty_tracked_files"],
        "cryptography": data["cryptography"],
    }])


# ---------------------------------------------------------------- week 4 tables
def tab_d6_numerical_correctness(out: Path, src: list[Path]) -> None:
    inverse = json.loads(src[0].read_text(encoding="utf-8"))
    orthogonal = json.loads(src[1].read_text(encoding="utf-8"))
    rows = []
    for row in inverse["rows"]:
        rows.append({
            "task": "D6.1", "rounds": row["rounds"], "permutation": row["permutation"],
            "dimension": row["dimension"], "float64_max_abs": row["float64_max_abs"],
            "float64_norm_relative_error": row["float64_norm_relative_error"],
            "float32_max_abs": row["float32_max_abs"],
            "float32_bit_exact_rate": row["float32_bit_exact_rate"],
            "orthogonality_max_abs": "", "norm_error_spectral": "", "round_formula_max_abs": "",
        })
    rows.append({
        "task": "D6.2", "rounds": orthogonal["rounds"],
        "permutation": orthogonal["permutation"], "dimension": orthogonal["dimension"],
        "float64_max_abs": "", "float64_norm_relative_error": "", "float32_max_abs": "",
        "float32_bit_exact_rate": "", "orthogonality_max_abs": orthogonal["orthogonality_max_abs"],
        "norm_error_spectral": orthogonal["norm_error_spectral"],
        "round_formula_max_abs": orthogonal["round_formula_max_abs"],
    })
    _write_csv(out, rows)


def tab_d6_gaussian(out: Path, src: list[Path]) -> None:
    data = json.loads(src[0].read_text(encoding="utf-8"))
    rows = []
    for stage in ("before", "after"):
        values = data[stage]
        rows.append({"stage": stage, "vectors": data["vectors"], "dimension": data["dimension"], **values})
    _write_csv(out, rows)


def tab_d6_real_latent_shift(out: Path, src: list[Path]) -> None:
    rows = _read_csv(src[0])
    metrics = ("mean_shift", "variance_shift", "skew_shift", "excess_kurtosis_shift", "norm_ratio",
               "same_vector_cosine")
    summary = []
    for rounds in range(1, 5):
        for permutation in ("False", "True"):
            group = [r for r in rows if int(r["rounds"]) == rounds and r["permutation"] == permutation]
            item = {"rounds": rounds, "permutation": permutation, "n": len(group)}
            for metric in metrics:
                values = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_max_abs"] = float(np.abs(values).max())
            summary.append(item)
    _write_csv(out, summary)


def tab_d6_integration(out: Path, src: list[Path]) -> None:
    rows = _read_csv(src[0])
    metrics = ("m2_decrypted_latent_cosine", "m2_decrypted_latent_max_abs",
               "m1_decrypted_latent_cosine", "m1_vs_x0_psnr_db", "m1_vs_x0_ssim",
               "runtime_seconds_anonymous_generation", "runtime_seconds_m1_inversion",
               "runtime_seconds_m1_recovery_generation")
    summary = []
    for metric in metrics:
        values = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
        summary.append({"metric": metric, "n": len(values), "mean": float(values.mean()),
                        "min": float(values.min()), "max": float(values.max())})
    summary.append({"metric": "m1_authenticated_before_inversion", "n": len(rows),
                    "mean": float(np.mean([float(r["m1_authenticated_before_inversion"]) for r in rows])),
                    "min": "", "max": ""})
    _write_csv(out, summary)


# ---------------------------------------------------------------- week 5 figures
def fig_w5_norm_remediation(out: Path, src: list[Path]) -> None:
    """Compare the audited S2-v1 norm leak with the fixed-norm v2 rerun."""
    plt = _style()
    v1 = _read_csv(src[0])
    v2 = _read_csv(src[1])
    rows = [
        ("v1 T1", next(r for r in v1 if r["threat_model"] == "T1")),
        ("v1 T2-WB", next(r for r in v1 if r["threat_model"] == "T2-WB"
                          and r["storage"] == "range_preserving_png"
                          and r["calibration"] == "dev_median")),
        ("v2 T1", next(r for r in v2 if r["threat_model"] == "T1")),
        ("v2 T2-WB", next(r for r in v2 if r["threat_model"] == "T2-WB"
                          and r["calibration"] == "dev_median_to_C")),
    ]
    values = [float(row["top1"]) for _, row in rows]
    labels = [label for label, _ in rows]
    colors = [SLOT[1], SLOT[1], SLOT[0], SLOT[0]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(range(len(values)), values, color=colors, width=0.66, zorder=2)
    ax.axhline(0.005, color=INK_2, linewidth=1, linestyle="--", label="random (1/200)")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Norm-only gallery Top-1")
    ax.set_ylim(0, 1.08)
    ax.set_title("Fixed-norm v2 removes the measured norm-linkage signal (N=200)",
                 fontsize=10, color=INK)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------- registry
WEEKS: dict[int, list[Item]] = {
    1: [
        Item("W1-F1", "figure", "W1-F1_storage_reversibility.png", "正文", "Results／可逆性歸因",
             "M1 還原影像對原圖 x0 的 PSNR（dev_v1.1，每方案 20 張，95% bootstrap CI）。前作的 legacy PNG（per-image min-max、丟棄值域）使 S0/S1 只有約 31 dB；保存值域的 range-preserving PNG 讓 S0/S1 回到 38.4–39.2 dB，接近 float 交接上限；P 不受存檔方式影響。",
             "前作約 31 dB 的損失主要來自存檔流程，不是翻號／置換加密（AUD-20261010-05 裁決）",
             ["paper_assets/tables/table_m1_storage_comparison.csv"],
             ["E2.5_20261009T191319376995Z_f990c7d7", "AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8"],
             fig_storage_reversibility),
        Item("W1-F2", "figure", "W1-F2_pixel_cosine_vs_psnr.png", "正文", "Metric semantics／對前作指標的評析",
             "像素空間 cosine 與 PSNR 的關係（dev_v1.1 20 張加高斯雜訊，y 軸為 1−cosine 對數）。灰帶為不同病人兩兩 cosine 的 p05–p95。前作報告的 0.9989 對應約 30 dB，且不同病人的中位數已達 0.884。",
             "前作 0.9989 是輸出影像 cosine，只等於約 30 dB，不代表 latent 一致或零失真（AF-022）",
             ["results/E2.4_metric_semantics.json"], [], fig_pixel_cosine_vs_psnr),
        Item("W1-F3", "figure", "W1-F3_latent_vs_image_fidelity.png", "正文", "Results／指標選擇",
             "解密後 latent 與原 latent 的 cosine 對 M1 還原 PSNR（每點一張 dev 影像）。P 的 latent cosine 僅約 0.3 但影像 37.7 dB；S0/S1 在 legacy PNG 下 latent cosine 0.99 卻只有約 31 dB。latent 距離無法預測影像可逆性。",
             "可逆性必須以影像端指標判斷，latent 指標只作診斷（AF-019、AUD-20261010-02）",
             ["results/E2.5_roundtrip_per_sample.csv", "results/AF024_m1_storage_per_sample.csv"],
             ["E2.5_20261009T191319376995Z_f990c7d7", "AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8"],
             fig_latent_vs_image),
        Item("W1-F4", "figure", "W1-F4_roundtrip_grid.png", "正文", "Results／定性比較",
             "預先固定的 dev_v1.1_000（健康）與 _010（積水）在 P/S0/S1 下的原圖、匿名圖、T1 還原、legacy PNG M1 還原與固定色階 0–0.1 差異圖。",
             "S0/S1 匿名圖視覺上已打亂；legacy PNG 還原可見顆粒紋理與局部幻覺",
             ["image/E2.5_roundtrip_grid.png"], ["E2.5_20261009T191319376995Z_f990c7d7"]),
        Item("W1-F5", "figure", "W1-F5_P_anonymization_smoke.png", "附錄", "附錄／P 基線",
             "恆等金鑰 P 在學長加密流程上的 4 張 smoke：原圖、P 輸出、M1 float 與 PNG 還原、差異圖。可見還原圖出現原圖沒有的細小字樣狀紋理。",
             "P 基線走與 S0/S1 相同的流程（AF-017）；生成模型局部幻覺的觀察",
             ["image/AF017_P_anonymization_smoke.png"], ["AF017_P_SMOKE_20261009T152434977798Z_df76812e"]),
        Item("W1-F6", "figure", "W1-F6_inference_flow_smoke.png", "僅內部", "（不放論文；推論流程）",
             "推論流程（病灶定位用）4 張 smoke 的原圖、重建與差異圖。不是 P/S0/S1 比較的基線。",
             "僅作 E1.4 歷史紀錄",
             ["image/E1.4_ddim_smoke_contact_sheet.png"], ["E1.4_20261009T153121667310Z_aa04e79c"]),
        Item("W1-T1", "table", "W1-T1_environment_model.csv", "附錄", "Experimental setup",
             "實驗環境、checkpoint 與 diffusion 設定（含生成條件、T=500、batch 1、前處理）。",
             "實驗設定可重現（E1.1）", ["artifacts/environment_baseline.txt", "model_inventory.csv"], [], tab_environment),
        Item("W1-T2", "table", "W1-T2_splits.csv", "正文", "Experimental setup／資料",
             "dev_v1.1 與 security_v1 的樣本數、病人數、類別數、抽樣規則與 split hash。security_v1 排除了重現前作訓練抽樣的 22,253 位病人。",
             "評估資料與模型訓練資料 patient-disjoint（E1.3、E2.1）",
             ["splits/dev_v1.1.csv", "splits/security_v1.csv", "splits/security_v1_manifest.json"], [], tab_splits),
        Item("W1-T3", "table", "W1-T3_e0_canonical_preflight.csv", "附錄", "附錄／前置可行性",
             "E0 canonical synthetic preflight（SOT-WHT、CDF torus pad、KCI toy；d=4,096 與 65,536；N=100）。舊 artifact 已標為 historical。",
             "E0.1 evidence reset；WHT 保留 norm（norm-only Top-1=1.0）",
             ["canonical_preflight.csv"], []),
        Item("W1-T4", "table", "W1-T4_e2_2_legacy_regression.csv", "附錄", "附錄／前作重現",
             "E2.2：P/S0/S1 wrapper 與直接呼叫學長函式、guidance −1 與 0、deanonymize、transform round-trip 的 MaxAbs（T=500，皆為 0）。",
             "本專案以前作原始程式與參數逐位元重現前作流程（E2.2、AF-021）",
             ["results/E2.2.json"], [], tab_legacy_regression),
        Item("W1-T5", "table", "W1-T5_e2_3_latent_cache.csv", "附錄", "附錄／可重現性",
             "E2.3 latent cache：security_v1 200 張與 dev 20 張，batch 1 抽樣重算全部逐位元相同。batch 4 建 cache 的負結果（MaxAbs 6.93e-4）記錄於 reports/compute_budget.md。",
             "攻擊用的 latent cache 可逐位元重現；batch 組成會改變 GPU 結果",
             ["results/E2.3_DONE.json", "results/E2.3_dev_v1.1_cache.json"],
             ["E2.3_20261009T183736363217Z_84cf642a", "E2.3_20261009T190608331743Z_6edaf199"], tab_latent_cache),
        Item("W1-T6", "table", "W1-T6_e2_4_pixel_cosine_baselines.csv", "正文", "Metric semantics",
             "像素 cosine 的基準：不同病人、去均值、加雜訊、y=0.5x 反例與前作報告值。",
             "cosine=1 不等於相等；0.9989 只對應約 30 dB（E2.4、AF-022）",
             ["results/E2.4_metric_semantics.json"], [], tab_pixel_cosine),
        Item("W1-T7", "table", "W1-T7_e2_5_baseline_correctness.csv", "正文主表", "Results／前作可逆性基線",
             "E2.5 主表：P/S0/S1 的 transform-only、T1、M1（legacy PNG 與 float）影像與 latent 指標，平均與 95% bootstrap CI。",
             "前作在其實際流程下的可逆性（M1 legacy PNG 約 31 dB）",
             ["paper_assets/tables/table_baseline_correctness.csv"], ["E2.5_20261009T191319376995Z_f990c7d7"]),
        Item("W1-T8", "table", "W1-T8_m1_storage_comparison.csv", "正文", "Results／可逆性歸因",
             "AF-024：P/S0/S1 × legacy PNG／range-preserving PNG／float 的影像與 latent 指標（含攻擊者視角 |z| Pearson）。W1-F1 的數值表。",
             "存檔協定歸因；保存值域同時提升攻擊者保真度（AF-025）",
             ["paper_assets/tables/table_m1_storage_comparison.csv"],
             ["AF024_M1_STORAGE_20261010T023636939496Z_278c5ed8"]),
        Item("W1-T9", "table", "W1-T9_compute_benchmark.csv", "附錄", "附錄／計算成本",
             "學長加密流程每張影像的 inversion 與生成半週期時間（batch 1/4/8；guidance −1 與 0）。",
             "正式實驗算力預算的依據（AF-021）",
             ["results/AF021_anonymization_benchmark.json"], ["AF021_ANON_BENCH_20261009T152410677702Z_a5f3dcdc"], tab_compute),
        Item("W1-T10", "table", "W1-T10_e1_4_inference_flow.csv", "僅內部", "（不放論文；推論流程）",
             "推論流程 smoke 的影像指標與 re-inversion cosine：共用雜訊 0.906 為樂觀上限，未知雜訊 0.371。",
             "0.906 不能當可逆性證據（AF-019）",
             ["results/E1.4_ddim_smoke.csv"], ["E1.4_20261009T153121667310Z_aa04e79c"], tab_inference_flow),
    ],
    2: [
        # -------------------------------------------------------------- figures
        Item("W2-F1", "figure", "W2-F1_T1_magnitude_cmc.png", "正文", "Results／T1 linkage",
             "T1（精確 latent）magnitude gallery linkage 的 CMC 曲線（security_v1，200 查詢 × 200 gallery）。S0 以 |z|、S1 以 sorted|z| 的指定 scorer rank-1 即 1.00；random 基線 1/200。",
             "前作的 |z|／sorted|z| 不變量讓無金鑰 T1 linkage 在精確 latent 下達 Top-1=1.00（A3.4）",
             ["image/A3.4_T1_cmc.png"], []),
        Item("W2-F2", "figure", "W2-F2_attack_overview.png", "正文", "Method／攻擊示意",
             "攻擊面示意圖（固定 dev 病例）：原圖、S0／S1 匿名圖、以及同一人 vs 不同人的 sorted|z| 逐 rank 差（symlog）。說明幅值不變量如何提供無金鑰的鏈結線索。",
             "視覺化 A3.3 不變量與 A3.4／A4.2 linkage 的攻擊構想（P3.1）",
             ["image/P3.1_attack_overview.png"], []),
        Item("W2-F3", "figure", "W2-F3_T2WB_magnitude_cmc.png", "正文", "Results／T2-WB linkage",
             "T2-WB（重新反演匿名影像）magnitude gallery linkage 的 CMC（security_v1，200×200）。前作部署的 legacy PNG 下 S0 以 |z| rank-1=1.00、S1 以尺度不變 sorted|z| rank-1=0.465；range-preserving 參考欄 S1 達 0.995。",
             "即便攻擊者只拿到匿名影像重新反演的近似 latent，magnitude linkage 仍遠高於隨機（A4.2、AF-033）",
             ["image/A4.2_T2WB_cmc.png"],
             ["A4.1_20261012T083425685874Z_3132fa49"]),
        # -------------------------------------------------------------- tables
        Item("W2-T1", "table", "W2-T1_keyspace.csv", "正文", "Results／金鑰空間",
             "前作有效金鑰空間審計：S0 金鑰上界 32 bits、S1 64 bits（password 模式兩獨立 32-bit 種子）；已知一對明密文的完整配對搜尋為 0 bits；唯密文 seed 搜尋 S0≈2^32、S1≈2^33。每列附前作程式行號與證據標記。",
             "前作加密的有效金鑰遠小於其名目 keyspace，且已知明文時可直接讀出金鑰（A3.1、AF-026、AF-027）",
             ["paper_assets/tables/table_keyspace.csv", "reports/keyspace_method_note.md"], []),
        Item("W2-T2", "table", "W2-T2_invariants.csv", "正文", "Results／不變量",
             "S0／S1 的金鑰無關不變量（A3.3，合成與真實 latent 各 5 筆）：S0 的 |z| 逐座標、S1 的 sorted|z| 皆逐位元相等（最大差 0.0）；||z|| float64 相對誤差 ≤8.02e-16、float32 ≤3.68e-07；逆運算 round-trip 誤差 0.0；負控制（不同 latent 幅值）差值明顯非零。",
             "幅值／範數是金鑰無關的不變量，構成無金鑰攻擊面（A3.3）",
             ["results/A3.3.json"],
             [], tab_invariants),
        Item("W2-T3", "table", "W2-T3_T1_magnitude_gallery.csv", "正文主表", "Results／T1 linkage",
             "T1 magnitude gallery linkage 主表（security_v1，200×200）：P／S0／S1 × abs／sorted_abs／norm 的 Top-1（含 Wilson 95% CI）、Top-5、mAP、中位 rank 與 shuffle 負控制。W2-F1 的數值表。",
             "精確 latent 下，除 S1+abs 外所有指定 scorer 皆 Top-1=1.00，負控制 ≤0.01（A3.4）",
             ["paper_assets/tables/table_T1_magnitude.csv"], []),
        Item("W2-T4", "table", "W2-T4_T2WB_inversion.csv", "附錄", "Results／T2-WB 保真度",
             "T2-WB 攻擊者重新反演所得 latent 的保真度（security_v1，n=200，95% CI）：cosine、RMSE、max-abs、norm ratio、lowfreq32 cosine、|z| Pearson，兩種存檔欄並列。legacy PNG 下 cosine 0.990、range-preserving 0.9997。",
             "量化 T2-WB 攻擊者能取得的 latent 品質，作為 magnitude linkage 的前提（A4.1）",
             ["paper_assets/tables/table_T2_WB_inversion.csv"],
             ["A4.1_20261012T083425685874Z_3132fa49"]),
        Item("W2-T5", "table", "W2-T5_T2WB_magnitude_gallery.csv", "正文主表", "Results／T2-WB linkage",
             "T2-WB magnitude gallery linkage 主表（security_v1，200×200）：S0／S1 × 6 種 scorer（含 AF-033 尺度不變 abs_unit／sorted_abs_unit）× 兩種存檔的 Top-1、Top-5、mAP、負控制、校正後 norm-only 與正控制判定。前作部署＝legacy，range 作參考。",
             "T2-WB 下 S0 |z| Top-1=1.00、S1 尺度不變 sorted|z| legacy 0.465／range 0.995；校正 norm-only 僅 range 顯著（S0 0.255／S1 0.285）（A4.2、AF-033、AF-025）",
             ["paper_assets/tables/table_T2_WB_magnitude.csv"],
             ["A4.1_20261012T083425685874Z_3132fa49", "A4.1_DEV_CAL_20261012T133002259066Z_0fa671c3"]),
        Item("W2-T6", "table", "W2-T6_KPA.csv", "正文", "Results／KPA",
             "已知明文攻擊（A4.3，d=65536 真實 latent 3 筆 + d=4096 合成）：S0 單一對即逐位元還原金鑰並對 held-out 密文精確解密；S1 單一對因 float32 幅值重複（114–150 個並列座標）無法定序，兩對即精確還原。",
             "S0 一對、S1 兩對已知明密文即可完整還原金鑰（A4.3）",
             ["paper_assets/tables/table_KPA.csv"], []),
        Item("W2-T7", "table", "W2-T7_CPA.csv", "正文", "Results／CPA",
             "選擇明文攻擊（A4.4）：固定金鑰下，d=4096 全基底查詢可重建整個解密線性映射，但前作的逐座標結構使實際只需 1 次查詢；對全新明文的重建 max-abs 誤差 0.0（d=4096 與 d=65536 真實金鑰皆然）。",
             "固定金鑰下單次查詢即可重建解密映射，對新明文零誤差（A4.4）",
             ["paper_assets/tables/table_CPA.csv"], []),
        Item("W2-T8", "table", "W2-T8_tamper_baseline.csv", "正文", "Results／完整性",
             "竄改測試（A4.5，container／latent 層，S0／S1 × 5 種竄改 × 3 筆，共 30 列）：翻一個符號、加高斯雜訊、改存檔 hi、改存檔像素、legacy 存檔像素竄改，全部被接受（accepted=True），無任何拒絕。",
             "前作沒有任何完整性檢查，所有竄改都被接受（A4.5，T4）",
             ["paper_assets/tables/table_tamper_baseline.csv"], []),
        Item("W2-T9", "table", "W2-T9_tamper_deployment.csv", "附錄", "Results／完整性",
             "竄改測試（部署路徑）：匿名影像→legacy 存檔→DDIM 反演→逆金鑰→生成，翻一個存檔像素 +128。2 筆皆被接受、輸出有限、與未竄改部署不同（PSNR 49.8／53.5 dB）。補足 container/latent 層的端到端對照。",
             "端到端部署路徑同樣接受竄改、無拒絕（A4.5、AF-036）",
             ["paper_assets/tables/table_tamper_deployment.csv"],
             ["A4.5_DEPLOY_20261012T182847866867Z_acb4e5a1"]),
        Item("W2-T10", "table", "W2-T10_attack_package_claims.csv", "正文", "Method／攻擊素材包",
             "Track A 攻擊素材包（P4.1）：11 條攻擊 claim 與其威脅模型、一句話、headline 數字、來源檔 SHA-256 與 run ID。數字由 scripts/build_attack_package.py 從已追蹤結果檔讀取，不手抄。",
             "每條攻擊宣稱都對應可追溯的表、圖、result JSON 與 run ID（P4.1、AF-034）",
             ["paper_assets/attack_package/claims.csv", "paper_assets/attack_package/index.md"], []),
    ],
    3: [
        Item("W3-T1", "table", "W3-T1_d5_key_pipeline.csv", "正文", "Method／S2 key pipeline",
             "D5.1–D5.7 的密碼材料、nonce、authenticated container、HMAC 與 key-file 設計摘要。數值常數由實作原始碼 AST 讀取，安全屬性以來源字串斷言防止摘要與實作漂移。",
             "S2 不再使用前作的短 seed／非密碼學 PRNG；容器採驗證後才開啟的 fail-closed 路徑（AF-024、AF-030）",
             ["src/crypto/kdf.py", "src/crypto/prng.py", "src/crypto/nonce.py",
              "src/crypto/container.py", "src/crypto/mac.py", "src/crypto/key_io.py"], [], tab_d5_pipeline),
        Item("W3-T2", "table", "W3-T2_d5_nonce_registry.csv", "正文", "Results／nonce reuse",
             "D5.4 clean-HEAD nonce evidence：100,000 個 128-bit random nonce 全部唯一；顯式重用 fail closed；相同 nonce 在不同 master-key scope 可登記；artifact 不含 raw nonce／key。",
             "per-image nonce registry 具持久化 UNIQUE constraint，阻止同一 master-key scope 的 nonce 重用（D5.4）",
             ["results/D5.4_nonce_registry.json"], [], tab_d5_nonce),
        Item("W3-T3", "table", "W3-T3_d5_tamper_matrix.csv", "正文主表", "Results／完整性",
             "D5.6 clean-HEAD tamper matrix：未竄改容器接受且 opener 呼叫一次；wrong key 與 12 種內容／metadata 竄改全部拒絕，opener 不被呼叫，外部錯誤統一。",
             "authenticated container 對 key、tag、nonce、header、AAD、storage、lo/hi 與 ciphertext 竄改皆 fail closed（D5.6、AF-024）",
             ["paper_assets/tables/table_D5_tamper_matrix.csv", "results/D5.6_tamper_matrix.json"], []),
    ],
    4: [
        Item("W4-F1", "figure", "W4-F1_gaussian_distribution.png", "附錄", "Results／distribution diagnostic",
             "D6.3 合成 N(0,1) 向量經 SOT 前後的分布診斷（128×4,096 維）；此圖只檢查實作與理論相容，不是安全證明。",
             "SOT 對合成高斯輸入未造成可檢出的邊際分布偏離；不得外推到真實 latent 或安全性",
             ["image/D6.3_gaussian_distribution.png"], []),
        Item("W4-F2", "figure", "W4-F2_s2_dev_integration.png", "正文", "Results／S2 integration",
             "S2 固定 dev 四病例之原圖、匿名圖、M2/T1 與 M1 還原圖；R=1、permutation off、range-preserving M1。",
             "S2 匿名輸出已明顯改變，且 authenticated M1 路徑可回復高保真影像（D6.7）",
             ["image/D6.7_s2_dev.png"], ["D6.7_INTEGRATION_20261013T164724711647Z_ecec4bf5"]),
        Item("W4-F3", "figure", "W4-F3_method_comparison.png", "正文", "Results／five-method comparison",
             "固定積水病例 dev_v1.1_010 的 P、S0、S1、S2a、S2：原圖、匿名圖與重構圖。P/S0/S1 採前作 legacy 存檔；S2a/S2 採保存值域存檔。",
             "在指定存檔條件下比較五種方法的定性匿名化與重構；S2a/S2 的像素相同，差別是 S2 加入認證",
             ["image/method_comparison.png"], []),
        Item("W4-T1", "table", "W4-T1_numerical_correctness.csv", "附錄", "Method／SOT correctness",
             "D6.1 之 d=65,536、R=1–4、有無置換的 float64／float32 往返誤差，以及 D6.2 d=1,024 顯式矩陣正交性。",
             "FWHT/SOT 公式與逆運算通過數值正確性檢查；float32 不是 bit-exact",
             ["results/D6.1.json", "results/D6.2.json"], [], tab_d6_numerical_correctness),
        Item("W4-T2", "table", "W4-T2_gaussian_diagnostic.csv", "附錄", "Results／distribution diagnostic",
             "D6.3 合成高斯輸入變換前後的 mean、variance、skew、excess kurtosis 與 KS 檢定（128×4,096 維）。",
             "合成 Gaussian 診斷與理論相容，但不是安全證明",
             ["results/D6.3.json"], [], tab_d6_gaussian),
        Item("W4-T3", "table", "W4-T3_real_latent_shift.csv", "正文", "Results／real-latent diagnostics",
             "D6.4 dev_v1.1 20 張真實 latent 在 R=1–4、有無置換下的動差位移、norm ratio 與同向量 cosine 摘要。",
             "真實 latent 不可沿用 Gaussian 不變分布宣稱；正交變換仍保留 norm",
             ["results/D6.4.csv"], [], tab_d6_real_latent_shift),
        Item("W4-T4", "table", "W4-T4_rounds_permutation.csv", "正文主表", "Results／R/P ablation",
             "D6.5 dev_v1.1（n=20）之 R=1–4 × permutation on/off：original-/sorted-|z|、norm-only Top-k、CPU transform runtime 與往返誤差。",
             "R=1、permutation off 已使兩個 magnitude scorer 降至 dev random baseline；所有設定仍完整洩漏 norm",
             ["paper_assets/tables/table_rounds_permutation.csv"], []),
        Item("W4-T5", "table", "W4-T5_structure_diagnostic.csv", "附錄", "Results／structure diagnostic",
             "D6.6 trace、同向量 cosine、basis probe、sparsity與正交誤差等結構診斷；全部僅為診斷，不作安全證明。",
             "記錄 SOT 的結構性質與已知限制",
             ["paper_assets/tables/table_structure.csv"], []),
        Item("W4-T6", "table", "W4-T6_generation_quality.csv", "正文", "Results／anonymous-image quality",
             "D6.4 quality managed run：dev_v1.1 20 張 × 8 組 R/P，共 160 次匿名影像生成的 PSNR、SSIM、cosine 與 95% CI。",
             "所有 R/P 組合皆成功生成有限值影像；匿名圖與原圖差異大，但不宣稱臨床效用",
             ["paper_assets/tables/table_D6_generation_quality.csv"], ["D6.4_QUALITY_20261013T162058881536Z_c2c4465a"]),
        Item("W4-T7", "table", "W4-T7_s2_dev_integration.csv", "正文主表", "Results／S2 integration",
             "D6.7 固定四病例之 authenticated S2 整合摘要：M2 latent 往返、M1 re-inversion、影像重構與 runtime。",
             "S2 dev integration 的正確性、認證順序與重構保真度",
             ["results/D6.7_dev_integration.csv"], ["D6.7_INTEGRATION_20261013T164724711647Z_ecec4bf5"], tab_d6_integration),
        Item("W4-T8", "table", "W4-T8_method_comparison.csv", "正文", "Results／five-method comparison",
             "固定積水病例 dev_v1.1_010 的五方法影像 PSNR、SSIM、L∞、最大 uint8 pixel error、pixel equality、bit-exact 與三階段 runtime。",
             "五方法在指定存檔條件下的單病例定量比較；S2a/S2 相同說明 authentication 不改變像素結果",
             ["paper_assets/tables/table_method_comparison.csv", "results/method_comparison.json"], []),
    ],
    5: [
        Item("W5-F1", "figure", "W5-F1_d7_attack_summary.png", "正文", "Results／security evaluation",
             "D7 與 AF-025 的主要 gallery-linkage Top-1：S2 v1 的 magnitude、norm 與 geometry 結果，以及 fixed-norm v2 補驗。正式 security_v1 N=200；random baseline 為 1/200。",
             "S2 的 per-image transform 抑制已測 magnitude／geometry linkage，但 v1 保留 norm；fixed-norm v2 將指定 norm attack 降回隨機基準",
             ["image/D7_attack_summary.png"], []),
        Item("W5-F2", "figure", "W5-F2_norm_remediation.png", "正文", "Results／AF-025 remediation",
             "S2 v1 與 fixed-norm v2 的 norm-only gallery Top-1（security_v1 N=200）。T2-WB 採 dev-only calibration；虛線為 1/200 隨機基準。",
             "AF-025 的固定-norm 修補把本研究指定的 T1 與 T2-WB norm linkage 降至 0.005；此為特定攻擊實測，不外推為通用密碼學安全",
             ["paper_assets/tables/table_norm_leakage.csv",
              "paper_assets/tables/table_AF025_fixed_norm_norm_leakage.csv"],
             ["D7.1_T2WB_20261014T021454463115Z_7b959db9",
              "AF025_FIXED_NORM_FORMAL_20261014T161327001754Z_c4eda689"],
             fig_w5_norm_remediation),
        Item("W5-T1", "table", "W5-T1_d7_t2wb_inversion.csv", "正文", "Results／T2-WB inversion",
             "D7.1 formal N=200：S2 v1 匿名影像在 legacy 與 range-preserving storage 下的攻擊者重新反演 latent 品質、runtime 與 95% bootstrap CI。",
             "T2-WB query 的品質與正控制足以解釋後續 linkage 結果",
             ["paper_assets/tables/table_D7_T2WB_inversion.csv"],
             ["D7.1_T2WB_20261014T021454463115Z_7b959db9"]),
        Item("W5-T2", "table", "W5-T2_d7_magnitude_linkage.csv", "正文主表", "Results／magnitude linkage",
             "D7.2 formal N=200：S2 v1 在 T1／T2-WB、兩種 storage 與四種 magnitude scorer 下的 Top-k、mAP、rank、正負控制。",
             "指定 original／sorted magnitude linkage 在 S2 v1 上接近 1/200 隨機基準",
             ["paper_assets/tables/table_attack_original.csv"],
             ["D7.1_T2WB_20261014T021454463115Z_7b959db9"]),
        Item("W5-T3", "table", "W5-T3_d7_norm_leakage_v1.csv", "正文主表", "Results／AF-025 diagnosis",
             "D7.3 formal N=200：S2 v1 的 norm-only T1／T2-WB linkage；T2-WB 同時列 raw 與只由 dev 凍結的 median calibration。",
             "正交 S2 v1 保留 exact-latent norm，且 range-preserving T2-WB 經 dev calibration 後仍可鏈結，形成 AF-025 的裁決依據",
             ["paper_assets/tables/table_norm_leakage.csv"],
             ["D7.1_DEV_20261014T020638335355Z_9223ea81",
              "D7.1_T2WB_20261014T021454463115Z_7b959db9"]),
        Item("W5-T4", "table", "W5-T4_cross_nonce.csv", "附錄", "Results／cross-nonce KPA/CPA",
             "D7.4 d=1,024 synthetic diagnostic：same-nonce full-basis CPA／full-rank KPA 正控制，及相同映射跨 nonce 套用的 RMSE、MaxAbs 與 cosine。",
             "固定 nonce 的完整映射可重建，但不轉移至另一 nonce；此合成診斷不冒充 d=65,536 端到端證明",
             ["paper_assets/tables/table_cross_nonce.csv"], []),
        Item("W5-T5", "table", "W5-T5_geometry.csv", "正文", "Results／pairwise geometry",
             "D7.5 formal N=200：fixed-Q 與 per-image-Q 的 pairwise cosine／distance 誤差及 distance-signature linkage。",
             "fixed-Q 保留跨影像 geometry；per-image-Q 使指定 distance-signature linkage 降至隨機基準",
             ["paper_assets/tables/table_geometry.csv"], []),
        Item("W5-T6", "table", "W5-T6_structured_probe_v1.csv", "附錄", "Results／structured probes",
             "D7.6：basis、constant 與 sparse 非 Gaussian probe 的輸入／輸出稀疏度、動差、最大值與 norm ratio。",
             "結構探針補足 Gaussian 測試但僅作診斷；S2 v1 對所有 probe 保 norm",
             ["paper_assets/tables/table_structured_probe.csv"], []),
        Item("W5-T7", "table", "W5-T7_repeat_v1.csv", "附錄", "Results／repeat transform",
             "D7.7：同一真實 latent 以 100 個 nonce 重複變換；每列含 nonce／payload hash、全體唯一數與跨密文 cosine 摘要。",
             "每影像 nonce 讓 100 次 protected payload 全部不同",
             ["paper_assets/tables/table_repeat.csv"], []),
        Item("W5-T8", "table", "W5-T8_tamper.csv", "正文", "Results／integrity",
             "D7.8：S2 對 tag、nonce、ciphertext、version、parameters、AAD、storage 的竄改拒絕，以及 S2a 無認證正控制。",
             "S2 authenticated container fail closed；S2a 不提供完整性保護",
             ["paper_assets/tables/table_tamper.csv"], []),
        Item("W5-T9", "table", "W5-T9_fixed_norm_linkage.csv", "正文主表", "Results／AF-025 remediation",
             "AF-025 fixed-norm v2 formal N=200：T1 與 T2-WB norm-only raw／dev-calibrated Top-k、mAP、rank、負控制與固定公開 C。",
             "fixed-norm v2 將本研究指定的 T1／T2-WB norm-only Top-1 降至 1/200 隨機基準",
             ["paper_assets/tables/table_AF025_fixed_norm_norm_leakage.csv"],
             ["AF025_FIXED_NORM_FORMAL_20261014T161327001754Z_c4eda689"]),
        Item("W5-T10", "table", "W5-T10_fixed_norm_inversion.csv", "正文", "Results／AF-025 correctness and cost",
             "AF-025 fixed-norm v2 formal N=200：攻擊者 public-PNG norm、合法 private-range latent 重構品質，以及匿名生成／兩種 inversion runtime 與 95% CI。",
             "隱藏真實 norm／lo-hi 後，合法端仍可用 AEAD metadata 恢復並維持高 latent cosine",
             ["paper_assets/tables/table_AF025_fixed_norm_T2WB_inversion.csv"],
             ["AF025_FIXED_NORM_FORMAL_20261014T161327001754Z_c4eda689"]),
        Item("W5-T11", "table", "W5-T11_fixed_norm_probes.csv", "附錄", "Results／AF-025 structured probes",
             "AF-025 v2 的 basis、constant 與 sparse probe：原 norm、正規化 norm、protected norm 與輸出統計。",
             "fixed-norm preprocessing 移除 probe 間的 norm 差異；結構輸出仍僅作診斷",
             ["paper_assets/tables/table_AF025_fixed_norm_probe.csv"], []),
        Item("W5-T12", "table", "W5-T12_fixed_norm_repeat.csv", "附錄", "Results／AF-025 repeat transform",
             "AF-025 v2：同一 normalized latent 以 100 個 nonce 重複變換，記錄 protected norm、唯一 nonce／payload 數與跨密文 cosine。",
             "fixed-norm v2 仍維持 per-image nonce 的重複變換差異性",
             ["paper_assets/tables/table_AF025_fixed_norm_repeat.csv"], []),
        Item("W5-T13", "table", "W5-T13_defense_claims.csv", "正文", "Method／defense package",
             "P7.1 defense package 的 10 條 claim→evidence 索引，含 headline、來源 SHA-256 與適用範圍。",
             "D7 與 AF-025 的對外宣稱均可追溯至已提交的正式結果",
             ["paper_assets/defense_package/claims.csv", "paper_assets/defense_package/index.md"], []),
    ],
}

MANIFEST_FIELDS = ("id", "kind", "file", "paper_use", "section", "caption_zh", "supports_claim",
                   "sources", "run_ids", "sha256")


def build(week: int) -> Path:
    if week not in WEEKS:
        raise SystemExit(f"no registry for week {week}")
    target = ROOT / "appendix" / f"week{week}"
    for sub in ("figures", "tables"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in WEEKS[week]:
        sources = [ROOT / source for source in item.sources]
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise SystemExit(f"{item.item_id}: missing sources {missing}")
        out = target / ("figures" if item.kind == "figure" else "tables") / item.filename
        if item.build is None:
            shutil.copyfile(sources[0], out)
        else:
            item.build(out, sources)
        manifest.append({
            "id": item.item_id, "kind": item.kind, "file": str(out.relative_to(ROOT)),
            "paper_use": item.paper_use, "section": item.section, "caption_zh": item.caption,
            "supports_claim": item.supports,
            "sources": "; ".join(f"{source}@{sha256(ROOT / source)[:12]}" for source in item.sources),
            "run_ids": "; ".join(item.run_ids), "sha256": sha256(out),
        })
        print(f"{item.item_id}: {out.relative_to(ROOT)}")
    _write_csv(target / "MANIFEST.csv", manifest, list(MANIFEST_FIELDS))
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([ROOT / 'appendix' / f'week{args.week}' / 'MANIFEST.csv'], args.force)
    target = build(args.week)
    print(f"wrote {target.relative_to(ROOT)}/MANIFEST.csv")


if __name__ == "__main__":
    main()
