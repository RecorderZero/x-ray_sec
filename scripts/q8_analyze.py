#!/usr/bin/env python3
"""Q8 / P8 CPU analyzer: one validated pilot run -> five tables + fixed visual grid.

Outputs (all deterministic; re-running yields byte-identical files):
  table_fidelity.csv    Q8.1  P/S0/S1/S2 image + latent fidelity, paired differences
  table_anonymity.csv   Q8.2  anonymous-vs-original PSNR/SSIM/LPIPS/cosine + NN rank
  table_wrong_key.csv   Q8.3  correct vs wrong key/nonce vs unrelated-image control
  table_runtime.csv     Q8.4  disjoint stage runtimes (median/p95) + peak memory
  table_storage.csv     Q8.5  nonce / tag / manifest / private metadata / payload bytes
  fig_P8_visual_grid.png + P8_selection.json   P8.1 pre-registered fixed/median/worst
  per_sample_long.csv   CI-ready long-form rows behind every table
  analysis_manifest.json  provenance (run, hashes, script hashes, LPIPS state)

The analyzer fails closed on schema/hash/finite/failure-accounting violations.  It
reports scientific outcomes but never gates on them; the only flag it raises
(S2 PSNR more than 1 dB below S0) is the WORKFLOW "inspect the implementation"
trigger and is informational.

LPIPS is computed only when the real LPIPS(alex) can be built locally
(``scripts/q8_metrics.lpips_status``).  ``--lpips auto`` (default) records an
explicit ``deferred`` status and leaves the values blank; ``--lpips require`` aborts
before writing anything; ``--lpips off`` marks it not requested.  No other metric is
ever substituted.

Example:
  scripts/run_cfg_ddim.sh python -m scripts.q8_analyze --run-dir artifacts/runs/Q8_PILOT/<run_id> \\
      --output-dir paper_assets/q8_pilot --lpips auto
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from scripts.artifact_schema import validate_run
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.q8_metrics import (
    ARRAY_KINDS, CI_METHOD, METHODS, RAW_IMAGE_BYTES, SCHEMA_VERSION, SELECTION_METRIC, SELECTION_RULE,
    SEP, SHARED, STORAGES, LpipsUnavailable, build_lpips_metric, col, load_verified_arrays, lpips_status,
    nn_ranks, parse_col, rank_summary, rate_summary, select_visual_cases, shuffled_truth, stage_kind,
    stage_base, STAGE_COMPONENT, summarize, timing_summary,
)
from scripts.run_artifacts import sha256_file


STAGE_SUM_NOTES = {
    "cached_latent_to_m1": "sum of measured stages starting from the CACHED key-independent latent z (E2.3); EXCLUDES the initial x0->z inversion",
    "m1_to_reconstruction": "sum of measured stages: M1 decode/open + inversion + inverse transform + generation; EXCLUDES the initial x0->z inversion",
}
ID_COLUMNS = {
    "schema_version", "sample_id", "patient_id", "label", "seed", "noise_level", "rounds", "permutation",
    "x0_sha256", "latent_sha256",
}
IMAGE_TABLE_METRICS = ("psnr_db", "ssim", "lpips", "cosine", "linf", "rmse", "uint8_pixel_equality_rate")
VS_P_TABLE_METRICS = ("psnr_db", "ssim", "cosine", "linf")
LATENT_TABLE_METRICS = (
    "cosine", "rmse", "max_abs", "norm_ratio", "float32_bit_exact_rate",
    "lowfreq32_cosine", "highfreq_cosine", "abs_pearson",
)
PAIRED_IMAGE_METRICS = ("psnr_db", "ssim", "lpips", "cosine", "linf")
PAIRED_COMPARISONS = (("S2", "S0"), ("S2", "S1"), ("S2", "P"), ("S0", "P"), ("S1", "P"))
PSNR_REVIEW_DB = 1.0  # WORKFLOW Week 6: S2 more than 1 dB below S0 => inspect implementation
NN_SCORERS = ("cosine_raw", "pearson")
# public_available is the primary anonymity view: what a holder of the stored object can reconstruct.
# P/S0/S1: the range-restored PNG (lo/hi are public).  S2: q/255 only (lo/hi are AEAD-encrypted).
# The public information is therefore asymmetric by protocol design.  The other two views are diagnostics.
NN_VIEWS = (("public_available", "anonymous_public_available"), ("stored_pixels_q255", "anonymous_public_q"),
            ("generator_float", "anonymous_float"))
IMAGE_VIEWS = (("public_available", "anonpub"), ("generator_float", "anon"))
PUBLIC_VIEW_NOTE = (
    "deployment public_available: P/S0/S1 = predecessor legacy min-max PNG q/255; "
    "S2 = range-preserving q/255 with lo/hi encrypted - asymmetric public information by design"
)
RANGE_PARENT_PUBLIC_VIEW_NOTE = (
    "public_available: P/S0/S1 = range-restored PNG (lo/hi public); "
    "S2 = q/255 (lo/hi encrypted) - asymmetric public information by design"
)
DEPLOYMENT_STORAGE = {"P": "legacy_png_handoff", "S0": "legacy_png_handoff",
                      "S1": "legacy_png_handoff", "S2": "range_preserving_private_range_v2"}
GRID_CELL, GRID_LABEL = 256, 30

LPIPS_COLUMNS = ["input_view", "clip_fraction_mean", "clip_fraction_max"]
LPIPS_INPUT_VIEW = (
    "LPIPS(alex) inputs clipped to [0,1] then replicated to 3 channels; PSNR/SSIM/cosine/L-inf use the "
    "UNCLIPPED floats (SSIM data_range=1), so LPIPS and PSNR do not see identical pixels where clip_fraction>0"
)
COMMON = [
    "n_expected", "n", "n_failed", "mean", "ci95_low", "ci95_high", "ci_method", "median", "p95", "min", "max",
    "status",
]
FIELDS = {
    "table_fidelity.csv": [
        "method", "comparison", "baseline", "storage", "path", "space", "reference", "metric", *COMMON, "review_trigger",
        *LPIPS_COLUMNS,
    ],
    "table_anonymity.csv": [
        "method", "kind", "view", "scorer", "control", "metric", *COMMON, "gallery_size", "random_expectation",
        *LPIPS_COLUMNS, "note",
    ],
    "table_wrong_key.csv": [
        "method", "condition", "path", "space", "reference", "metric", "comparison", *COMMON, "note", *LPIPS_COLUMNS,
    ],
    "table_runtime.csv": [
        "method", "stage", "component", "kind", "storage_cell", "unit", "n_expected", "n", "n_failed",
        "median", "p95", "mean", "min", "max", "gpu_peak_mb_median", "gpu_peak_mb_max", "status", "note",
    ],
    "table_storage.csv": [
        "method", "mode", "component", "applicable", "n_expected", "n", "n_failed", "mean", "median", "min", "max",
        "share_of_total_mean", "status", "note",
    ],
}
LONG_FIELDS = ["sample_id", "patient_id", "label", "method", "group", "storage", "reference", "metric", "value"]


# --------------------------------------------------------------------------- loading
class Run:
    """A validated Q8 pilot run directory (everything read-only)."""

    def __init__(self, run_dir: Path, allow_synthetic: bool = False):
        self.run_dir = Path(run_dir)
        validation = validate_run(self.run_dir)
        if not validation["valid"]:
            raise SystemExit(f"run artifacts invalid: {validation['errors']}")
        if (self.run_dir / "failures.json").exists():
            raise SystemExit("run recorded failures; refusing to analyze a failed run")
        self.manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest = self.manifest
        if manifest.get("q8_schema_version") != SCHEMA_VERSION:
            raise SystemExit("not a Q8 run or unsupported q8_schema_version")
        if manifest.get("failures") != 0 or manifest.get("failure_reason") or manifest.get("skip_reason"):
            raise SystemExit("manifest records failures/skips; refusing to analyze")
        status_path = self.run_dir / "status.json"
        if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("state") != "succeeded":
            raise SystemExit("managed run state is not 'succeeded'")
        self.role = manifest["evidence_role"]
        if self.role == "synthetic_selftest" and not allow_synthetic:
            raise SystemExit("synthetic self-test runs carry no scientific meaning; pass --allow-synthetic to analyze")
        if manifest.get("synthetic_backend") != (self.role == "synthetic_selftest"):
            raise SystemExit("evidence_role and synthetic_backend disagree")
        index_path = self.run_dir / manifest["array_index_file"]
        if sha256_file(index_path) != manifest["array_index_sha256"]:
            raise SystemExit("array_index.json hash differs from the manifest")
        evidence = json.loads(index_path.read_text(encoding="utf-8"))
        for kind, entry in manifest["arrays"].items():
            if entry["sha256"] != evidence[kind]["sha256"]:
                raise SystemExit(f"array hash differs between manifest and index: {kind}")
        self.arrays = load_verified_arrays(self.run_dir, evidence)
        with (self.run_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.columns = list(reader.fieldnames or [])
            rows = list(reader)
        self.n = len(rows)
        self.n_expected = int(manifest["expected_samples"])
        if self.n != self.n_expected:
            raise SystemExit("row count differs from expected_samples")
        self.hash_columns = {name: [row[name] for row in rows] for name in ("x0_sha256", "latent_sha256")}
        self.sample_ids = [row["sample_id"] for row in rows]
        self.patients = [row["patient_id"] for row in rows]
        self.labels = [row["label"] for row in rows]
        for kind, array in self.arrays.items():
            if array.shape[0] != self.n:
                raise SystemExit(f"array {kind} has {array.shape[0]} samples, expected {self.n}")
        self.data: dict[str, np.ndarray] = {}
        for column in self.columns:
            if column in ID_COLUMNS:
                continue
            if parse_col(column) is None:
                raise SystemExit(f"unrecognized per-sample column: {column}")
            values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
            if not np.isfinite(values).all():
                raise SystemExit(f"non-finite values in {column}")
            self.data[column] = values
        self.storages = tuple(manifest["storages"])
        self.has_legacy = "legacy" in self.storages
        if self.storages[0] != STORAGES[0]:
            raise SystemExit("main storage cell must come first")
        self.gpu_memory = bool(manifest.get("gpu_memory_measured"))
        self.evidence = evidence
        self.deployment_supplement = None
        self.deployment_storage = {method: self.storages[0] for method in METHODS}
        self.public_view_note = RANGE_PARENT_PUBLIC_VIEW_NOTE
        self._check_consistency()

    def _check_consistency(self) -> None:
        """Engineering-correctness gates on the evidence itself (never on scientific values)."""
        manifest = self.manifest
        if len(set(self.patients)) != self.n:
            raise SystemExit("patient IDs are not unique; the patient-cluster CI assumption is violated")
        if len(set(self.sample_ids)) != self.n:
            raise SystemExit("duplicate sample IDs")
        for condition in ("wrong_key", "wrong_nonce"):
            rejected = self.get("S2", "auth", f"{condition}_rejected")
            calls = self.get("S2", "auth", f"{condition}_inverter_calls")
            if not (rejected == 1).all() or not (calls == 0).all():
                raise SystemExit(
                    f"inconsistent evidence: S2 authenticated {condition} open must be rejected (rejected==1) "
                    "with zero inverter calls in every sample")
        for i, sample_id in enumerate(self.sample_ids):
            if self.row_hash("x0_sha256", i) != self.evidence["x0"]["slice_sha256"][i]:
                raise SystemExit(f"inconsistent evidence: x0_sha256 of {sample_id} differs from the stored x0 array")
        if self.has("S2", "t1", "reused_p_output") and (self.get("S2", "t1", "reused_p_output") != 0).any():
            raise SystemExit("inconsistent evidence: S2 T1 must go through the real M2 container")
        if self.has_legacy != ("recon_m1_legacy" in self.arrays):
            raise SystemExit("inconsistent evidence: storages and recon_m1_legacy array disagree")
        if "anonymous_public_available" not in self.arrays:
            raise SystemExit("run lacks the anonymous_public_available array (re-run with the current runner)")
        mode = manifest.get("key_mode")
        if mode not in ("fixture", "random"):
            raise SystemExit("manifest lacks a valid key_mode")
        if mode == "fixture" and self.role not in ("synthetic_selftest", "smoke"):
            raise SystemExit(f"inconsistent evidence: deterministic fixture key/nonces are not allowed for role {self.role}")
        if manifest.get("raw_key_material_recorded") is not False or not manifest.get("master_key_sha256"):
            raise SystemExit("manifest must record only the master key hash and raw_key_material_recorded=false")
        if int(manifest.get("nonce_registry_count", -1)) != 2 * self.n:
            raise SystemExit("inconsistent evidence: nonce registry must hold exactly one M1 and one M2 nonce per sample")
        if self.role != "synthetic_selftest":
            needed = ("latent_cache_manifest_sha256", "latent_cache_config_sha256", "latent_cache_index_sha256",
                      "latent_cache_sha256", "latent_cache_run_id")
            if any(not manifest.get(key) for key in needed) or manifest.get("latent_cache_x0_verified") != self.n:
                raise SystemExit("inconsistent evidence: E2.3 cache provenance (hashes and per-sample x0 verification) is incomplete")

    def row_hash(self, column: str, index: int) -> str:
        return self.hash_columns[column][index]

    def has(self, method: str, group: str, name: str) -> bool:
        return col(method, group, name) in self.data

    def get(self, method: str, group: str, name: str) -> np.ndarray:
        return self.data[col(method, group, name)]

    def apply_deployment_supplement(self, supplement_dir: Path) -> None:
        """Join a hash-bound P/S0/S1 legacy cell without mutating the source run."""
        supplement_dir = Path(supplement_dir)
        validation = validate_run(supplement_dir)
        if not validation["valid"]:
            raise SystemExit(f"deployment supplement invalid: {validation['errors']}")
        manifest = json.loads((supplement_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("q8_supplement_schema") != 1 or manifest.get("parent_run_id") != self.manifest["run_id"]:
            raise SystemExit("deployment supplement does not identify this Q8 parent")
        actual_parent = {name: sha256_file(self.run_dir / name) for name in
                         ("manifest.json", "config.json", "per_sample.csv", "array_index.json")}
        if manifest.get("parent_hashes") != actual_parent:
            raise SystemExit("deployment supplement parent hashes differ from this Q8 run")
        if manifest.get("checkpoint_hash") != self.manifest["checkpoint_hash"] or manifest.get("source_sampler_sha256") != self.manifest["source_sampler_sha256"]:
            raise SystemExit("deployment supplement model/source hashes differ from the parent")
        if manifest.get("sample_ids") != self.sample_ids or manifest.get("evidence_role") != self.role:
            raise SystemExit("deployment supplement sample order/role differs from the parent")
        index_path = supplement_dir / manifest["array_index_file"]
        if sha256_file(index_path) != manifest["array_index_sha256"]:
            raise SystemExit("deployment supplement array index hash mismatch")
        evidence = json.loads(index_path.read_text(encoding="utf-8"))
        expected_methods = {"anonymous_public_q": ["P", "S0", "S1"],
                            "anonymous_public_available": ["P", "S0", "S1"],
                            "recon_m1": ["P", "S0", "S1"], "recon_wrong_key": ["S0", "S1"]}
        if set(evidence) != set(expected_methods) or manifest.get("deployment_storage_by_method") != {
                "P": "legacy_png_handoff", "S0": "legacy_png_handoff", "S1": "legacy_png_handoff"}:
            raise SystemExit("deployment supplement arrays/storage map do not match the frozen schema")
        supplemental = {}
        for kind, entry in evidence.items():
            path = supplement_dir / entry["file"]
            if sha256_file(path) != entry["sha256"]:
                raise SystemExit(f"deployment supplement array hash mismatch: {kind}")
            array = np.load(path, mmap_mode="r")
            if (list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]
                    or entry.get("methods") != expected_methods[kind] or array.shape[0] != self.n
                    or tuple(array.shape[2:]) != (256, 256)):
                raise SystemExit(f"deployment supplement array schema mismatch: {kind}")
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    digest = hashlib.sha256(np.ascontiguousarray(array[i, j]).tobytes()).hexdigest()
                    if digest != entry["slice_sha256"][i][j]:
                        raise SystemExit(f"deployment supplement slice hash mismatch: {kind}[{i},{j}]")
            supplemental[kind] = array
        with (supplement_dir / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle); supplement_rows = list(reader)
        if [row["sample_id"] for row in supplement_rows] != self.sample_ids:
            raise SystemExit("deployment supplement per-sample order differs from parent")

        # Preserve the original range-preserving baseline cell as an attribution view.
        self.arrays["recon_m1_range_attribution"] = np.asarray(self.arrays["recon_m1"]).copy()
        self.arrays["anonymous_public_range_attribution"] = np.asarray(self.arrays["anonymous_public_available"]).copy()
        for kind in ("anonymous_public_q", "anonymous_public_available", "recon_m1", "recon_wrong_key"):
            self.arrays[kind] = np.asarray(self.arrays[kind]).copy()
        for method in ("P", "S0", "S1"):
            for group, target in (("m1", "m1range"), ("m1_vs_p", "m1range_vs_p"),
                                  ("m1lat", "m1rangelat"), ("anonpub", "anonpubrange")):
                for name in [p[2] for p in map(parse_col, self.data) if p and p[0] == method and p[1] == group]:
                    self.data[col(method, target, name)] = self.data[col(method, group, name)].copy()
            for column, values in list(self.data.items()):
                parsed = parse_col(column)
                if not parsed or parsed[0] != method:
                    continue
                _, group, name = parsed
                if group == "runtime" and name.endswith("_seconds"):
                    stage = name[:-len("_seconds")]
                    self.data[col(method, "runtime", f"{stage}_rangeattr_seconds")] = values.copy()
                elif group == "store":
                    self.data[col(method, "storerange", name)] = values.copy()

        method_positions = {method: i for i, method in enumerate(("P", "S0", "S1"))}
        wrong_positions = {method: i for i, method in enumerate(("S0", "S1"))}
        for method, position in method_positions.items():
            parent_position = METHODS.index(method)
            self.arrays["anonymous_public_q"][:, parent_position] = supplemental["anonymous_public_q"][:, position]
            self.arrays["anonymous_public_available"][:, parent_position] = supplemental["anonymous_public_available"][:, position]
            self.arrays["recon_m1"][:, parent_position] = supplemental["recon_m1"][:, position]
            if method != "P":
                self.arrays["recon_wrong_key"][:, ("S0", "S1", "S2").index(method)] = supplemental["recon_wrong_key"][:, wrong_positions[method]]
        for column in supplement_rows[0]:
            parsed = parse_col(column)
            if parsed is None:
                continue
            method, group, _ = parsed
            if method not in ("P", "S0", "S1"):
                raise SystemExit(f"unexpected supplement method column: {column}")
            values = np.asarray([float(row[column]) for row in supplement_rows], dtype=np.float64)
            if not np.isfinite(values).all():
                raise SystemExit(f"non-finite deployment supplement column: {column}")
            self.data[column] = values
        self.deployment_supplement = {"run_dir": supplement_dir, "manifest": manifest}
        self.deployment_storage = dict(DEPLOYMENT_STORAGE)
        self.public_view_note = PUBLIC_VIEW_NOTE


LPIPS_PAIRS = (
    ("anon", "anonymous_float"), ("anonpub", "anonymous_public_available"), ("m1", "recon_m1"),
    ("t1", "recon_t1"), ("m1leg", "recon_m1_legacy"), ("wrongkey", "recon_wrong_key"),
    ("wrongnonce", "recon_wrong_nonce"),
)


def array_methods(kind: str) -> tuple[str, ...]:
    if kind in ("recon_m1_range_attribution", "anonymous_public_range_attribution"):
        return ("P", "S0", "S1")
    return ARRAY_KINDS[kind][1]


def clip_fractions(array: np.ndarray) -> tuple[float, float]:
    """Mean and max over samples of the fraction of pixels outside [0,1]."""
    flat = np.asarray(array, dtype=np.float64).reshape(len(array), -1)
    fraction = np.mean((flat < 0.0) | (flat > 1.0), axis=1)
    return float(fraction.mean()), float(fraction.max())


class Lpips:
    """LPIPS state, per-(method, group) values and the input-clipping disclosure.

    LPIPS (torchmetrics ``normalize=True``) needs inputs in [0,1], so the compared image
    is clipped.  ``clip`` records, for every compared array, the fraction of pixels that
    clipping touches (always computed, even if LPIPS itself is deferred).
    """

    def __init__(self, run: Run, mode: str, metric_factory: Callable[[str], Callable] | None = None, device: str = "cpu"):
        self.mode = mode
        self.values: dict[tuple[str, str], np.ndarray] = {}
        self.clip: dict[tuple[str, str], tuple[float, float]] = {}
        x0 = run.arrays["x0"]
        self.clip[(SHARED, "ctrl_unrelated")] = clip_fractions(np.asarray(x0))
        pairs = list(LPIPS_PAIRS)
        if run.deployment_supplement:
            pairs += [("anonpubrange", "anonymous_public_range_attribution"),
                      ("m1range", "recon_m1_range_attribution")]
        for method in METHODS:
            for group, kind in pairs:
                methods = array_methods(kind)
                if kind in run.arrays and method in methods:
                    self.clip[(method, group)] = clip_fractions(run.arrays[kind][:, methods.index(method)])
        self.info = lpips_status() if metric_factory is None and mode != "off" else {"available": mode != "off", "reason": ""}
        if mode == "off":
            self.state, self.reason = "not_requested", "--lpips off"
            return
        try:
            metric = (metric_factory or build_lpips_metric)(device)
        except LpipsUnavailable as error:
            if mode == "require":
                raise SystemExit(f"LPIPS required but unavailable: {error}") from error
            self.state, self.reason = "deferred", str(error)
            return
        self.state, self.reason = "computed", ""
        for method in METHODS:
            for group, kind in pairs:
                methods = array_methods(kind)
                if kind in run.arrays and method in methods:
                    position = methods.index(method)
                    self.values[(method, group)] = np.asarray([
                        metric(np.asarray(x0[i]), np.asarray(run.arrays[kind][i, position])) for i in range(run.n)
                    ], dtype=np.float64)
        self.values[(SHARED, "ctrl_unrelated")] = np.asarray([
            metric(np.asarray(x0[i]), np.asarray(x0[(i + 1) % run.n])) for i in range(run.n)
        ], dtype=np.float64)

    def disclosure(self, *keys: tuple[str, str]) -> dict[str, Any]:
        """Row-level input-view text and (worst-of-inputs) clipping fractions for an LPIPS row."""
        stats = [self.clip[key] for key in keys if key in self.clip]
        return {"input_view": LPIPS_INPUT_VIEW,
                "clip_fraction_mean": max((m for m, _ in stats), default=""),
                "clip_fraction_max": max((x for _, x in stats), default="")}

    def manifest_clip(self) -> dict[str, Any]:
        return {f"{method}:{group}": {"mean": mean, "max": maximum}
                for (method, group), (mean, maximum) in sorted(self.clip.items())}

    @property
    def status_text(self) -> str:
        return "ok" if self.state == "computed" else f"lpips {self.state}: {self.reason}"


# --------------------------------------------------------------------------- row builders
def stats_row(run: Run, values: np.ndarray | None, *, rate: bool = False, status: str = "ok") -> dict[str, Any]:
    base = {"n_expected": run.n_expected, "status": status}
    if values is None:
        return {**base, "n": 0, "n_failed": run.n_expected}
    result = rate_summary(values) if rate else summarize(values, run.patients)
    return {**base, **result, "n_failed": run.n_expected - result["n"]}


def metric_values(run: Run, lpips: Lpips, method: str, group: str, name: str) -> tuple[np.ndarray | None, str]:
    """(values, status).  LPIPS comes from the CPU analyzer, everything else from the run rows."""
    if name == "lpips":
        values = lpips.values.get((method, group))
        return (values, "ok") if values is not None else (None, lpips.status_text)
    return run.get(method, group, name), "ok"


def fidelity_rows(run: Run, lpips: Lpips) -> list[dict[str, Any]]:
    storage_main = run.storages[0]
    specs = [
        ("m1", "m1_image_inversion", storage_main, "image", "x0", IMAGE_TABLE_METRICS),
        ("t1", "t1_exact_latent_m2", "none", "image", "x0", IMAGE_TABLE_METRICS),
        ("m1_vs_p", "m1_image_inversion", storage_main, "image", "p_output", VS_P_TABLE_METRICS),
        ("m1lat", "m1_image_inversion", storage_main, "latent", "z", LATENT_TABLE_METRICS),
        ("t1lat", "t1_exact_latent_m2", "none", "latent", "z", LATENT_TABLE_METRICS),
        ("xform", "transform_only", "none", "latent", "z", LATENT_TABLE_METRICS),
    ]
    if run.has_legacy:
        specs[2:2] = [("m1leg", "m1_image_inversion", "legacy", "image", "x0", IMAGE_TABLE_METRICS)]
        specs += [("m1leg_vs_p", "m1_image_inversion", "legacy", "image", "p_output", VS_P_TABLE_METRICS),
                  ("m1leglat", "m1_image_inversion", "legacy", "latent", "z", LATENT_TABLE_METRICS)]
    rows = []
    for group, path, storage, space, reference, metrics in specs:
        for method in METHODS:
            for name in metrics:
                values, status = metric_values(run, lpips, method, group, name)
                row_storage = run.deployment_storage.get(method, storage) if group in {"m1", "m1_vs_p", "m1lat"} else storage
                rows.append({
                    "method": method, "comparison": "marginal", "baseline": "", "storage": row_storage, "path": path,
                    "space": space, "reference": reference, "metric": name, **stats_row(run, values, status=status),
                    "review_trigger": "", **(lpips.disclosure((method, group)) if name == "lpips" else {}),
                })
        paired = PAIRED_IMAGE_METRICS if space == "image" and reference == "x0" else (("cosine",) if space == "latent" else ())
        if group in {"xform", "m1_vs_p", "m1leg_vs_p"}:
            paired = ()
        for method, baseline in PAIRED_COMPARISONS:
            for name in paired:
                a, status_a = metric_values(run, lpips, method, group, name)
                b, status_b = metric_values(run, lpips, baseline, group, name)
                values = None if a is None or b is None else a - b
                trigger = ""
                if (name == "psnr_db" and space == "image" and (method, baseline) == ("S2", "S0")
                        and values is not None and float(values.mean()) < -PSNR_REVIEW_DB):
                    trigger = "S2 mean PSNR is more than 1 dB below S0: inspect the implementation before interpreting (not a pass/fail gate)"
                rows.append({
                    "method": method, "comparison": "paired_difference", "baseline": baseline,
                    "storage": (f"{run.deployment_storage[method]} vs {run.deployment_storage[baseline]}"
                                if run.deployment_supplement and group in {"m1", "m1_vs_p", "m1lat"} else storage),
                    "path": path, "space": space, "reference": reference, "metric": name,
                    **stats_row(run, values, status=status_a if status_a != "ok" else status_b), "review_trigger": trigger,
                    **(lpips.disclosure((method, group), (baseline, group)) if name == "lpips" else {}),
                })
    if run.deployment_supplement:
        for group, path, space, reference, metrics in (
            ("m1range", "m1_image_inversion_attribution", "image", "x0", IMAGE_TABLE_METRICS),
            ("m1range_vs_p", "m1_image_inversion_attribution", "image", "p_output", VS_P_TABLE_METRICS),
            ("m1rangelat", "m1_image_inversion_attribution", "latent", "z", LATENT_TABLE_METRICS),
        ):
            for method in ("P", "S0", "S1"):
                for name in metrics:
                    values, status = metric_values(run, lpips, method, group, name)
                    rows.append({"method": method, "comparison": "attribution_marginal", "baseline": "",
                                 "storage": "range_preserving_public_range_v1", "path": path, "space": space,
                                 "reference": reference, "metric": name, **stats_row(run, values, status=status),
                                 "review_trigger": "", **(lpips.disclosure((method, group)) if name == "lpips" else {})})
    return rows


def anonymity_rows(run: Run, lpips: Lpips) -> list[dict[str, Any]]:
    rows = []
    image_views = list(IMAGE_VIEWS)
    if run.deployment_supplement:
        image_views.append(("range_attribution", "anonpubrange"))
    for view, view_group in image_views:
        view_methods = ("P", "S0", "S1") if view == "range_attribution" else (*METHODS, "unrelated_control")
        for method in view_methods:
            if method == "unrelated_control":
                if view != "generator_float":
                    continue  # the unrelated-image floor does not depend on a storage view
                source_method, group, shown_view = SHARED, "ctrl_unrelated", "reference"
            else:
                source_method, group, shown_view = method, view_group, view
            for name in ("psnr_db", "ssim", "lpips", "cosine", "linf"):
                if name == "lpips":
                    values, status = lpips.values.get((source_method, group)), lpips.status_text
                    status = "ok" if values is not None else status
                else:
                    values, status = run.get(source_method, group, name), "ok"
                rows.append({"method": method, "kind": "image_difference", "view": shown_view, "scorer": "",
                             "control": "negative_floor" if method == "unrelated_control" else "", "metric": name,
                             **stats_row(run, values, status=status), "gallery_size": "", "random_expectation": "",
                             "note": run.public_view_note if view == "public_available" else "",
                             **(lpips.disclosure((source_method, group)) if name == "lpips" else {})})
    x0 = run.arrays["x0"]
    nn_views = list(NN_VIEWS)
    if run.deployment_supplement:
        nn_views.append(("range_attribution", "anonymous_public_range_attribution"))
    for view, kind in nn_views:
        array = run.arrays[kind]
        view_methods = ("P", "S0", "S1") if view == "range_attribution" else METHODS
        for method in view_methods:
            queries = np.asarray(array[:, array_methods(kind).index(method)], dtype=np.float64)
            if kind == "anonymous_public_q":
                queries = queries / 255.0
            for scorer in NN_SCORERS:
                for control, truth in (("true_match", None), ("shuffled_truth_negative", shuffled_truth(run.n))):
                    ranks = nn_ranks(queries, np.asarray(x0), scorer, truth)
                    summary = rank_summary(ranks, run.n)
                    entries = {
                        "top1": (summary["top1"], *summary["top1_ci95_wilson"], "wilson 95%"),
                        "top5": (summary["top5"], *summary["top5_ci95_wilson"], "wilson 95%"),
                        "mrr": (summary["mAP"], *summary["mAP_ci95_bootstrap"], "percentile bootstrap, seed 1911, 10000 resamples"),
                        "median_rank": (summary["median_rank"], "", "", ""),
                        "mean_rank": (summary["mean_rank"], "", "", ""),
                    }
                    random_expectation = {"top1": 1.0 / run.n, "top5": min(5, run.n) / run.n,
                                          "mrr": float(np.mean(1.0 / np.arange(1, run.n + 1))),
                                          "median_rank": (run.n + 1) / 2, "mean_rank": summary["expected_random_mean_rank"]}
                    for name, (mean, low, high, ci_method) in entries.items():
                        rows.append({
                            "method": method, "kind": "nn_rank", "view": view, "scorer": scorer, "control": control,
                            "metric": name, "n_expected": run.n_expected, "n": run.n, "n_failed": 0, "mean": mean,
                            "ci95_low": low, "ci95_high": high, "ci_method": ci_method,
                            "median": summary["median_rank"] if name == "median_rank" else "", "p95": "",
                            "min": "", "max": summary["max_rank"] if name == "median_rank" else "", "status": "ok",
                            "gallery_size": run.n, "random_expectation": random_expectation[name],
                            "note": run.public_view_note if view == "public_available" else "",
                        })
    return rows


def wrong_key_rows(run: Run, lpips: Lpips) -> list[dict[str, Any]]:
    rows = []
    image_metrics = ("psnr_db", "ssim", "lpips", "cosine", "linf")

    def add(method, condition, path, space, reference, name, values, status="ok", comparison="marginal", note="",
            rate=False, lp=()):
        rows.append({"method": method, "condition": condition, "path": path, "space": space,
                     "reference": reference, "metric": name, "comparison": comparison,
                     **stats_row(run, values, rate=rate, status=status), "note": note,
                     **(lpips.disclosure(*lp) if name == "lpips" else {})})

    for method in METHODS:
        for name in image_metrics:
            values, status = metric_values(run, lpips, method, "m1", name)
            add(method, "correct_key_nonce", "legitimate_open", "image", "x0", name, values, status, lp=((method, "m1"),))
        for name in ("cosine", "rmse"):
            add(method, "correct_key_nonce", "legitimate_open", "latent", "z", name, run.get(method, "m1lat", name))
    for method, conditions in (("S0", ("wrongkey",)), ("S1", ("wrongkey",)), ("S2", ("wrongkey", "wrongnonce"))):
        for group in conditions:
            condition = "wrong_key" if group == "wrongkey" else "wrong_nonce"
            note = ("authentication bypassed; decryptor given true range/norm (conservative oracle)" if method == "S2"
                    else "legacy password replaced; no authenticator exists")
            for name in image_metrics:
                values, status = metric_values(run, lpips, method, group, name)
                add(method, condition, "bypass_auth_counterfactual", "image", "x0", name, values, status, note=note,
                    lp=((method, group),))
                unrelated, _ = metric_values(run, lpips, SHARED, "ctrl_unrelated", name)
                if values is not None and unrelated is not None:
                    add(method, condition, "bypass_auth_counterfactual", "image", "x0", name, values - unrelated,
                        comparison="wrong_minus_unrelated_control", note=note,
                        lp=((method, group), (SHARED, "ctrl_unrelated")))
            for name in ("cosine", "rmse"):
                add(method, condition, "bypass_auth_counterfactual", "latent", "z", name, run.get(method, f"{group}lat", name), note=note)
                add(method, condition, "bypass_auth_counterfactual", "latent", "z", name,
                    run.get(method, f"{group}lat", name) - run.get(SHARED, "ctrl_unrelated_lat", name),
                    comparison="wrong_minus_unrelated_control", note=note)
    for name in image_metrics:
        values, status = metric_values(run, lpips, SHARED, "ctrl_unrelated", name)
        add("unrelated_control", "unrelated_image", "reference", "image", "x0", name, values, status,
            note="x0 of the next sample in split order vs x0; numeric floor for 'unrelated'",
            lp=((SHARED, "ctrl_unrelated"),))
    for name in ("cosine", "rmse"):
        add("unrelated_control", "unrelated_image", "reference", "latent", "z", name, run.get(SHARED, "ctrl_unrelated_lat", name))
    for condition in ("wrong_key", "wrong_nonce"):
        add("S2", condition, "authenticated_open", "decision", "", "rejection_rate",
            run.get("S2", "auth", f"{condition}_rejected"), rate=True, note="HMAC verified before any decode/inversion")
        calls = run.get("S2", "auth", f"{condition}_inverter_calls")
        add("S2", condition, "authenticated_open", "decision", "", "inverter_calls", calls,
            note="per-sample count; max must be 0: the diffusion inverter is never invoked on a rejected container")
    for method in ("P", "S0", "S1"):
        rows.append({"method": method, "condition": "wrong_key", "path": "authenticated_open", "space": "decision",
                     "reference": "", "metric": "rejection_rate", "comparison": "marginal", "n_expected": run.n_expected,
                     "n": 0, "n_failed": 0, "status": "not_applicable: no authenticator in this scheme", "note": ""})
    return rows


def runtime_rows(run: Run) -> list[dict[str, Any]]:
    rows = []
    stages: dict[tuple[str, str], str] = {}
    for column in run.data:
        parsed = parse_col(column)
        if parsed and parsed[1] == "runtime" and parsed[2].endswith("_seconds"):
            stages[(parsed[0], parsed[2][:-len("_seconds")])] = column
    order = {name: position for position, name in enumerate(STAGE_COMPONENT)}
    for (method, stage), column in sorted(stages.items(), key=lambda item: (METHODS.index(item[0][0]), order.get(stage_base(item[0][1])[0], 99), item[0][1])):
        base, cell = stage_base(stage)
        if run.deployment_supplement and method in ("P", "S0", "S1") and cell == "main":
            cell = "legacy_png_handoff"
        elif run.deployment_supplement and method == "S2" and cell == "main":
            cell = "range_preserving_private_range_v2"
        values = run.data[column]
        mask = np.ones(run.n, dtype=bool)
        note = ""
        if base in STAGE_SUM_NOTES:
            note = STAGE_SUM_NOTES[base]
        if base == "diffusion_generate_t1" and run.has(method, "t1", "reused_p_output"):
            mask = run.get(method, "t1", "reused_p_output") == 0
            note = "samples that reused P's output (bit-exact transform) are excluded, not counted as zero-time"
        memory = run.data.get(col(method, "mem", f"{stage}_gpu_peak_mb"))
        stat = timing_summary(values[mask]) if mask.any() else None
        rows.append({
            "method": method, "stage": base, "component": STAGE_COMPONENT.get(base, "unknown"), "kind": stage_kind(base),
            "storage_cell": cell, "unit": "seconds", "n_expected": run.n_expected, "n": int(mask.sum()),
            "n_failed": run.n_expected - run.n if stat else run.n_expected,
            **({} if stat is None else {k: stat[k] for k in ("median", "p95", "mean", "min", "max")}),
            "gpu_peak_mb_median": "" if memory is None or stat is None else float(np.median(memory[mask])),
            "gpu_peak_mb_max": "" if memory is None or stat is None else float(memory[mask].max()),
            "status": ("ok" if run.gpu_memory else "ok; GPU memory not measured (CPU run)") if stat else "no samples",
            "note": note,
        })
    for method in METHODS:  # derived non-diffusion shares of the end-to-end stage sums
        for label, numerator_stages, total_stage in (
            ("derived_nondiffusion_share_cached_latent_to_m1",
             ("fixed_norm_scale", "transform_forward", "container_seal_total", "storage_encode"), "cached_latent_to_m1"),
            ("derived_nondiffusion_share_m1_to_reconstruction",
             ("container_open_total", "storage_decode", "transform_inverse"), "m1_to_reconstruction"),
        ):
            total = run.data.get(col(method, "runtime", f"{total_stage}_seconds"))
            parts = [run.data[col(method, "runtime", f"{s}_seconds")] for s in numerator_stages
                     if col(method, "runtime", f"{s}_seconds") in run.data]
            if total is None or not parts:
                continue
            share = np.sum(parts, axis=0) / np.maximum(total, 1e-30)
            stat = timing_summary(share)
            rows.append({"method": method, "stage": label, "component": "derived", "kind": "derived",
                         "storage_cell": "", "unit": "fraction", "n_expected": run.n_expected, "n": run.n, "n_failed": 0,
                         **{k: stat[k] for k in ("median", "p95", "mean", "min", "max")}, "gpu_peak_mb_median": "",
                         "gpu_peak_mb_max": "", "status": "ok",
                         "note": "(KDF+transform+MAC+AEAD+codec stages) / end-to-end stage sum, per sample"})
    rss = run.data.get(col(SHARED, "mem", "process_peak_rss_mb"))
    if rss is not None:
        stat = timing_summary(rss)
        rows.append({"method": SHARED, "stage": "process_peak_rss", "component": "memory", "kind": "pipeline",
                     "storage_cell": "", "unit": "MiB", "n_expected": run.n_expected, "n": run.n, "n_failed": 0,
                     **{k: stat[k] for k in ("median", "p95", "mean", "min", "max")}, "gpu_peak_mb_median": "",
                     "gpu_peak_mb_max": "", "status": "ok", "note": "process high-water mark (ru_maxrss), not per stage"})
    return rows


STORAGE_COMPONENTS = (
    "nonce_bytes", "tag_bytes", "manifest_bytes", "private_metadata_bytes", "latent_payload_bytes",
    "image_payload_bytes", "encoding_overhead_bytes", "total_bytes",
)


def storage_rows(run: Run) -> list[dict[str, Any]]:
    rows = []

    def add(method, mode, component, applicable, values, total_mean=None, note=""):
        if values is None:
            rows.append({"method": method, "mode": mode, "component": component, "applicable": int(applicable),
                         "n_expected": run.n_expected, "n": 0, "n_failed": 0, "status": "not_applicable", "note": note})
            return
        array = np.asarray(values, dtype=np.float64)
        rows.append({"method": method, "mode": mode, "component": component, "applicable": int(applicable),
                     "n_expected": run.n_expected, "n": len(array), "n_failed": run.n_expected - len(array),
                     "mean": float(array.mean()), "median": float(np.median(array)), "min": float(array.min()),
                     "max": float(array.max()),
                     "share_of_total_mean": "" if not total_mean else float(array.mean() / total_mean),
                     "status": "ok", "note": note})

    zeros = np.zeros(run.n)
    for method in METHODS:
        has_container = run.has(method, "store", "m1_total_bytes")
        preview = run.get(method, "store", "m1_image_payload_bytes")
        for mode in ("M1_image_only", "M2_latent_payload", "M2_plus_preview"):
            prefix = "m1" if mode == "M1_image_only" else "m2"
            parts: dict[str, np.ndarray] = {}
            for component in STORAGE_COMPONENTS:
                if has_container:
                    parts[component] = run.get(method, "store", f"{prefix}_{component}")
                else:
                    parts[component] = zeros
            if not has_container:
                payload = preview if mode == "M1_image_only" else run.get(method, "store", "m2_latent_payload_bytes")
                key = "image_payload_bytes" if mode == "M1_image_only" else "latent_payload_bytes"
                parts[key] = np.asarray(payload, dtype=np.float64)
                parts["total_bytes"] = np.asarray(payload, dtype=np.float64)
            if mode == "M2_plus_preview":
                parts["preview_image_bytes"] = np.asarray(preview, dtype=np.float64)
                parts["total_bytes"] = parts["total_bytes"] + parts["preview_image_bytes"]
            total_mean = float(parts["total_bytes"].mean())
            note = "" if has_container else "no container: payload only; key held outside the object, no nonce/tag/manifest"
            if mode == "M2_plus_preview":
                note = ("anonymous preview PNG stored next to the M2 payload; " + note).strip("; ")
            for component, values in parts.items():
                applicable = has_container or component in {"image_payload_bytes", "latent_payload_bytes", "total_bytes", "preview_image_bytes"}
                add(method, mode, component, applicable, values, total_mean, note)
        base_total = run.get("P", "store", "m1_image_payload_bytes")
        add(method, "M1_image_only", "total_vs_P_ratio",
            True, (run.get(method, "store", "m1_total_bytes") if has_container else preview) / base_total,
            note="per-sample ratio of this method's M1 object to P's PNG")
    if run.deployment_supplement:
        range_p = run.get("P", "storerange", "m1_image_payload_bytes")
        for method in ("P", "S0", "S1"):
            payload = run.get(method, "storerange", "m1_image_payload_bytes")
            add(method, "M1_range_attribution", "image_payload_bytes", True, payload, float(payload.mean()),
                note="parent public-range PNG attribution cell; not the selected deployment")
            add(method, "M1_range_attribution", "total_bytes", True, payload, float(payload.mean()),
                note="parent public-range PNG attribution cell; not the selected deployment")
            add(method, "M1_range_attribution", "total_vs_P_ratio", True, payload / range_p,
                note="parent public-range PNG attribution cell; not the selected deployment")
    add("raw_reference", "reference", "uint8_image_bytes", True, np.full(run.n, float(RAW_IMAGE_BYTES)), note="256x256x1 byte, uncompressed")
    add("raw_reference", "reference", "float32_latent_bytes", True, np.full(run.n, float(RAW_IMAGE_BYTES * 4)), note="256x256 float32")
    return rows


# --------------------------------------------------------------------------- long rows
def long_rows(run: Run, lpips: Lpips) -> list[dict[str, Any]]:
    rows = []
    for column, values in run.data.items():
        method, group, name = parse_col(column)
        kind = group
        if group.startswith("m1range"):
            storage = "range_preserving_public_range_v1"
        elif group.startswith("m1leg"):
            storage = "legacy"
        elif group.startswith("m1"):
            storage = run.deployment_storage.get(method, run.storages[0])
        else:
            storage = ""
        for i in range(run.n):
            rows.append({"sample_id": run.sample_ids[i], "patient_id": run.patients[i], "label": run.labels[i],
                         "method": method, "group": kind, "storage": storage, "reference": "", "metric": name,
                         "value": repr(float(values[i]))})
    for (method, group), values in sorted(lpips.values.items()):
        for i in range(run.n):
            rows.append({"sample_id": run.sample_ids[i], "patient_id": run.patients[i], "label": run.labels[i],
                         "method": method, "group": group, "storage": "", "reference": "x0", "metric": "lpips",
                         "value": repr(float(values[i]))})
    return rows


# --------------------------------------------------------------------------- P8 grid
def selection(run: Run) -> dict[str, Any]:
    scores = run.get("S2", "m1", "psnr_db")
    cases = select_visual_cases(run.sample_ids, run.labels, scores)
    ranking = sorted(zip(run.sample_ids, scores.tolist()), key=lambda item: (item[1], item[0]))
    return {"rule": SELECTION_RULE, "metric": SELECTION_METRIC, "cases": cases,
            "ranking_ascending": [{"sample_id": s, "score": v} for s, v in ranking]}


def _gray(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(np.asarray(array, dtype=np.float32), 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")


def render_grid(run: Run, cases: list[dict[str, Any]], path: Path) -> None:
    columns = ["original"] + [f"{m} public view" for m in METHODS] + [f"{m} recon" for m in METHODS] + ["|x0-S2 recon| 0..0.1"]
    width, height = GRID_CELL * len(columns), (GRID_CELL + GRID_LABEL) * len(cases) + 34
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    storage_label = ("deployment P/S0/S1=legacy, S2=range-private" if run.deployment_supplement
                     else f"main storage {run.storages[0]}")
    draw.text((4, 4), f"Q8/P8 {run.role} run {run.manifest['run_id']} | fixed display range [0,1] | "
                      f"{storage_label} | anon columns = public_available | "
                      f"selection by {SELECTION_METRIC}", fill="black")
    for column, title in enumerate(columns):
        draw.text((column * GRID_CELL + 4, 18), title, fill="black")
    for row_index, case in enumerate(cases):
        i = run.sample_ids.index(case["sample_id"])
        top = 34 + row_index * (GRID_CELL + GRID_LABEL)
        draw.text((4, top + 4), f"{case['role']}: {case['sample_id']} (label {case['label']}) "
                                f"S2 m1 PSNR={case['selection_score']:.2f} dB, rank {case['rank_ascending']}/{case['n']} ascending",
                  fill="black")
        x0 = np.asarray(run.arrays["x0"][i])
        panels = [x0]
        panels += [np.asarray(run.arrays["anonymous_public_available"][i, k]) for k in range(len(METHODS))]
        panels += [np.asarray(run.arrays["recon_m1"][i, k]) for k in range(len(METHODS))]
        panels.append(np.abs(np.asarray(run.arrays["recon_m1"][i, METHODS.index("S2")]) - x0) / 0.1)
        for column, panel in enumerate(panels):
            canvas.paste(_gray(panel), (column * GRID_CELL, top + GRID_LABEL))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


# --------------------------------------------------------------------------- writing
def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = FIELDS[path.name]
    unknown = {key for row in rows for key in row} - set(fields)
    if unknown:
        raise ValueError(f"{path.name}: undeclared columns {sorted(unknown)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _fmt(value: Any) -> Any:
    return repr(float(value)) if isinstance(value, (float, np.floating)) else value


def output_paths(output_dir: Path) -> dict[str, Path]:
    names = [*FIELDS, "fig_P8_visual_grid.png", "P8_selection.json", "per_sample_long.csv", "analysis_manifest.json"]
    return {name: output_dir / name for name in names}


def analyze(run_dir: Path, output_dir: Path, *, deployment_supplement: Path | None = None,
            lpips_mode: str = "auto", allow_synthetic: bool = False,
            force: bool = False, metric_factory: Callable[[str], Callable] | None = None) -> dict[str, Any]:
    run = Run(run_dir, allow_synthetic)
    if deployment_supplement is not None:
        run.apply_deployment_supplement(deployment_supplement)
    paths = output_paths(Path(output_dir))
    refuse_overwrite(paths.values(), force)
    lpips = Lpips(run, lpips_mode, metric_factory)  # may SystemExit before anything is written
    tables = {
        "table_fidelity.csv": fidelity_rows(run, lpips), "table_anonymity.csv": anonymity_rows(run, lpips),
        "table_wrong_key.csv": wrong_key_rows(run, lpips), "table_runtime.csv": runtime_rows(run),
        "table_storage.csv": storage_rows(run),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_table(paths[name], rows)
    selected = selection(run)
    paths["P8_selection.json"].write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_grid(run, selected["cases"], paths["fig_P8_visual_grid.png"])
    long = long_rows(run, lpips)
    with paths["per_sample_long.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(long)
    here = Path(__file__).resolve().parent
    manifest = {
        "schema_version": 1, "source_run_id": run.manifest["run_id"], "evidence_role": run.role,
        "formal": bool(run.manifest.get("formal")), "synthetic_backend": bool(run.manifest.get("synthetic_backend")),
        "n_expected": run.n_expected, "n_ok": run.n, "n_failed": run.n_expected - run.n,
        "storages": sorted(set(run.deployment_storage.values())),
        "source_parent_storages": list(run.storages), "ci_method": CI_METHOD,
        "deployment_storage_by_method": run.deployment_storage,
        "source_manifest_sha256": sha256_file(run.run_dir / "manifest.json"),
        "source_per_sample_sha256": sha256_file(run.run_dir / "per_sample.csv"),
        "source_array_index_sha256": run.manifest["array_index_sha256"],
        "source_git_commit": run.manifest["git_commit"], "source_git_dirty": run.manifest["git_dirty"],
        "deployment_supplement": (None if run.deployment_supplement is None else {
            "run_id": run.deployment_supplement["manifest"]["run_id"],
            "manifest_sha256": sha256_file(run.deployment_supplement["run_dir"] / "manifest.json"),
            "per_sample_sha256": sha256_file(run.deployment_supplement["run_dir"] / "per_sample.csv"),
            "array_index_sha256": run.deployment_supplement["manifest"]["array_index_sha256"],
        }),
        "script_sha256": {name: sha256_file(here / name) for name in ("q8_analyze.py", "q8_metrics.py")},
        "lpips": {"state": lpips.state, "reason": lpips.reason, "info": lpips.info,
                  "input_view": LPIPS_INPUT_VIEW, "clip_fraction_per_array": lpips.manifest_clip()},
        "public_view_note": run.public_view_note, "key_mode": run.manifest["key_mode"],
        "master_key_sha256": run.manifest["master_key_sha256"], "patients_unique": True,
        "latent_cache_hashes": {k: run.manifest.get(k) for k in (
            "latent_cache_run_id", "latent_cache_manifest_sha256", "latent_cache_config_sha256",
            "latent_cache_index_sha256", "latent_cache_sha256", "latent_cache_x0_verified")},
        "selection": {"rule": SELECTION_RULE, "cases": selected["cases"]},
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "outputs_sha256": {name: sha256_file(path) for name, path in sorted(paths.items()) if name != "analysis_manifest.json"},
        "interpretation_guard": "pilot infrastructure output; scientific values are reported, never used as pass/fail gates",
    }
    paths["analysis_manifest.json"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deployment-supplement", type=Path,
                        help="validated P/S0/S1 legacy derived run; makes the mixed deployment cell primary")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_assets/q8_pilot"))
    parser.add_argument("--lpips", choices=("auto", "require", "off"), default="auto")
    parser.add_argument("--allow-synthetic", action="store_true")
    add_force_flag(parser)
    args = parser.parse_args(argv)
    manifest = analyze(args.run_dir, args.output_dir, deployment_supplement=args.deployment_supplement, lpips_mode=args.lpips,
                       allow_synthetic=args.allow_synthetic, force=args.force)
    print(json.dumps({"output_dir": str(args.output_dir), "lpips": manifest["lpips"]["state"],
                      "table_rows": manifest["table_rows"]}, indent=2))


if __name__ == "__main__":
    main()
