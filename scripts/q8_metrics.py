"""Q8/P8 shared CPU-only metric, statistics, selection and accounting helpers.

Nothing here touches a GPU or the diffusion model; the pilot runner
(``scripts/q8_pilot_runner.py``) and the analyzer (``scripts/q8_analyze.py``)
both import these definitions so that a number computed on the GPU host and a
number recomputed on CPU share one estimand.  Scientific outcomes are only ever
reported, never used as pass/fail gates (loop_engineering_spec 1.2).

Frozen conventions (pre-registered before any Q8 data exists):
  * bootstrap: seed 1911, 10,000 percentile resamples, resampling whole patients
    (identical to per-sample resampling when every patient has one image);
  * percentiles (runtime p95): numpy ``linear`` interpolation;
  * NN rank: pessimistic ties (the true match is last among exact ties), the
    same convention as ``scripts/d7_metrics.py``;
  * LPIPS: never substituted; see :func:`lpips_status`.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from scripts.d7_metrics import rank_metrics, ranks_cosine, wilson
from scripts.run_artifacts import sha256_file


SEED = 1911
BOOTSTRAP_RESAMPLES = 10_000
CI_METHOD = "percentile bootstrap, seed 1911, 10000 resamples, cluster=patient"
METHODS = ("P", "S0", "S1", "S2")
# Storage cells: the first is the externally claimed main result (PROPOSAL 4.5).
STORAGES = ("range_preserving", "legacy")
RAW_IMAGE_BYTES = 256 * 256  # one uint8 pixel per sample, reference size only
SCHEMA_VERSION = 1
SEP = "__"
SHARED = "ALL"  # method token of method-independent columns (controls, process memory)

# P8.1 pre-registered case selection (criterion fixed here, before data).
SELECTION_METRIC = "S2__m1__psnr_db"
SELECTION_RULE = (
    "fixed = first no_finding (label 0) and first label-1 sample in split order; "
    f"median = lower-median of ascending {SELECTION_METRIC} (ties by sample_id); "
    f"worst = minimum {SELECTION_METRIC} (ties by sample_id); "
    "a sample selected twice is shown twice, never substituted"
)


# --------------------------------------------------------------------------- columns
def col(method: str, group: str, name: str) -> str:
    """Per-sample wide-table column name, e.g. ``S2__m1__psnr_db``."""
    return f"{method}{SEP}{group}{SEP}{name}"


def parse_col(column: str) -> tuple[str, str, str] | None:
    parts = column.split(SEP, 2)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 and parts[0] in (*METHODS, SHARED) else None


# --------------------------------------------------------------------------- statistics
def bootstrap_mean_ci(
    values: Sequence[float], clusters: Sequence[str] | None = None, *,
    seed: int = SEED, resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Percentile CI of the mean, resampling clusters (patients) with replacement.

    A fixed seed and identical cluster layout give identical resample indices for
    every metric, so paired differences share their resampling with the marginals.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("bootstrap input must be a non-empty finite 1-D array")
    labels = list(range(len(array))) if clusters is None else list(clusters)
    if len(labels) != len(array):
        raise ValueError("clusters/values length mismatch")
    unique = sorted(set(labels))
    index = {label: position for position, label in enumerate(unique)}
    sums = np.zeros(len(unique)); counts = np.zeros(len(unique))
    for label, value in zip(labels, array):
        sums[index[label]] += value; counts[index[label]] += 1
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(resamples, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(
    values: Sequence[float], clusters: Sequence[str] | None = None, *,
    seed: int = SEED, resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_mean_ci(array, clusters, seed=seed, resamples=resamples)
    return {
        "n": int(len(array)), "mean": float(array.mean()), "ci95_low": low, "ci95_high": high,
        "ci_method": CI_METHOD, "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "min": float(array.min()), "max": float(array.max()),
    }


def rate_summary(flags: Sequence[bool | int]) -> dict[str, Any]:
    """Proportion with a Wilson 95% interval (exact counts kept)."""
    array = np.asarray(flags, dtype=np.int64)
    low, high = wilson(int(array.sum()), len(array))
    return {
        "n": int(len(array)), "mean": float(array.mean()), "ci95_low": low, "ci95_high": high,
        "ci_method": "wilson 95%", "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "min": float(array.min()), "max": float(array.max()),
    }


def timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 1 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("timing values must be finite and non-negative")
    return {
        "n": int(len(array)), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95, method="linear")), "mean": float(array.mean()),
        "min": float(array.min()), "max": float(array.max()),
    }


# --------------------------------------------------------------------------- linkage
def nn_ranks(queries: np.ndarray, gallery: np.ndarray, scorer: str, truth: np.ndarray | None = None) -> np.ndarray:
    """Rank (1 = best) of the true original among the gallery of originals."""
    flat_q = np.asarray(queries, dtype=np.float64).reshape(len(queries), -1)
    flat_g = np.asarray(gallery, dtype=np.float64).reshape(len(gallery), -1)
    if scorer == "pearson":
        flat_q = flat_q - flat_q.mean(axis=1, keepdims=True)
        flat_g = flat_g - flat_g.mean(axis=1, keepdims=True)
    elif scorer != "cosine_raw":
        raise ValueError(f"unknown NN scorer: {scorer}")
    labels = np.arange(len(flat_q)) if truth is None else np.asarray(truth)
    return ranks_cosine(flat_q, flat_g, labels)


def shuffled_truth(size: int, seed: int = SEED) -> np.ndarray:
    """Deterministic derangement (cyclic shift) for the negative control."""
    if size < 2:
        raise ValueError("shuffled control needs at least two samples")
    shift = 1 + int(np.random.default_rng(seed).integers(0, size - 1))
    return (np.arange(size) + shift) % size


def rank_summary(ranks: np.ndarray, gallery_size: int) -> dict[str, Any]:
    out = rank_metrics(ranks, gallery_size)
    out["mean_rank"] = float(np.mean(ranks))
    out["expected_random_mean_rank"] = (gallery_size + 1) / 2
    return out


# --------------------------------------------------------------------------- P8 selection
def select_visual_cases(
    sample_ids: Sequence[str], labels: Sequence[str], scores: Sequence[float],
) -> list[dict[str, Any]]:
    """Deterministic fixed/median/worst cases; see ``SELECTION_RULE``."""
    if not (len(sample_ids) == len(labels) == len(scores)) or not sample_ids:
        raise ValueError("selection inputs must be equal-length and non-empty")
    if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
        raise ValueError("selection scores must be finite")
    ranked = sorted(range(len(sample_ids)), key=lambda i: (float(scores[i]), sample_ids[i]))
    cases: list[dict[str, Any]] = []
    for wanted in ("0", "1"):
        matches = [i for i, label in enumerate(labels) if str(label) == wanted]
        if matches:
            cases.append({"role": f"fixed_label{wanted}", "index": matches[0]})
    cases.append({"role": "median", "index": ranked[(len(ranked) - 1) // 2]})
    cases.append({"role": "worst", "index": ranked[0]})
    return [{
        "role": case["role"], "sample_id": sample_ids[case["index"]],
        "label": str(labels[case["index"]]), "selection_score": float(scores[case["index"]]),
        "selection_metric": SELECTION_METRIC,
        "rank_ascending": ranked.index(case["index"]) + 1, "n": len(sample_ids),
    } for case in cases]


# --------------------------------------------------------------------------- LPIPS
LPIPS_BACKBONE_SHA_PREFIX = "7be5be79"  # torchvision alexnet-owt-<prefix>.pth file-name hash


class LpipsUnavailable(RuntimeError):
    """LPIPS cannot be computed faithfully on this host; never substitute another metric."""


def lpips_status(hub_dir: Path | None = None) -> dict[str, Any]:
    """Report whether true LPIPS(alex) is computable, without any network access.

    Requires (a) the ``torchmetrics`` LPIPS implementation with its packaged
    linear-layer weights and (b) the torchvision AlexNet ImageNet backbone already
    present in the torch hub cache with the expected file-name hash.  Missing
    pieces yield ``available=False`` and a human-readable reason; the caller
    decides between fail-closed (``require``) and explicit deferral (``auto``).
    """
    info: dict[str, Any] = {"metric": "LPIPS", "net": "alex", "available": False, "reason": "", "backend": "torchmetrics"}
    try:
        import torch
        import torchmetrics
        import torchmetrics.functional.image.lpips as module
    except Exception as error:  # pragma: no cover - environment dependent
        info["reason"] = f"torchmetrics LPIPS import failed: {type(error).__name__}"
        return info
    info["torchmetrics_version"] = torchmetrics.__version__
    linear = Path(module.__file__).parent / "lpips_models" / "alex.pth"
    if not linear.is_file():
        info["reason"] = "torchmetrics packaged LPIPS linear weights alex.pth are missing"
        return info
    info["linear_weights_sha256"] = sha256_file(linear)
    directory = Path(hub_dir) if hub_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
    candidates = sorted(glob.glob(str(directory / "alexnet-owt-*.pth")))
    if not candidates:
        info["reason"] = f"AlexNet ImageNet backbone weights absent from {directory}; downloading is not performed"
        return info
    backbone = Path(candidates[0])
    digest = sha256_file(backbone)
    if not digest.startswith(LPIPS_BACKBONE_SHA_PREFIX):
        info["reason"] = f"{backbone.name} sha256 {digest[:8]} does not match expected prefix {LPIPS_BACKBONE_SHA_PREFIX}"
        return info
    info.update(available=True, backbone_weights=str(backbone.name), backbone_weights_sha256=digest)
    return info


def build_lpips_metric(device: str = "cpu") -> Callable[[np.ndarray, np.ndarray], float]:
    """Return ``f(ref, other) -> LPIPS`` for two HxW float images in [0,1] (clipped).

    Raises :class:`LpipsUnavailable` when :func:`lpips_status` is not available.
    """
    status = lpips_status()
    if not status["available"]:
        raise LpipsUnavailable(status["reason"])
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()

    def compute(reference: np.ndarray, other: np.ndarray) -> float:
        def tensor(array: np.ndarray) -> "torch.Tensor":
            clipped = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
            return torch.from_numpy(clipped)[None, None].repeat(1, 3, 1, 1).to(device)
        with torch.inference_mode():
            metric.reset()
            metric.update(tensor(other), tensor(reference))
            return float(metric.compute().item())

    return compute


# --------------------------------------------------------------------------- runtime catalog
# stage -> component.  Stage prefixes give the measurement kind: ``micro_`` stages
# re-time one primitive in isolation (not additive with ``pipeline`` stages),
# ``control_`` stages belong to wrong-key controls and are excluded from
# deployment totals, everything else is measured inside the real pipeline.
STAGE_COMPONENT = {
    "fixed_norm_scale": "transform", "transform_forward": "transform", "transform_inverse": "transform",
    "micro_transform_inverse": "transform", "micro_kdf_subkeys": "kdf",
    "micro_mac_compute": "mac", "micro_mac_verify": "mac",
    "container_seal_total": "container", "container_open_total": "container",
    "m2_seal_total": "container", "m2_open_total": "container",
    "storage_encode": "storage_codec", "storage_decode": "storage_codec",
    "diffusion_generate_anonymous": "diffusion", "diffusion_invert": "diffusion",
    "diffusion_generate_recon": "diffusion", "diffusion_generate_t1": "diffusion",
    "control_generate_wrong_key": "control", "control_generate_wrong_nonce": "control",
    "control_reject_wrong_key": "control", "control_reject_wrong_nonce": "control",
    "cached_latent_to_m1": "stage_sum", "m1_to_reconstruction": "stage_sum",
}


def stage_base(stage: str) -> tuple[str, str]:
    """Split a stage name into (base stage, cell): ``*_legacy`` is the legacy-PNG cell, else ``main``."""
    if stage.endswith("_legacy"):
        return stage[: -len("_legacy")], "legacy"
    return stage, "main"


def stage_kind(stage: str) -> str:
    if stage.startswith("micro_"):
        return "microbenchmark"
    return "control" if stage.startswith("control_") else "pipeline"


# --------------------------------------------------------------------------- array evidence
# kind -> (dtype, methods stored along axis 1; () means one array per sample)
ARRAY_KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "x0": ("float32", ()),
    "anonymous_float": ("float32", METHODS),
    "anonymous_public_q": ("uint8", METHODS),  # 8-bit stored pixels q, common to every method, no lo/hi
    # What a viewer of the stored object can actually reconstruct: P/S0/S1 = range-restored PNG (lo/hi
    # public); S2 = q/255 only (lo/hi encrypted).  The public information is deliberately asymmetric.
    "anonymous_public_available": ("float32", METHODS),
    "recon_m1": ("float32", METHODS),
    "recon_t1": ("float32", METHODS),
    "recon_wrong_key": ("float32", ("S0", "S1", "S2")),
    "recon_wrong_nonce": ("float32", ("S2",)),
    "recon_m1_legacy": ("float32", METHODS),
}
IMAGE_SHAPE = (256, 256)


class ArrayStore:
    """Pre-allocated ``.npy`` memmaps so that a long run keeps partial evidence on disk."""

    def __init__(self, directory: Path, count: int, kinds: Sequence[str]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.kinds = tuple(kinds)
        self.maps: dict[str, Any] = {}
        for kind in self.kinds:
            dtype, methods = ARRAY_KINDS[kind]
            shape = (count, *( (len(methods),) if methods else ()), *IMAGE_SHAPE)
            self.maps[kind] = np.lib.format.open_memmap(
                self.directory / f"{kind}.npy", mode="w+", dtype=dtype, shape=shape,
            )

    def put(self, kind: str, index: int, array: np.ndarray, method: str | None = None) -> None:
        dtype, methods = ARRAY_KINDS[kind]
        value = np.asarray(array).reshape(IMAGE_SHAPE).astype(dtype, copy=False)
        if dtype.startswith("float") and not np.isfinite(value).all():
            raise ValueError(f"non-finite values for array {kind}[{index}]")
        if methods:
            self.maps[kind][index, methods.index(method)] = value
        else:
            self.maps[kind][index] = value

    def close(self) -> dict[str, Any]:
        """Flush, then return the per-array SHA-256 and per-slice hash index."""
        for memmap in self.maps.values():
            memmap.flush()
        evidence = {}
        for kind in self.kinds:
            path = self.directory / f"{kind}.npy"
            evidence[kind] = {
                "file": f"{self.directory.name}/{kind}.npy", "sha256": sha256_file(path),
                "methods": list(ARRAY_KINDS[kind][1]), "shape": list(self.maps[kind].shape),
                "dtype": ARRAY_KINDS[kind][0], "slice_sha256": slice_hashes(self.maps[kind]),
            }
        return evidence


def slice_hashes(array: np.ndarray) -> list[Any]:
    """SHA-256 of every sample (or sample x method) slice, in storage order."""
    if array.ndim not in (3, 4):
        raise ValueError("expected (N,H,W) or (N,M,H,W) array")
    if array.ndim == 3:
        return [hashlib.sha256(np.ascontiguousarray(array[i]).tobytes()).hexdigest() for i in range(array.shape[0])]
    return [
        [hashlib.sha256(np.ascontiguousarray(array[i, j]).tobytes()).hexdigest() for j in range(array.shape[1])]
        for i in range(array.shape[0])
    ]


def load_verified_arrays(run_dir: Path, evidence: dict[str, Any]) -> dict[str, np.ndarray]:
    """Reload every array read-only and verify file and per-slice hashes (fail closed)."""
    loaded = {}
    for kind, entry in evidence.items():
        path = Path(run_dir) / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"array file hash mismatch: {kind}")
        array = np.load(path, mmap_mode="r")
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
            raise ValueError(f"array shape/dtype mismatch: {kind}")
        if slice_hashes(array) != entry["slice_sha256"]:
            raise ValueError(f"array slice hash mismatch: {kind}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"non-finite values in array: {kind}")
        loaded[kind] = array
    return loaded


# --------------------------------------------------------------------------- storage accounting
def container_storage_breakdown(serialized: bytes) -> dict[str, int | str]:
    """Split one serialized S2 container into logical byte components.

    ``payload`` is the raw ciphertext (PNG bytes for M1, float latent bytes for
    M2); ``private_metadata`` is the raw AEAD ciphertext in ``parameters``;
    ``manifest`` is the canonical JSON of version, public parameters, AAD and the
    storage record; ``encoding_overhead`` is the non-negative residual that the
    hex/base64/JSON framing adds, so the components always sum to ``total``.
    """
    from src.crypto.container import deserialize_container, NONCE_BYTES, TAG_BYTES
    container = deserialize_container(serialized)
    parameters = dict(container.parameters)
    sealed_metadata = parameters.pop("encrypted_private_metadata", None)
    private_bytes = len(base64.b64decode(sealed_metadata, validate=True)) if sealed_metadata else 0
    manifest = json.dumps(
        {"version": container.version, "parameters": parameters, "aad": dict(container.aad),
         "storage": dict(container.storage)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    mode = container.storage["mode"]
    parts = {
        "nonce_bytes": NONCE_BYTES, "tag_bytes": TAG_BYTES, "manifest_bytes": len(manifest),
        "private_metadata_bytes": private_bytes,
        "latent_payload_bytes": len(container.ciphertext) if mode == "M2" else 0,
        "image_payload_bytes": len(container.ciphertext) if mode == "M1" else 0,
    }
    overhead = len(serialized) - sum(parts.values())
    if overhead < 0:
        raise ValueError("storage components exceed the serialized container size")
    return {"mode": mode, **parts, "encoding_overhead_bytes": overhead, "total_bytes": len(serialized)}
