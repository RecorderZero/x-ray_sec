#!/usr/bin/env python3
"""T2-BB anon-to-anon linkage using frozen encoders on exported M1 PNGs only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
SEED = 1911
ENCODERS = ("densenet121_imagenet", "rad_dino")
RAD_DINO_ID = "microsoft/rad-dino"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    values = np.concatenate([pos, neg])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average_ranks = (cumulative - counts + cumulative + 1) / 2.0
    positive_rank_sum = average_ranks[inverse[: len(pos)]].sum()
    return float((positive_rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _metrics(similarity: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    n_query, n_gallery = similarity.shape
    ranks = []
    positives, negatives = [], []
    for query in range(n_query):
        target = int(truth[query])
        order = np.argsort(-similarity[query], kind="stable")
        ranks.append(int(np.flatnonzero(order == target)[0]))
        positives.append(similarity[query, target])
        negatives.extend(np.delete(similarity[query], target))
    ranks_array = np.asarray(ranks)
    return {
        "top1": float(np.mean(ranks_array < 1)),
        "top5": float(np.mean(ranks_array < min(5, n_gallery))),
        "auc": auc_rank(np.asarray(positives), np.asarray(negatives)),
    }


def evaluate_similarity(similarity: np.ndarray, *, repeats: int = 5000, seed: int = SEED) -> dict[str, float]:
    """Conditional-query bootstrap and label-permutation controls.

    The gallery cohort stays fixed.  Consequently the interval describes query
    sampling uncertainty conditional on this N-patient gallery; it is not a
    population-level cohort bootstrap and is labelled accordingly in outputs.
    """
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1] or similarity.shape[0] < 2:
        raise ValueError("expected a square similarity matrix with at least two patients")
    n = similarity.shape[0]
    truth = np.arange(n)
    observed = _metrics(similarity, truth)
    rng = np.random.default_rng(seed)
    boot_top1, boot_auc = np.empty(repeats), np.empty(repeats)
    perm_top1, perm_auc = np.empty(repeats), np.empty(repeats)
    for index in range(repeats):
        selected = rng.integers(0, n, size=n)
        boot = _metrics(similarity[selected], truth[selected])
        boot_top1[index], boot_auc[index] = boot["top1"], boot["auc"]
        permutation = rng.permutation(n)
        null = _metrics(similarity, permutation)
        perm_top1[index], perm_auc[index] = null["top1"], null["auc"]
    observed.update({
        "top1_ci_low": float(np.quantile(boot_top1, 0.025)),
        "top1_ci_high": float(np.quantile(boot_top1, 0.975)),
        "auc_ci_low": float(np.quantile(boot_auc, 0.025)),
        "auc_ci_high": float(np.quantile(boot_auc, 0.975)),
        "top1_permutation_p": float((1 + np.sum(perm_top1 >= observed["top1"])) / (repeats + 1)),
        "auc_permutation_p": float((1 + np.sum(perm_auc >= observed["auc"])) / (repeats + 1)),
        "negative_top1": float(perm_top1[0]),
        "negative_auc": float(perm_auc[0]),
        "null_top1_mean": float(perm_top1.mean()),
        "null_top1_ci_low": float(np.quantile(perm_top1, 0.025)),
        "null_top1_ci_high": float(np.quantile(perm_top1, 0.975)),
        "null_auc_mean": float(perm_auc.mean()),
        "null_auc_ci_low": float(np.quantile(perm_auc, 0.025)),
        "null_auc_ci_high": float(np.quantile(perm_auc, 0.975)),
        "random_top1": 1.0 / n,
        "ci_scope": "conditional_query_bootstrap_with_fixed_gallery",
        "bootstrap_repeats": repeats,
        "permutation_repeats": repeats,
    })
    return observed


def _contained_path(base: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RuntimeError(f"absolute artifact path is forbidden: {relative}")
    resolved_base = base.resolve()
    resolved = (base / candidate).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise RuntimeError(f"artifact path escapes export directory: {relative}")
    return resolved


def load_export(export_dir: Path) -> tuple[dict[str, Any], dict[str, list[tuple[dict[str, Any], Image.Image]]]]:
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "t2bb_m1_public/v2":
        raise RuntimeError("not a T2-BB public M1 export")
    grouped: dict[str, list[tuple[dict[str, Any], Image.Image]]] = {}
    for item in manifest["artifacts"]:
        forbidden = {"sample_id", "patient_id", "pair_id", "pair_role"} & set(item)
        if forbidden:
            raise RuntimeError(f"public manifest leaks evaluation truth: {sorted(forbidden)}")
        path = _contained_path(export_dir, item["relative_path"])
        if sha256_file(path) != item["png_sha256"]:
            raise RuntimeError(f"public artifact SHA mismatch: {path}")
        with Image.open(path) as handle:
            image = handle.convert("RGB").copy()
        grouped.setdefault(item["method"], []).append((item, image))
    for method, entries in grouped.items():
        if len(entries) != manifest["images"]:
            raise RuntimeError(f"incomplete method export: {method}")
    return manifest, grouped


def load_evaluation_map(path: Path, manifest_path: Path,
                        expected_split_sha256: str | None = None) -> dict[str, dict[str, str]]:
    private = json.loads(path.read_text(encoding="utf-8"))
    if private.get("schema") != "t2bb_m1_private_eval/v1":
        raise RuntimeError("not a T2-BB private evaluation map")
    if private["source_export_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("private evaluation map/public manifest SHA mismatch")
    if expected_split_sha256 is not None and private.get("source_split_sha256") != expected_split_sha256:
        raise RuntimeError("private evaluation map/public split SHA mismatch")
    truth = {}
    for entry in private["entries"]:
        relative = entry["relative_path"]
        if relative in truth:
            raise RuntimeError(f"duplicate private truth entry: {relative}")
        truth[relative] = entry
    pairs: dict[tuple[str, str], list[dict[str, str]]] = {}
    for entry in truth.values():
        pairs.setdefault((entry["method"], entry["pair_id"]), []).append(entry)
    for key, members in pairs.items():
        if len(members) != 2 or {member["pair_role"] for member in members} != {"gallery", "query"}:
            raise RuntimeError(f"private truth pair is incomplete: {key}")
        if len({member["patient_id"] for member in members}) != 1:
            raise RuntimeError(f"private truth patient mismatch: {key}")
    return truth


@torch.inference_mode()
def densenet_embeddings(images: list[Image.Image], device: str, batch_size: int) -> np.ndarray:
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).eval().to(device)
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensors = [preprocessing(image) for image in images]
    output = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start:start + batch_size]).to(device)
        features = F.relu(model.features(batch))
        output.append(F.adaptive_avg_pool2d(features, 1).flatten(1).cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def rad_dino_embeddings(images: list[Image.Image], device: str, batch_size: int, model_id: str) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True).eval().to(device)
    output = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        result = model(**inputs)
        features = result.pooler_output if result.pooler_output is not None else result.last_hidden_state[:, 0]
        output.append(features.float().cpu().numpy())
    return np.concatenate(output)


def paired_similarity(entries: list[tuple[dict[str, Any], Image.Image]], embeddings: np.ndarray,
                      truth_by_path: dict[str, dict[str, str]]) -> np.ndarray:
    gallery, query = {}, {}
    for (item, _image), embedding in zip(entries, embeddings):
        truth = truth_by_path[item["relative_path"]]
        if truth["method"] != item["method"]:
            raise RuntimeError("private/public method mismatch")
        target = gallery if truth["pair_role"] == "gallery" else query
        target[truth["pair_id"]] = embedding
    pairs = sorted(gallery)
    if set(pairs) != set(query):
        raise RuntimeError("gallery/query pair mismatch")
    g = l2_normalize(np.stack([gallery[pair] for pair in pairs]))
    q = l2_normalize(np.stack([query[pair] for pair in pairs]))
    return q @ g.T


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def encoder_provenance(encoder: str, model_id: str) -> str:
    if encoder == "densenet121_imagenet":
        weights = models.DenseNet121_Weights.DEFAULT
        filename = Path(urllib.parse.urlparse(weights.url).path).name
        weight_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        return json.dumps({
            "package": "torchvision", "version": importlib.metadata.version("torchvision"),
            "weight_enum": str(weights), "weight_url": weights.url,
            "weight_sha256": sha256_file(weight_path),
        }, sort_keys=True)
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model_id, local_files_only=True))
    files = {}
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        candidate = snapshot / name
        if candidate.exists():
            files[name] = sha256_file(candidate)
    return json.dumps({
        "package": "transformers", "version": importlib.metadata.version("transformers"),
        "model_id": model_id, "resolved_revision": snapshot.name, "file_sha256": files,
    }, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--export-dir", type=Path, default=ROOT / "artifacts/t2bb_m1_public_v1")
    parser.add_argument("--eval-map", type=Path, default=None,
                        help="private truth map, kept outside the attacker-visible export")
    parser.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_t2bb_m1_frozen.csv")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--rad-dino-model", default=RAD_DINO_ID)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    eval_map = args.eval_map or args.export_dir.with_name(f"{args.export_dir.name}_private_eval.json")
    json_out = args.json_out or args.out.with_suffix(".json")
    if not args.force and (args.out.exists() or json_out.exists()):
        raise SystemExit("refusing to overwrite results without --force")

    manifest, grouped = load_export(args.export_dir)
    truth_by_path = load_evaluation_map(
        eval_map, args.export_dir / "manifest.json", manifest["provenance"]["split_sha256"]
    )
    public_paths = {item[0]["relative_path"] for entries in grouped.values() for item in entries}
    if public_paths != set(truth_by_path):
        raise RuntimeError("private evaluation map does not exactly cover the public export")
    rows = []
    for method in manifest["methods"]:
        entries = grouped[method]
        images = [entry[1] for entry in entries]
        for encoder in args.encoders:
            if encoder == "densenet121_imagenet":
                embeddings = densenet_embeddings(images, args.device, args.batch_size)
                model_provenance = encoder_provenance(encoder, args.rad_dino_model)
            else:
                embeddings = rad_dino_embeddings(images, args.device, args.batch_size, args.rad_dino_model)
                model_provenance = encoder_provenance(encoder, args.rad_dino_model)
            metrics = evaluate_similarity(
                paired_similarity(entries, embeddings, truth_by_path), repeats=args.repeats
            )
            rows.append({
                "task": "T2-BB-M1-Frozen", "evidence_role": manifest["evidence_role"],
                "threat_model": manifest["threat_model"], "method": method,
                "storage": manifest["storage_policy"][method], "encoder": encoder,
                "encoder_provenance": model_provenance, "patients": manifest["patients"],
                **metrics,
            })
            print(f"{method:>2} {encoder:<22} top1={metrics['top1']:.3f} auc={metrics['auc']:.3f}", flush=True)
    write_csv(args.out, rows)
    payload = {
        "schema": "t2bb_m1_frozen_linkage/v1", "seed": SEED,
        "source_export_manifest_sha256": sha256_file(args.export_dir / "manifest.json"),
        "private_eval_map_sha256": sha256_file(eval_map),
        "source_export": str(args.export_dir), "results": rows,
        "runtime": {"device": args.device, "batch_size": args.batch_size,
                    "script_sha256": sha256_file(Path(__file__))},
        "interpretation_rule": "P is the unprotected pipeline control; S0/S1/S2 are compared only when P has signal. Pilot N=20 is exploratory.",
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_out.with_name(f".{json_out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(json_out)
    print(f"wrote {args.out} and {json_out}")


if __name__ == "__main__":
    main()
