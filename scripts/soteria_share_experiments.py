#!/usr/bin/env python3
"""S10 GPU runner for receiver-bound sharing: E1 fidelity and E2 cross-recipient linkage.

E1 (reports/cross_institution_sharing.md §8.8, P4): on security_v1 (N=200), run the
real DPM-Solver++(2M) n=100 null-backward pipeline through the v2 protocol
(A: x0 -> invert -> fixed-norm DPHD keyed by an HPKE-wrapped image key -> generate
-> private-range PNG -> signed container; B: verify -> unwrap -> invert -> DPHD^-1
-> generate) and pair every sample with the frozen v1 S2fn result
(paper_assets/rq4_formal_dphd/per_sample_long.csv) by sample_id + patient_id.
Pre-registered: the patient-bootstrap 95% CI of mean dPSNR (v2 - v1) must lie
inside +-0.5 dB to write "mean reconstruction PSNR within the pre-set bound";
nothing is extended to SSIM/LPIPS, per-patient, lesion or security equivalence.

Metrics follow scripts/rq4_formal_redo.py exactly (image_pair_metrics, LPIPS alex on
CPU, q8_metrics.summarize: percentile bootstrap seed 1911, 10000 resamples,
cluster=patient). Before any image is processed the runner checks that the real
checkpoint SHA-256 and the live sampler schedule equal the signed manifest.

Keys: PUBLIC labelled test fixtures (TestFixtureStream "S10/E1/<sample_id>/B") so the
anonymous images are reproducible; HPKE ephemeral keys stay random. Containers are
kept under artifacts/runs/ (git-ignored) for E2 reuse.

  smoke : scripts/run_cfg_ddim.sh python -m scripts.soteria_share_experiments --mode e1 --evidence-role smoke --limit 4
  formal: scripts/run_cfg_ddim.sh python -m scripts.soteria_share_experiments --mode e1 --evidence-role formal

E2 (§8.8, F3=(a)): reuse E1's B containers as gallery, generate C versions of the same
x0 (fresh fixture image key) as queries; black-box scorers cos / norm / DenseNet121 on
the public PNG only (q/255). AUC CI is conditional on the fixed gallery (query-patient
block bootstrap, 2000x). Positive control: plaintext x0 copies.

  smoke : ... --mode e2 --evidence-role smoke --limit 4      (needs the E1 smoke containers)
  formal: ... --mode e2 --evidence-role formal               (needs the E1 formal containers)

AF-056/057 (2026-11-04): E2 scoring uses the committed frozen evaluator
scripts/share_linkage_eval.py; E1/E2 write research-private container manifests
(sample/patient -> SHA-256) that E2 verifies before loading the model; existing formal
containers get a clearly labelled retroactive manifest via --mode backfill-manifests;
--mode e2-analysis replays E2 scoring from verified containers without diffusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts import q8_metrics
from scripts.e1_ddim_runner import create_runtime, preprocess
from scripts.e2_anonymization_runner import image_pair_metrics
from scripts.exact_inv_sweep import CKPT, DPMSolverpp
from scripts.output_guard import add_force_flag, refuse_overwrite
from scripts.respaced_ddim_sweep import SEED, Respaced, schedule
from scripts.rq4_formal_redo import IMG_METRICS, _img_metrics, _to_hw, git_head, sha256_file
from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE
from src.share.keys import TestFixtureStream, fixture_x25519
from src.share.manifest import PROFILE_MANIFESTS, sign_manifest
from src.share.receiver import RecipientIdentity, recover_image
from src.share.registry import KeyRecord, TrustStore, build_snapshot
from src.share.sender import SenderIdentity, share_image


V1_PER_SAMPLE = Path("paper_assets/rq4_formal_dphd/per_sample_long.csv")
V1_SCHEME = "S2fn"
DELTA_BOUND_DB = 0.5
MANIFEST = PROFILE_MANIFESTS[PROFILE]


def _ed25519(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(TestFixtureStream("ed25519/" + label).read(32))


class FixtureWorld:
    """A (sender), B, C (recipients), a root-signed registry valid around wall-clock now, A's signed manifest."""

    def __init__(self, state_dir: Path):
        root, a_sign = _ed25519("root"), _ed25519("A")
        self.sender = SenderIdentity("hospital-A", "A-ed25519-2026a", a_sign)
        self.b = RecipientIdentity("hospital-B", "B-x25519-2026a", fixture_x25519("B"))
        self.c = RecipientIdentity("hospital-C", "C-x25519-2026a", fixture_x25519("C"))
        records = [
            KeyRecord("hospital-A", "A-ed25519-2026a", "ed25519-sign", "active", a_sign.public_key().public_bytes_raw()),
            KeyRecord("hospital-B", "B-x25519-2026a", "x25519-recv", "active", self.b.private_key.public_key().public_bytes_raw()),
            KeyRecord("hospital-C", "C-x25519-2026a", "x25519-recv", "active", self.c.private_key.public_key().public_bytes_raw()),
        ]
        now = int(time.time())
        state_dir.mkdir(parents=True, exist_ok=True)
        state = state_dir / "trust_state.json"
        if state.exists():
            state.unlink()  # research run: fresh fixture trust state each invocation
        self.trust = TrustStore(root.public_key(), state)
        self.trust.load(build_snapshot(root, 1, now - 60, now + 7 * 86400, records))
        self.manifest_blob = sign_manifest(MANIFEST, a_sign)

    def fetch_manifest(self, digest: bytes) -> bytes:
        if digest != MANIFEST.digest():
            raise LookupError("unknown manifest digest")
        return self.manifest_blob


def check_runtime_binding(checkpoint: Path, steps: int, b: np.ndarray) -> dict:
    """Fail closed unless the real checkpoint and live sampler equal the signed manifest."""
    ckpt_sha = sha256_file(checkpoint)
    live = tuple(int(t) for t in b)
    problems = []
    if bytes.fromhex(ckpt_sha) != MANIFEST.checkpoint_sha256:
        problems.append(f"checkpoint sha256 {ckpt_sha} != manifest")
    if steps != MANIFEST.steps:
        problems.append(f"steps {steps} != manifest {MANIFEST.steps}")
    if live != MANIFEST.timesteps:
        problems.append("live timestep schedule != manifest schedule")
    if problems:
        raise SystemExit("runtime/manifest binding failed: " + "; ".join(problems))
    return {"checkpoint_sha256": ckpt_sha, "steps": steps, "timesteps_match": True,
            "backward": "null (Respaced backward_null=True)", "manifest_digest": MANIFEST.digest().hex()}


# --------------------------------------------------------------------------- container run manifests (AF-057)
def _aggregate(blobs: list[bytes]) -> str:
    return hashlib.sha256(b"".join(hashlib.sha256(x).digest() for x in blobs)).hexdigest()


def write_container_manifest(container_dir: Path, role: str, rows: list[dict], blobs: list[bytes], split: Path,
                             created_by: str, extra: dict | None = None) -> Path:
    """Research-private map sample/patient -> container SHA-256 (never put into the public header)."""
    if len(rows) != len(blobs) or not rows:
        raise SystemExit(f"{role}: rows ({len(rows)}) and containers ({len(blobs)}) must be equal and non-empty")
    entries = [{"sample_id": r["sample_id"], "patient_id": r["patient_id"], "file": f"{r['sample_id']}.soteria",
                "sha256": hashlib.sha256(b).hexdigest()} for r, b in zip(rows, blobs)]
    payload = {"schema_version": 1, "role": role, "created_by": created_by,
               "run_id": hashlib.sha256((role + created_by + str(time.time_ns())).encode()).hexdigest()[:16],
               "profile_id": PROFILE, "manifest_digest": MANIFEST.digest().hex(),
               "split": str(split), "split_sha256": sha256_file(split), "n": len(entries),
               "aggregate_sha256": _aggregate(blobs), "git": git_head(), "entries": entries, **(extra or {})}
    path = container_dir / "container_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_verified_containers(container_dir: Path, role: str, rows: list[dict], split: Path, recipient: str) -> tuple[list[bytes], dict]:
    """Fail closed unless every row maps to the manifest entry and file SHA-256 it was produced with."""
    from src.share.container import parse
    path = container_dir / "container_manifest.json"
    if not path.exists():
        raise SystemExit(f"{role}: container manifest missing at {path}")
    man = json.loads(path.read_text(encoding="utf-8"))
    if man.get("role") != role or man.get("profile_id") != PROFILE or man.get("manifest_digest") != MANIFEST.digest().hex():
        raise SystemExit(f"{role}: container manifest role/profile/manifest mismatch")
    if man.get("split_sha256") != sha256_file(split) or man.get("n") != len(rows):
        raise SystemExit(f"{role}: container manifest was built for a different split")
    if not isinstance(man.get("entries"), list) or len(man["entries"]) != len(rows) or man["n"] != len(man["entries"]):
        raise SystemExit(f"{role}: container manifest entry count does not equal n and the split rows")
    blobs = []
    for r, e in zip(rows, man["entries"]):
        if (e["sample_id"], e["patient_id"]) != (r["sample_id"], r["patient_id"]):
            raise SystemExit(f"{role}: manifest entry order/identity mismatch at {r['sample_id']}")
        blob = (container_dir / e["file"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != e["sha256"]:
            raise SystemExit(f"{role}: container {e['file']} does not match its manifest SHA-256")
        h = parse(blob).header
        if h.profile_id != PROFILE or h.manifest_digest != MANIFEST.digest() or h.inst_id_b != recipient:
            raise SystemExit(f"{role}: container {e['file']} profile/recipient mismatch")
        blobs.append(blob)
    if _aggregate(blobs) != man["aggregate_sha256"]:
        raise SystemExit(f"{role}: aggregate SHA-256 mismatch")
    return blobs, {"path": str(path), "run_id": man["run_id"], "created_by": man["created_by"],
                   "aggregate_sha256": man["aggregate_sha256"], "manifest_sha256": sha256_file(path)}


def _guard_container_dir(container_dir: Path, force: bool) -> None:
    if container_dir.exists() and any(container_dir.iterdir()) and not force:
        raise SystemExit(f"refusing to overwrite existing evidence in {container_dir} (use --force)")


def run_e1(args) -> None:
    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    if args.evidence_role == "smoke":
        rows = rows[: (args.limit or 4)]
    elif args.limit is not None:
        raise SystemExit("formal E1 uses the full split; --limit is smoke-only")
    if args.evidence_role == "formal" and len(rows) != 200:
        raise SystemExit(f"formal requires the full 200-image security_v1 (got {len(rows)})")
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise SystemExit("patient IDs are not unique")

    tag = "" if args.evidence_role == "formal" else f"_{args.evidence_role}_n{len(rows)}"
    out_csv = args.out_dir / f"table_share_fidelity_n200{tag}.csv" if not tag else args.out_dir / f"table_share_fidelity{tag}.csv"
    out_json = out_csv.with_suffix(".json")
    out_per = out_csv.with_name(out_csv.stem + "_per_sample.csv")
    refuse_overwrite([out_csv, out_json, out_per], args.force)

    v1 = {(r["sample_id"], r["patient_id"]): r for r in csv.DictReader(V1_PER_SAMPLE.open(encoding="utf-8"))
          if r["scheme"] == V1_SCHEME}
    missing = [r["sample_id"] for r in rows if (r["sample_id"], r["patient_id"]) not in v1]
    if missing:
        raise SystemExit(f"v1 per-sample rows missing for {missing[:5]}")

    lp_status = q8_metrics.lpips_status()
    if not lp_status["available"]:
        raise SystemExit(f"LPIPS required but unavailable: {lp_status['reason']}")
    lpips = q8_metrics.build_lpips_metric("cpu")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device, backward_null=True)
    dpm = DPMSolverpp(rs)
    b = schedule(args.steps)
    binding = check_runtime_binding(args.checkpoint, args.steps, b)
    rs.backward(rs.forward(preprocess(Path(rows[0]["local_path"]))[0].unsqueeze(0).to(device), schedule(10)), schedule(10))

    def invert(x: torch.Tensor) -> torch.Tensor:
        return dpm.forward(x.to(device, torch.float32), b)

    captured: dict[str, torch.Tensor] = {}

    def generate_a(z: torch.Tensor) -> torch.Tensor:
        anon = dpm.backward(z.to(device, torch.float32), b)
        captured["anon"] = anon
        return anon

    def generate_b(z: torch.Tensor) -> torch.Tensor:
        return dpm.backward(z.to(device, torch.float32), b)

    def factory(manifest):
        if manifest != MANIFEST or tuple(int(t) for t in b) != manifest.timesteps:
            raise SystemExit("verified manifest does not match the loaded pipeline")
        return invert, generate_b

    world = FixtureWorld(args.artifact_dir / f"e1{tag}")
    container_dir = args.artifact_dir / f"e1{tag}" / "containers_B"
    _guard_container_dir(container_dir, args.force)
    container_dir.mkdir(parents=True, exist_ok=True)
    b_blobs: list[bytes] = []

    records = []
    t_start = time.time()
    with torch.inference_mode():
        for i, r in enumerate(rows, 1):
            sid = r["sample_id"]
            x0 = preprocess(Path(r["local_path"]))[0].unsqueeze(0).to(device)
            torch.cuda.synchronize(); t0 = time.time()
            blob = share_image(x0, sender=world.sender, recipient_inst_id=world.b.inst_id, recipient_key_id=world.b.key_id,
                               registry=world.trust, manifest_digest=MANIFEST.digest(), invert=invert, generate=generate_a,
                               fixture=TestFixtureStream(f"S10/E1/{sid}/B"))
            torch.cuda.synchronize(); t1 = time.time()
            x_rec = recover_image(blob, recipient=world.b, registry=world.trust, fetch_manifest=world.fetch_manifest,
                                  pipeline_factory=factory).to(device)
            torch.cuda.synchronize(); t2 = time.time()
            (container_dir / f"{sid}.soteria").write_bytes(blob)
            b_blobs.append(blob)
            rec = _img_metrics(x_rec, x0)
            rec["single_roundtrip_psnr_db"] = float(image_pair_metrics("", captured["anon"], x0)["_psnr_db"])
            rec["lpips"] = lpips(_to_hw(x0), _to_hw(x_rec))
            ref = v1[(sid, r["patient_id"])]
            rec.update({
                "sample_id": sid, "patient_id": r["patient_id"], "label": r.get("label", ""),
                "v1_psnr_db": float(ref["psnr_db"]), "v1_ssim": float(ref["ssim"]), "v1_lpips": float(ref["lpips"]),
                "sender_seconds": t1 - t0, "receiver_seconds": t2 - t1, "container_bytes": len(blob),
                "finite": bool(torch.isfinite(x_rec).all()),
            })
            for m in ("psnr_db", "ssim", "lpips"):
                rec[f"delta_{m}"] = rec[m] - rec[f"v1_{m}"]
            records.append(rec)
            if i % 10 == 0 or i == len(rows):
                el = time.time() - t_start
                print(f"  {i}/{len(rows)}  elapsed {el:.0f}s  ~{el / i:.2f}s/img", flush=True)

    write_container_manifest(container_dir, "E1-B", rows, b_blobs, args.split, created_by="runner E1")
    clusters = [r["patient_id"] for r in records]
    summary_rows = []
    for key in [*IMG_METRICS, "single_roundtrip_psnr_db", "lpips", "delta_psnr_db", "delta_ssim", "delta_lpips",
                "sender_seconds", "receiver_seconds", "container_bytes"]:
        s = q8_metrics.summarize([r[key] for r in records], clusters)
        summary_rows.append({"method": "v2_receiver_bound", "metric": key, **s})
    dpsnr = next(r for r in summary_rows if r["metric"] == "delta_psnr_db")
    within = bool(-DELTA_BOUND_DB <= dpsnr["ci95_low"] and dpsnr["ci95_high"] <= DELTA_BOUND_DB)
    worst = sorted(records, key=lambda r: r["delta_psnr_db"])[:5]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(summary_rows)
    with out_per.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader(); w.writerows(records)
    payload = {
        "schema_version": 1, "experiment": "E1", "evidence_role": args.evidence_role, "n": len(records),
        "plan": "reports/cross_institution_sharing.md §8.8 P4/E1",
        "pre_registered": f"patient-bootstrap 95% CI of mean dPSNR (v2 - v1 {V1_SCHEME}) inside +-{DELTA_BOUND_DB} dB",
        "result": {"delta_psnr_db": dpsnr, "within_preregistered_bound": within,
                   "claim_if_within": "mean reconstruction PSNR within the pre-set bound (no extension to SSIM/LPIPS/"
                                      "per-patient/lesion/security)"},
        "worst_delta_psnr_samples": [{k: r[k] for k in ("sample_id", "psnr_db", "v1_psnr_db", "delta_psnr_db")} for r in worst],
        "v1_reference": {"file": str(V1_PER_SAMPLE), "scheme": V1_SCHEME,
                         "provenance_note": "frozen rq4_formal_dphd result; see paper_assets/rq4_formal_dphd/git_dirty_provenance.md",
                         "v1_marginal_psnr_mean_db": 46.39854044572381},
        "runtime_binding": binding,
        "operating_point": {"sampler": "DPM-Solver++(2M)", "steps": args.steps, "backward": "null",
                            "storage": "range_preserving_png_private_range/v2 (8-bit, lo/hi in AEAD metadata)"},
        "keys": "PUBLIC test fixtures (S10/E1/<sample_id>/B); HPKE ephemeral random",
        "ci_method": q8_metrics.CI_METHOD, "bootstrap_resamples": q8_metrics.BOOTSTRAP_RESAMPLES,
        "lpips": lp_status, "split": str(args.split), "split_sha256": sha256_file(args.split),
        "timing_note": "sender/receiver seconds measured in this run; compare to other runtimes only with matching conditions",
        "wall_seconds_total": time.time() - t_start, "git": git_head(), "gpu": torch.cuda.get_device_name(0),
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nE1 ({args.evidence_role}, n={len(records)}): v2 PSNR mean "
          f"{next(r for r in summary_rows if r['metric'] == 'psnr_db')['mean']:.3f} dB; "
          f"dPSNR {dpsnr['mean']:+.3f} [{dpsnr['ci95_low']:+.3f}, {dpsnr['ci95_high']:+.3f}] dB; within +-{DELTA_BOUND_DB}: {within}")
    print(f"wrote {out_csv}, {out_per}, {out_json}")


E2_SCORERS = ("cos", "norm", "densenet121")   # pre-registered (§8.8 E2); score direction: higher = same image


def _attacker_arrays(containers: list[bytes]) -> list[np.ndarray]:
    """Black-box view: only the public PNG, decoded as q/255 (no lo/hi, no keys, no ids)."""
    from scripts.m1_storage import decode_private_png_attacker_view
    from src.share.container import parse
    return [decode_private_png_attacker_view(parse(c).png).astype(np.float32) for c in containers]


def score_e2(b_blobs: list[bytes], c_blobs: list[bytes], x0s: list[np.ndarray], rows: list[dict], device: str) -> list[dict]:
    """Frozen E2 scoring (scripts/share_linkage_eval.py): gallery = B, query = C."""
    from scripts.share_linkage_eval import evaluate, similarity
    pid = [r["patient_id"] for r in rows] * 2
    roles = ["gallery"] * len(rows) + ["query"] * len(rows)
    conditions = {
        "v2_public_png": _attacker_arrays(b_blobs) + _attacker_arrays(c_blobs),
        "positive_control_plaintext_x0": x0s + x0s,
    }
    out_rows = []
    for condition, arrays in conditions.items():
        for scorer in E2_SCORERS:
            sim = similarity(arrays, scorer, device)
            res = evaluate(sim, pid, roles, np.random.default_rng(SEED))
            out_rows.append({"condition": condition, "scorer": scorer, "n_patients": len(rows),
                             "n_positive_pairs": len(rows), "n_negative_pairs": len(rows) * (len(rows) - 1), **res})
            print(f"  {condition:32s} {scorer:12s} AUC {res['auc']:.6f} [{res['auc_ci_low']}, {res['auc_ci_high']}] "
                  f"top1 {res['top1']:.3f} (random {res['random_top1']:.3f}, shuffled {res['neg_control_top1']:.3f})")
    return out_rows


def scoring_provenance() -> dict:
    import torchvision
    from torchvision import models
    weights = models.DenseNet121_Weights.DEFAULT
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    return {"evaluator_module": "scripts/share_linkage_eval.py",
            "evaluator_module_sha256": sha256_file(Path("scripts/share_linkage_eval.py")),
            "evaluator_origin": "verbatim copy of scripts/t2bb_inner_linkage.py working tree sha256 "
                                "b02f78b6d10406da9dbf9523778c51b8acee64aaa4243bddec8e9205e14ba055",
            "densenet121_weights_url": str(weights.url),
            "densenet121_weights_sha256": sha256_file(cache) if cache.exists() else "not-cached",
            "densenet_preprocess": "per-image min-max -> uint8 -> Grayscale(3) -> Resize(224, bicubic) -> ToTensor "
                                   "-> ImageNet Normalize; features -> ReLU -> global avg pool; L2-normalized cosine",
            "torch": torch.__version__, "torchvision": torchvision.__version__}


def _e2_design(n: int) -> dict:
    return {"gallery": "B versions (reused from E1 containers, verified against the E1 container manifest)",
            "query": "C versions (same x0, fresh image key)",
            "positives": "same x0 B/C", "negatives": "different-patient B/C",
            "random_top1": 1 / n, "random_top5": 5 / n,
            "attacker_view": "public PNG only, decoded as q/255 (decode_private_png_attacker_view); "
                             "no sample/patient ids, paths, private metadata or keys",
            "routing": "all gallery items carry B routing, all queries C routing (same for positives and negatives)",
            "scorers": list(E2_SCORERS), "score_direction": "higher similarity = same image",
            "ties": "numpy argsort order for ranks; AUC uses average ranks for ties",
            "ci": "conditional-on-fixed-gallery: bootstrap of query-patient blocks, 2000x, seed 1911; "
                  "does not cover gallery-population uncertainty",
            "controls": "positive = plaintext x0 copies; negative = shuffled query->gallery pairing (neg_control_top1)"}


def _rows_for(args) -> tuple[list[dict], str]:
    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    if args.evidence_role == "smoke":
        rows = rows[: (args.limit or 4)]
    elif args.limit is not None:
        raise SystemExit("formal E2 uses the full split; --limit is smoke-only")
    if args.evidence_role == "formal" and len(rows) != 200:
        raise SystemExit(f"formal requires the full 200-image security_v1 (got {len(rows)})")
    return rows, ("" if args.evidence_role == "formal" else f"_{args.evidence_role}_n{len(rows)}")


def run_e2(args) -> None:
    rows, tag = _rows_for(args)
    out_csv = args.out_dir / f"table_share_cross_recipient_linkage{tag}.csv"
    out_json = out_csv.with_suffix(".json")
    refuse_overwrite([out_csv, out_json], args.force)
    b_blobs, b_source = load_verified_containers(args.artifact_dir / f"e1{tag}" / "containers_B", "E1-B", rows,
                                                 args.split, "hospital-B")          # before any model load (AF-057)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")
    _, diffusion, model_fn = create_runtime(Path("past/SourceCode"), args.checkpoint, device)
    rs = Respaced(diffusion, model_fn, device, backward_null=True)
    dpm = DPMSolverpp(rs)
    b = schedule(args.steps)
    binding = check_runtime_binding(args.checkpoint, args.steps, b)

    world = FixtureWorld(args.artifact_dir / f"e2{tag}")
    c_dir = args.artifact_dir / f"e2{tag}" / "containers_C"
    _guard_container_dir(c_dir, args.force)
    c_dir.mkdir(parents=True, exist_ok=True)
    c_blobs, x0s, t_gen = [], [], time.time()
    with torch.inference_mode():
        for i, r in enumerate(rows, 1):
            x0 = preprocess(Path(r["local_path"]))[0].unsqueeze(0).to(device)
            x0s.append(x0.detach().cpu().numpy().reshape(256, 256).astype(np.float32))
            blob = share_image(x0, sender=world.sender, recipient_inst_id=world.c.inst_id, recipient_key_id=world.c.key_id,
                               registry=world.trust, manifest_digest=MANIFEST.digest(),
                               invert=lambda x: dpm.forward(x.to(device, torch.float32), b),
                               generate=lambda z: dpm.backward(z.to(device, torch.float32), b),
                               fixture=TestFixtureStream(f"S10/E2/{r['sample_id']}/C"))
            (c_dir / f"{r['sample_id']}.soteria").write_bytes(blob)
            c_blobs.append(blob)
            if i % 20 == 0 or i == len(rows):
                el = time.time() - t_gen
                print(f"  C {i}/{len(rows)}  elapsed {el:.0f}s  ~{el / i:.2f}s/img", flush=True)
    gen_seconds = time.time() - t_gen
    write_container_manifest(c_dir, "E2-C", rows, c_blobs, args.split, created_by="runner E2")
    t_an = time.time()
    out_rows = score_e2(b_blobs, c_blobs, x0s, rows, "cuda")
    _write_e2(out_csv, out_json, out_rows, args, rows, {
        "b_source": b_source, "c_aggregate_sha256": _aggregate(c_blobs), "runtime_binding": binding,
        "c_generation_seconds": gen_seconds, "analysis_seconds": time.time() - t_an,
        "gpu": torch.cuda.get_device_name(0)})


def _write_e2(out_csv: Path, out_json: Path, out_rows: list[dict], args, rows: list[dict], extra: dict) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(out_rows)
    payload = {
        "schema_version": 2, "experiment": "E2", "evidence_role": args.evidence_role, "n_patients": len(rows),
        "plan": "reports/cross_institution_sharing.md §8.8 E2 (F3=(a))", "design": _e2_design(len(rows)),
        "reading": "AUC/Top-k are scientific results, not an engineering pass gate; a CI containing 0.5 only means no "
                   "linkage signal was detected with these scorers and samples -- not a proof of unlinkability.",
        "scoring": scoring_provenance(), "keys": "PUBLIC test fixtures (S10/E1/<sid>/B, S10/E2/<sid>/C)",
        "split": str(args.split), "split_sha256": sha256_file(args.split), "git": git_head(), **extra, "rows": out_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}, {out_json}")


def run_e2_analysis(args) -> None:
    """AF-056: CPU-capable replay of E2 scoring from existing, manifest-verified B/C containers (no diffusion)."""
    rows, tag = _rows_for(args)
    out_csv = args.out_dir / f"table_share_cross_recipient_linkage{tag}_replay_{args.device}.csv"
    out_json = out_csv.with_suffix(".json")
    refuse_overwrite([out_csv, out_json], args.force)
    b_blobs, b_source = load_verified_containers(args.artifact_dir / f"e1{tag}" / "containers_B", "E1-B", rows,
                                                 args.split, "hospital-B")
    c_blobs, c_source = load_verified_containers(args.artifact_dir / f"e2{tag}" / "containers_C", "E2-C", rows,
                                                 args.split, "hospital-C")
    x0s = [preprocess(Path(r["local_path"]))[0].numpy().reshape(256, 256).astype(np.float32) for r in rows]
    t_an = time.time()
    out_rows = score_e2(b_blobs, c_blobs, x0s, rows, args.device)
    comparison = None
    if args.reference is not None and args.reference.exists():
        ref = {(r["condition"], r["scorer"]): r for r in json.loads(args.reference.read_text())["rows"]}
        comparison = []
        for r in out_rows:
            o = ref[(r["condition"], r["scorer"])]
            comparison.append({"condition": r["condition"], "scorer": r["scorer"],
                               **{f"abs_diff_{k}": abs(float(r[k]) - float(o[k]))
                                  for k in ("auc", "auc_ci_low", "auc_ci_high", "top1", "top5", "neg_control_top1")}})
    _write_e2(out_csv, out_json, out_rows, args, rows, {
        "mode": "analysis-only replay (no diffusion)", "device": args.device,
        "b_source": b_source, "c_source": c_source, "analysis_seconds": time.time() - t_an,
        "reference": str(args.reference) if args.reference else None, "comparison_to_reference": comparison})


def run_backfill(args) -> None:
    """AF-057: build manifests for containers produced before manifests existed; verified against the formal E2 JSON."""
    rows, tag = _rows_for(args)
    ref = json.loads(args.reference.read_text())
    # AF-057: bind the backfill to the exact run being referenced BEFORE anything is written.
    problems = []
    if ref.get("experiment") != "E2":
        problems.append("reference is not an E2 result")
    if ref.get("evidence_role") != args.evidence_role:
        problems.append(f"reference evidence_role {ref.get('evidence_role')!r} != {args.evidence_role!r}")
    if ref.get("split_sha256") != sha256_file(args.split):
        problems.append("current split SHA-256 differs from the split recorded by the reference run")
    if ref.get("n_patients") != len(rows):
        problems.append(f"reference n_patients {ref.get('n_patients')} != {len(rows)} rows")
    if problems:
        raise SystemExit("backfill refused: " + "; ".join(problems))
    for role, sub, key in (("E1-B", f"e1{tag}/containers_B", "b_containers_sha256"),
                           ("E2-C", f"e2{tag}/containers_C", "c_containers_sha256")):
        d = args.artifact_dir / sub
        if (d / "container_manifest.json").exists():
            print(f"{role}: manifest already present, left unchanged")
            continue
        blobs = [(d / f"{r['sample_id']}.soteria").read_bytes() for r in rows]
        agg = _aggregate(blobs)
        if agg != ref[key]:
            raise SystemExit(f"{role}: on-disk containers do not match the aggregate recorded by the formal E2 run")
        write_container_manifest(d, role, rows, blobs, args.split,
                                 created_by="retroactive backfill 2026-11-04 (AF-057); not produced by the original run",
                                 extra={"verified_against": str(args.reference), "verified_field": key})
        print(f"{role}: backfilled manifest; aggregate matches {key} in {args.reference}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--mode", choices=["e1", "e2", "e2-analysis", "backfill-manifests"], required=True)
    ap.add_argument("--device", default="cpu", help="e2-analysis scoring device")
    ap.add_argument("--reference", type=Path, default=Path("paper_assets/tables/table_share_cross_recipient_linkage.json"),
                    help="formal E2 JSON to compare against (e2-analysis) / verify against (backfill-manifests)")
    ap.add_argument("--evidence-role", choices=["smoke", "formal"], default="smoke")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/tables"))
    ap.add_argument("--artifact-dir", type=Path, default=Path("artifacts/runs/share_s10"))
    add_force_flag(ap)
    args = ap.parse_args()
    if args.mode in ("e1", "e2") and not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    {"e1": run_e1, "e2": run_e2, "e2-analysis": run_e2_analysis, "backfill-manifests": run_backfill}[args.mode](args)


if __name__ == "__main__":
    main()
