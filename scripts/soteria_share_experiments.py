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
    container_dir.mkdir(parents=True, exist_ok=True)

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


def run_e2(args) -> None:
    from scripts.e1_ddim_runner import install_headless_visdom
    install_headless_visdom()   # t2bb_inner_linkage pulls in legacy modules that would dial a visdom server
    from scripts.t2bb_inner_linkage import evaluate, similarity
    from src.share.container import parse

    rows = list(csv.DictReader(args.split.open(encoding="utf-8")))
    if args.evidence_role == "smoke":
        rows = rows[: (args.limit or 4)]
    elif args.limit is not None:
        raise SystemExit("formal E2 uses the full split; --limit is smoke-only")
    if args.evidence_role == "formal" and len(rows) != 200:
        raise SystemExit(f"formal requires the full 200-image security_v1 (got {len(rows)})")
    tag = "" if args.evidence_role == "formal" else f"_{args.evidence_role}_n{len(rows)}"
    out_csv = args.out_dir / f"table_share_cross_recipient_linkage{tag}.csv"
    out_json = out_csv.with_suffix(".json")
    refuse_overwrite([out_csv, out_json], args.force)

    b_dir = args.artifact_dir / f"e1{tag}" / "containers_B"
    b_blobs = []
    for r in rows:
        path = b_dir / f"{r['sample_id']}.soteria"
        if not path.exists():
            raise SystemExit(f"E1 B container missing for reuse: {path} (run E1 {args.evidence_role} first)")
        blob = path.read_bytes()
        h = parse(blob).header
        if h.profile_id != PROFILE or h.manifest_digest != MANIFEST.digest() or h.inst_id_b != "hospital-B":
            raise SystemExit(f"E1 container {path} does not match the E2 profile/provenance")
        b_blobs.append(blob)

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
    c_dir.mkdir(parents=True, exist_ok=True)
    c_blobs, x0s, t_start = [], [], time.time()
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
                el = time.time() - t_start
                print(f"  C {i}/{len(rows)}  elapsed {el:.0f}s  ~{el / i:.2f}s/img", flush=True)

    pid = [r["patient_id"] for r in rows] * 2
    roles = ["gallery"] * len(rows) + ["query"] * len(rows)
    conditions = {
        "v2_public_png": _attacker_arrays(b_blobs) + _attacker_arrays(c_blobs),
        "positive_control_plaintext_x0": x0s + x0s,
    }
    out_rows = []
    for condition, arrays in conditions.items():
        for scorer in E2_SCORERS:
            sim = similarity(arrays, scorer, "cuda")
            res = evaluate(sim, pid, roles, np.random.default_rng(SEED))
            out_rows.append({"condition": condition, "scorer": scorer, "n_patients": len(rows),
                             "n_positive_pairs": len(rows), "n_negative_pairs": len(rows) * (len(rows) - 1), **res})
            print(f"  {condition:32s} {scorer:12s} AUC {res['auc']:.4f} [{res['auc_ci_low']}, {res['auc_ci_high']}] "
                  f"top1 {res['top1']:.3f} (random {res['random_top1']:.3f}, shuffled {res['neg_control_top1']:.3f})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(out_rows)
    from torchvision import models
    weights = models.DenseNet121_Weights.DEFAULT
    payload = {
        "schema_version": 1, "experiment": "E2", "evidence_role": args.evidence_role, "n_patients": len(rows),
        "plan": "reports/cross_institution_sharing.md §8.8 E2 (F3=(a))",
        "design": {"gallery": "B versions (reused from E1 containers)", "query": "C versions (same x0, fresh image key)",
                   "positives": "same x0 B/C", "negatives": "different-patient B/C",
                   "random_top1": 1 / len(rows), "random_top5": 5 / len(rows),
                   "attacker_view": "public PNG only, decoded as q/255 (decode_private_png_attacker_view); "
                                    "no sample/patient ids, paths, private metadata or keys",
                   "routing": "all gallery items carry B routing, all queries C routing (same for positives and negatives)",
                   "scorers": list(E2_SCORERS), "score_direction": "higher similarity = same image",
                   "ties": "numpy argsort order; continuous scores make exact ties measure-zero",
                   "densenet121_weights": str(weights.url),
                   "ci": "conditional-on-fixed-gallery: bootstrap of query-patient blocks, 2000x, seed 1911 "
                         "(t2bb_inner_linkage._boot_auc_ci); does not cover gallery-population uncertainty",
                   "controls": "positive = plaintext x0 copies; negative = shuffled query->gallery pairing (neg_control_top1)"},
        "reading": "AUC/Top-k are scientific results, not an engineering pass gate; a CI containing 0.5 only means no "
                   "difference was detected with these scorers -- it is not a proof of unlinkability.",
        "b_containers_sha256": hashlib.sha256(b"".join(hashlib.sha256(x).digest() for x in b_blobs)).hexdigest(),
        "c_containers_sha256": hashlib.sha256(b"".join(hashlib.sha256(x).digest() for x in c_blobs)).hexdigest(),
        "runtime_binding": binding, "keys": "PUBLIC test fixtures (S10/E1/<sid>/B, S10/E2/<sid>/C)",
        "split": str(args.split), "split_sha256": sha256_file(args.split),
        "wall_seconds_c_generation": time.time() - t_start, "git": git_head(), "gpu": torch.cuda.get_device_name(0),
        "rows": out_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}, {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--mode", choices=["e1", "e2"], required=True)
    ap.add_argument("--evidence-role", choices=["smoke", "formal"], default="smoke")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", type=Path, default=Path("splits/security_v1.csv"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/tables"))
    ap.add_argument("--artifact-dir", type=Path, default=Path("artifacts/runs/share_s10"))
    add_force_flag(ap)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if args.mode == "e1":
        run_e1(args)
    else:
        run_e2(args)


if __name__ == "__main__":
    main()
