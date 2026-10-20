#!/usr/bin/env python3
"""T2-BB inner-risk, learned (shadow) attack.

Upper-bound test for the predecessor's inner-risk leak: can a metric-learning net
trained on the attacker's own shadow patients beat the FIXED abs scorer (AUC 0.66,
table_t2bb_inner_linkage) at linking same-patient anonymous images on the disjoint
victim set? (reports/t2bb_inner_linkage_protocol.md 10.)

Channels (attacker-visible, key/model-free):
  s0_abs    |z| (range-preserving 8-bit). For the predecessor S0, |z_ano| = |z|
            regardless of the secret key, so this channel transfers shadow->victim.
  s2_anon   range-preserving view of transform(fixed_norm(z), per-image nonce).
            S2's per-image nonce makes shadow and victim independent, so no net can
            transfer -- the by-construction control (expected ~random).

Train on shadow patients (held-out val slice for a learning sanity check), evaluate
retrieval on the victim split; latents come from the shared cache (t2bb_invert_cache).
Reports learned Top-1/Top-5/AUC vs the fixed-abs baseline on the same victim features.

  scripts/run_cfg_ddim.sh python -m scripts.t2bb_learned_attack \
    --shadow-split splits/t2bb_shadow_v1.csv --max-patients 200 \
    --victim-split splits/t2bb_inner_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.d7_cpu_security import FIXTURE_KEY, nonce_for  # noqa: E402
from scripts.m1_storage import decode_range_preserving_png, encode_range_preserving_png  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from scripts.t2bb_inner_linkage import evaluate  # reuse Top-1/Top-5/AUC/neg-control  # noqa: E402
from src.crypto.s2_fixed_norm import normalize_to_fixed_norm  # noqa: E402
from src.crypto.sot import transform  # noqa: E402

SEED = 1911
IMG = 256
NET_IN = 128


def range_preserving_8bit(a: np.ndarray) -> np.ndarray:
    payload, _ = encode_range_preserving_png(a.astype(np.float32))
    return decode_range_preserving_png(payload).astype(np.float32)


def channel_feature(z: np.ndarray, channel: str, sample_id: str) -> np.ndarray:
    if channel == "s0_abs":
        return range_preserving_8bit(np.abs(z))
    if channel == "s2_anon":
        zt = torch.from_numpy(z).reshape(1, 1, IMG, IMG)
        u, _ = normalize_to_fixed_norm(zt)
        v = transform(u, FIXTURE_KEY, nonce_for(f"t2bb-inner::{sample_id}"), rounds=1, use_permutation=True)
        return range_preserving_8bit(v.reshape(IMG, IMG).numpy())
    raise ValueError(channel)


def load_features(split_csv: Path, cache_dir: Path, channel: str, max_patients: int | None = None):
    with split_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if max_patients is not None:
        rows = rows[: max_patients * 2]  # splits are pair-ordered (2 rows/patient)
    feats, pid, roles = [], [], []
    for r in rows:
        z = np.load(cache_dir / "latents" / f"{r['file_sha256']}.npy")
        feats.append(channel_feature(z, channel, r["sample_id"]))
        pid.append(r["pair_id"]); roles.append(r["pair_role"])
    return feats, pid, roles


def to_net_input(arrays: list[np.ndarray], device: str) -> torch.Tensor:
    t = torch.stack([torch.from_numpy(a).float() for a in arrays]).unsqueeze(1)  # (N,1,256,256)
    t = F.interpolate(t, size=(NET_IN, NET_IN), mode="bilinear", align_corners=False)
    m = t.mean(dim=(2, 3), keepdim=True); s = t.std(dim=(2, 3), keepdim=True) + 1e-6
    return ((t - m) / s).to(device)


class MetricNet(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.body = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, dim)

    def forward(self, x):
        return F.normalize(self.head(self.body(x).flatten(1)), dim=1)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    sim = emb @ emb.t() / temp
    n = emb.shape[0]
    mask_self = torch.eye(n, device=emb.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~mask_self
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp(min=1)
    return -((logprob * pos).sum(1) / pos_cnt).mean()


def gallery_query_sim(net, arrays, roles, device):
    net.eval()
    with torch.no_grad():
        emb = net(to_net_input(arrays, device)).cpu().numpy()
    return emb @ emb.T


def train(net, feats, pid, device, epochs, batch_patients, rng):
    # group image indices by patient (2 per patient)
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(pid):
        groups.setdefault(p, []).append(i)
    patients = sorted(groups)
    x = to_net_input(feats, device)
    lab = np.array([patients.index(p) for p in pid])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(patients))
        for b in range(0, len(patients), batch_patients):
            pick = order[b: b + batch_patients]
            idx = [i for pp in pick for i in groups[patients[pp]]]
            if len(idx) < 4:
                continue
            emb = net(x[idx])
            loss = supcon_loss(emb, torch.tensor(lab[idx], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--shadow-split", type=Path, default=ROOT / "splits/t2bb_shadow_v1.csv")
    ap.add_argument("--victim-split", type=Path, default=ROOT / "splits/t2bb_inner_v1.csv")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/t2bb_latent_cache")
    ap.add_argument("--max-patients", type=int, default=200, help="shadow prefix to train on")
    ap.add_argument("--channels", nargs="+", default=["s0_abs", "s2_anon"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--val-patients", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "paper_assets/tables/table_t2bb_learned_attack.csv")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_force_flag(ap)
    args = ap.parse_args()
    json_out = args.json_out or args.out.with_suffix(".json")
    refuse_overwrite([args.out, json_out], args.force)

    torch.manual_seed(SEED)
    device = args.device
    results = []
    for channel in args.channels:
        s_feats, s_pid, s_roles = load_features(args.shadow_split, args.cache_dir, channel, args.max_patients)
        v_feats, v_pid, v_roles = load_features(args.victim_split, args.cache_dir, channel)

        # hold out last val_patients shadow patients for a learning sanity check
        uniq = sorted(set(s_pid), key=s_pid.index)
        val_set = set(uniq[-args.val_patients:])
        tr = [i for i, p in enumerate(s_pid) if p not in val_set]
        va = [i for i, p in enumerate(s_pid) if p in val_set]

        net = MetricNet().to(device)
        final_loss = train(net, [s_feats[i] for i in tr], [s_pid[i] for i in tr],
                           device, args.epochs, batch_patients=32, rng=np.random.default_rng(SEED))

        def ev(feats, pid, roles):
            return evaluate(gallery_query_sim(net, feats, roles, device), pid, roles, np.random.default_rng(SEED))

        val_res = ev([s_feats[i] for i in va], [s_pid[i] for i in va], [s_roles[i] for i in va])
        vic_res = ev(v_feats, v_pid, v_roles)
        # fixed-abs baseline on the SAME victim features (cosine of flattened feature)
        flat = np.asarray([a.reshape(-1) for a in v_feats], dtype=np.float64)
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
        fixed = evaluate(flat @ flat.T, v_pid, v_roles, np.random.default_rng(SEED))

        for setname, res in [("shadow_val_learned", val_res), ("victim_learned", vic_res),
                             ("victim_fixed_cos", fixed)]:
            results.append({"channel": channel, "set": setname, "train_patients": args.max_patients,
                            "final_loss": round(final_loss, 4), **{k: round(v, 4) for k, v in res.items()}})
            print(f"{channel:8s} {setname:20s} Top-1={res['top1']:.3f} Top-5={res['top5']:.3f} "
                  f"AUC={res['auc']:.3f} (random Top-1={res['random_top1']:.3f})")

    cols = ["channel", "set", "train_patients", "final_loss", "top1", "top5", "auc",
            "neg_control_top1", "random_top1"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, lineterminator="\n"); w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})
    # efficacy verdict for the predecessor channel
    vic = next((r for r in results if r["channel"] == "s0_abs" and r["set"] == "victim_learned"), None)
    val = next((r for r in results if r["channel"] == "s0_abs" and r["set"] == "shadow_val_learned"), None)
    fx = next((r for r in results if r["channel"] == "s0_abs" and r["set"] == "victim_fixed_cos"), None)
    verdict = {
        "learned_victim_auc": vic["auc"] if vic else None,
        "fixed_victim_auc": fx["auc"] if fx else None,
        "shadow_val_learned_top1": val["top1"] if val else None,
        "net_learned": bool(val and val["top1"] >= 2 * val["random_top1"]),
        "beats_fixed": bool(vic and fx and vic["auc"] >= 0.70 and vic["top1"] >= 0.25),
    }
    verdict["efficacy"] = bool(verdict["net_learned"] and verdict["beats_fixed"])
    summary = {"schema_version": 1, "task_id": "T2BB-learned", "seed": SEED,
               "train_patients": args.max_patients, "epochs": args.epochs,
               "channels": args.channels, "verdict": verdict, "results": results,
               "note": ("Efficacy = net_learned (shadow-val Top-1 >= 2x random) AND beats_fixed "
                        "(victim AUC>=0.70 & Top-1>=0.25 vs fixed-abs 0.66). s2_anon is the "
                        "by-construction control (per-image nonce -> expected ~random).")}
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("VERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
