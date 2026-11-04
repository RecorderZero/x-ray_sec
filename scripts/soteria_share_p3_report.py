#!/usr/bin/env python3
"""P3 evidence report for the sharing prototype (AF-054).

For each fixture (identity stub pipelines, PUBLIC fixture keys, CPU):
* metadata agreement: service norm/lo/hi (opened with an independent key) vs the
  reference's own values computed from x0;
* encode half: q differences vs the reference (count, max);
* decode half: service output vs the reference decode of the service's q with the
  reference's own metadata (pre-registered gate max-abs < 1e-5);
* lossy error vs x0 (MaxAbs, RMSE, relative L2) -- REPORTED ONLY, no quality gate.

The encode-half criterion (max|dq| <= 1, <= 1e-3 of pixels) was written AFTER seeing
THRESHOLD_SETTING_SEEDS; it is evaluated here separately on HELD_OUT_SEEDS that were
not used to choose it. Sender-metadata mutations (norm x1.01, lo/hi shifts) are run
to show the oracle now detects them.

  scripts/run_cfg_ddim.sh python -m scripts.soteria_share_p3_report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.share.sender as sender_mod  # noqa: E402
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from src.share.keys import TestFixtureStream  # noqa: E402
from src.share.receiver import recover_image  # noqa: E402
from src.share.sender import share_image  # noqa: E402
from tests.security import _share_reference as ref  # noqa: E402
from tests.security._share_world import MANIFEST, FactorySpy, World, identity, x0_fixture  # noqa: E402

SEED = 1911
DECODE_GATE = 1e-5
DQ_MAX, DQ_FRACTION = 1, 1e-3


def evaluate(world: World, seed: int, label: str) -> dict:
    x0 = x0_fixture(seed)
    blob = share_image(x0, sender=world.sender, recipient_inst_id=world.b.inst_id, recipient_key_id=world.b.key_id,
                       registry=world.registry, manifest_digest=MANIFEST.digest(), invert=identity, generate=identity,
                       fixture=TestFixtureStream(label))
    expected = ref.encode(x0.numpy(), label)
    norm_s, lo_s, hi_s = ref.service_metadata(blob, label)
    q_s = ref.service_q(blob)
    dq = np.abs(q_s.astype(np.int16) - expected["q"].astype(np.int16))
    out = recover_image(blob, recipient=world.b, registry=world.registry, fetch_manifest=world.fetch_manifest,
                        pipeline_factory=FactorySpy()).numpy().reshape(256, 256)
    oracle = ref.decode(q_s, expected["norm"], expected["lo"], expected["hi"], label)
    x = x0.numpy().reshape(256, 256).astype(np.float64)
    diff = out.astype(np.float64) - x
    row = {
        "seed": seed, "label": label,
        "norm_rel_diff": abs(norm_s - expected["norm"]) / expected["norm"],
        "lo_abs_diff": abs(float(lo_s) - float(expected["lo"])),
        "hi_abs_diff": abs(float(hi_s) - float(expected["hi"])),
        "dq_pixels": int((dq > 0).sum()), "dq_max": int(dq.max()), "dq_fraction": float((dq > 0).mean()),
        "decode_half_max_abs": float(np.abs(out - oracle).max()),
        "lossy_max_abs": float(np.abs(diff).max()),
        "lossy_rmse": float(np.sqrt((diff ** 2).mean())),
        "lossy_rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(x)),
    }
    row["metadata_ok"] = row["norm_rel_diff"] <= 1e-9 and row["lo_abs_diff"] <= 1e-5 and row["hi_abs_diff"] <= 1e-5
    row["encode_ok"] = row["dq_max"] <= DQ_MAX and row["dq_fraction"] <= DQ_FRACTION
    row["decode_ok"] = row["decode_half_max_abs"] < DECODE_GATE
    return row


def mutation_rows(world: World) -> list[dict]:
    real = sender_mod.seal_metadata
    out = []
    for name in ("norm_x1.01", "lo_shift", "hi_shift"):
        def faulty(key, h, norm, lo, hi, name=name):
            span = float(hi) - float(lo)
            if name == "norm_x1.01":
                norm = norm * 1.01
            elif name == "lo_shift":
                lo = float(lo) - 0.01 * span
            else:
                hi = float(hi) + 0.01 * span
            return real(key, h, norm, lo, hi)
        with patch.object(sender_mod, "seal_metadata", faulty):
            row = evaluate(world, SEED, f"P3/mutation/{name}")
        out.append({"mutation": name, "detected": not (row["metadata_ok"] and row["encode_ok"] and row["decode_ok"]),
                    "metadata_ok": row["metadata_ok"], "decode_ok": row["decode_ok"],
                    "lossy_max_abs_under_mutation": row["lossy_max_abs"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_assets/tables")
    add_force_flag(parser)
    args = parser.parse_args()
    csv_path, json_path = args.out_dir / "table_share_p3.csv", args.out_dir / "table_share_p3.json"
    refuse_overwrite([csv_path, json_path], args.force)
    torch.manual_seed(SEED)
    world = World(Path(tempfile.mkdtemp(prefix="soteria_p3_")))

    rows = []
    for group, seeds in (("threshold_setting", ref.THRESHOLD_SETTING_SEEDS), ("held_out", ref.HELD_OUT_SEEDS)):
        for seed in seeds:
            rows.append({"group": group, **evaluate(world, seed, f"P3/report/{seed}")})
    mutations = mutation_rows(world)

    def summary(group: str) -> dict:
        g = [r for r in rows if r["group"] == group]
        lossy = np.asarray([r["lossy_max_abs"] for r in g])
        rmse = np.asarray([r["lossy_rmse"] for r in g])
        return {"n": len(g), "metadata_ok_all": all(r["metadata_ok"] for r in g),
                "encode_ok_all": all(r["encode_ok"] for r in g), "decode_ok_all": all(r["decode_ok"] for r in g),
                "dq_pixels_max": max(r["dq_pixels"] for r in g), "dq_pixels_total": sum(r["dq_pixels"] for r in g),
                "decode_half_max_abs_worst": max(r["decode_half_max_abs"] for r in g),
                "lossy_max_abs_mean": float(lossy.mean()), "lossy_max_abs_worst": float(lossy.max()),
                "lossy_rmse_mean": float(rmse.mean())}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "purpose": "P3 evidence for receiver-bound sharing (AF-054)",
        "pipelines": "identity stubs; lossy error is the 8-bit quantization of the protected latent, not E1 image quality",
        "keys": "PUBLIC test fixtures only",
        "criteria": {
            "decode_half": f"max-abs < {DECODE_GATE} (pre-registered P3 gate)",
            "encode_half": f"max|dq| <= {DQ_MAX} and fraction <= {DQ_FRACTION} (POST-HOC, chosen after threshold_setting seeds; "
                           "needs user confirmation before being cited as a formal acceptance criterion)",
            "metadata": "norm rel diff <= 1e-9, lo/hi abs diff <= 1e-5 vs reference computed from x0",
            "lossy": "reported only, no gate",
        },
        "threshold_setting_seeds": list(ref.THRESHOLD_SETTING_SEEDS),
        "held_out_seeds": [ref.HELD_OUT_SEEDS[0], ref.HELD_OUT_SEEDS[-1]],
        "summary": {"threshold_setting": summary("threshold_setting"), "held_out": summary("held_out")},
        "mutations": mutations,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "mutations": mutations}, indent=2))


if __name__ == "__main__":
    main()
