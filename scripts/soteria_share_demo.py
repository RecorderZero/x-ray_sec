#!/usr/bin/env python3
"""S9 CPU demo of the receiver-bound sharing protocol (§8.9 S9, V10).

Runs the A -> B flow and the rejection scenarios with IDENTITY stub pipelines
(no diffusion model), and reports protocol-layer CPU costs and container sizes.

Caveats written into the output:
* stub pipelines mean the PNG holds the quantized protected latent itself, so PNG
  size is NOT the size of a real anonymous image (real sizes/costs come from E1);
* A/B diffusion GPU cost is excluded -- only HPKE, DPHD, container and checks;
* all keys are PUBLIC test fixtures (TestFixtureStream / fixture_x25519).

  scripts/run_cfg_ddim.sh python -m scripts.soteria_share_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402
from src.share.container import assemble, compute_tag, parse, sign  # noqa: E402
from src.share.encoding import PROFILE_M1_PRIVATE_RANGE_V2 as PROFILE  # noqa: E402
from src.share.keys import (  # noqa: E402
    TestFixtureStream, build_info, derive_subkeys, fixture_x25519, unwrap_image_key, wipe_all, wrap_image_key,
)
from src.share.manifest import sign_manifest  # noqa: E402
from src.share.receiver import recover_image  # noqa: E402
from src.share.registry import KeyRecord, TrustStore, build_snapshot  # noqa: E402
from src.share.sender import share_image  # noqa: E402
from src.share.transform import decrypt_latent, encrypt_latent  # noqa: E402
from tests.security._share_world import MANIFEST, NOW, FactorySpy, World, ed25519, identity, x0_fixture  # noqa: E402

SEED = 1911


def _ms(fn, repeat: int) -> list[float]:
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _outcome(fn) -> tuple[str, str]:
    try:
        fn()
        return "accepted", "-"
    except Exception as exc:  # noqa: BLE001 - we report which layer fired
        return "rejected", type(exc).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--n", type=int, default=20, help="fixture images for size/timing")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_assets/tables")
    add_force_flag(parser)
    args = parser.parse_args()
    csv_path, json_path = args.out_dir / "table_share_protocol.csv", args.out_dir / "table_share_protocol.json"
    refuse_overwrite([csv_path, json_path], args.force)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tmp = Path(tempfile.mkdtemp(prefix="soteria_share_demo_"))
    world = World(tmp)

    def share(to=world.b, label=None, x0=None, registry=None):
        return share_image(x0 if x0 is not None else x0_fixture(), sender=world.sender, recipient_inst_id=to.inst_id,
                           recipient_key_id=to.key_id, registry=registry or world.registry,
                           manifest_digest=MANIFEST.digest(), invert=identity, generate=identity,
                           fixture=TestFixtureStream(label) if label else None)

    def recover(blob, who=world.b, fetch=None):
        return recover_image(blob, recipient=who, registry=world.registry,
                             fetch_manifest=fetch or world.fetch_manifest, pipeline_factory=FactorySpy())

    blob = share(label="demo/base")
    p = parse(blob)

    def readdressed():
        h = replace(p.header, inst_id_b=world.c.inst_id, key_id_b=world.c.key_id).encode()
        return assemble(h, p.meta_ct, p.png, p.tag, sign(world.a_sign, h, p.meta_ct, p.png, p.tag))

    def png_tampered():
        q = bytearray(p.png)
        flip = len(q) // 2
        q[flip] ^= 0x01
        return assemble(p.h_bytes, p.meta_ct, bytes(q), p.tag, sign(world.a_sign, p.h_bytes, p.meta_ct, bytes(q), p.tag))

    def meta_tampered():
        stream = TestFixtureStream("demo/base")
        sub = derive_subkeys(stream.image_key(), stream.nonce(), PROFILE)
        meta = p.meta_ct[:3] + bytes([p.meta_ct[3] ^ 0x80]) + p.meta_ct[4:]
        tag = compute_tag(sub["mac"], p.h_bytes, meta, p.png)
        return assemble(p.h_bytes, meta, p.png, tag, sign(world.a_sign, p.h_bytes, meta, p.png, tag))

    def unknown_signer():
        h = replace(p.header, signing_key_id_a="A-ed25519-unknown").encode()
        return assemble(h, p.meta_ct, p.png, p.tag, sign(world.a_sign, h, p.meta_ct, p.png, p.tag))

    def rollback():
        store = TrustStore(world.root.public_key(), tmp / "rollback_state.json")
        rec = lambda status: [  # noqa: E731
            KeyRecord("hospital-B", "B-x25519-2026a", "x25519-recv", status, fixture_x25519("B").public_key().public_bytes_raw())]
        store.load(build_snapshot(world.root, 7, NOW - 10, NOW + 3600, rec("revoked")), now=NOW)
        store.load(build_snapshot(world.root, 6, NOW - 10, NOW + 3600, rec("active")), now=NOW)

    revoked_world = World(tmp / "revoked", b_status="revoked")
    bad_manifest = lambda d: sign_manifest(replace(MANIFEST, steps=50), world.a_sign)  # noqa: E731

    scenarios = [
        ("A->B normal recovery", "B", lambda: recover(blob)),
        ("C intercepts B's container", "C", lambda: recover(blob, world.c)),
        ("re-addressed to C and re-signed (test signer)", "C", lambda: recover(readdressed(), world.c)),
        ("PNG byte changed, re-signed (test signer)", "B", lambda: recover(png_tampered())),
        ("meta_ct changed, tag recomputed + re-signed (fault injection)", "B", lambda: recover(meta_tampered())),
        ("signature forged by non-registered key", "B",
         lambda: recover(assemble(p.h_bytes, p.meta_ct, p.png, p.tag, sign(ed25519("mallory"), p.h_bytes, p.meta_ct, p.png, p.tag)))),
        ("unknown signing_key_id_A", "B", lambda: recover(unknown_signer())),
        ("off-profile manifest (steps=50)", "B", lambda: recover(blob, fetch=bad_manifest)),
        ("A sends to revoked B key", "A", lambda: share(registry=revoked_world.registry)),
        ("registry rollback v7(revoked) -> v6(active)", "A", rollback),
    ]
    rows = []
    for name, actor, fn in scenarios:
        outcome, layer = _outcome(fn)
        rows.append({"kind": "scenario", "item": name, "actor": actor, "outcome": outcome, "layer": layer})

    # ---- sizes and CPU timing over n fixture images
    sizes, wrap_ms, unwrap_ms, enc_ms, dec_ms, send_ms, recv_ms = [], [], [], [], [], [], []
    for i in range(args.n):
        label = f"demo/{i}"
        x0 = x0_fixture(SEED + i)
        t0 = time.perf_counter()
        b = share(label=label, x0=x0)
        send_ms.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        recover(b)
        recv_ms.append((time.perf_counter() - t0) * 1e3)
        pp = parse(b)
        sizes.append((len(b), len(pp.png), len(pp.h_bytes), len(b) - len(pp.png)))
        stream = TestFixtureStream(label)
        key, nonce = stream.image_key(), stream.nonce()
        info = build_info(PROFILE, nonce, world.b.inst_id, world.b.key_id, world.sender.inst_id, world.sender.signing_key_id)
        pk, sk = world.b.private_key.public_key(), world.b.private_key
        wrap_ms += _ms(lambda: wrap_image_key(key, pk, info), 5)
        w = wrap_image_key(key, pk, info)
        unwrap_ms += _ms(lambda: unwrap_image_key(w, sk, info).wipe(), 5)
        sub = derive_subkeys(key, nonce, PROFILE)
        holder = {}
        enc_ms += _ms(lambda: holder.__setitem__("c", encrypt_latent(x0, sub)), 1)
        dec_ms += _ms(lambda: decrypt_latent(holder["c"][0], sub, holder["c"][1]), 1)
        wipe_all(key, sub)

    def stat(name, values, unit):
        arr = np.asarray(values, dtype=np.float64)
        rows.append({"kind": "metric", "item": name, "unit": unit, "n": len(arr), "mean": float(arr.mean()),
                     "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())})

    s = np.asarray(sizes)
    stat("container_bytes (stub PNG)", s[:, 0], "B")
    stat("png_bytes (stub: quantized protected latent, not a real anon image)", s[:, 1], "B")
    stat("header_bytes", s[:, 2], "B")
    stat("non_png_overhead_bytes", s[:, 3], "B")
    stat("hpke_wrap_ms", wrap_ms, "ms")
    stat("hpke_unwrap_ms", unwrap_ms, "ms")
    stat("dphd_encrypt_incl_stream_gen_ms", enc_ms, "ms")
    stat("dphd_decrypt_incl_stream_gen_ms", dec_ms, "ms")
    stat("sender_total_stub_pipeline_ms", send_ms, "ms")
    stat("receiver_total_stub_pipeline_ms", recv_ms, "ms")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "item", "actor", "outcome", "layer", "unit", "n", "mean", "median", "min", "max"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "purpose": "S9 CPU demo of receiver-bound sharing (reports/cross_institution_sharing.md §8)",
        "pipelines": "identity stubs (no diffusion model); GPU diffusion cost excluded",
        "keys": "PUBLIC test fixtures only",
        "seed": SEED, "n_images": args.n, "profile_id": PROFILE,
        "host": {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.machine(),
                 "processor": platform.processor()},
        "caveats": [
            "PNG size here is the quantized protected latent, not a real anonymous image; real sizes come from E1.",
            "Timings are single-process CPU wall-clock on this host; not a security result.",
            "Scenario rejections are functional checks under these fixtures, not general security proofs.",
        ],
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        if r["kind"] == "scenario":
            print(f"{r['outcome']:9s} {r['layer']:16s} {r['item']}")
        else:
            print(f"{r['item']:72s} mean {r['mean']:10.2f} {r['unit']}  median {r['median']:10.2f}")


if __name__ == "__main__":
    main()
