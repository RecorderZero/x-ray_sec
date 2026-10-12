#!/usr/bin/env python3
"""A4.5 tamper baseline: the predecessor's S0 / S1 have no integrity check.

A stored record (anonymized latent, or a stored image record) can be modified and
the predecessor's inverse-key step still runs, returns a finite result and raises
nothing: there is no MAC, tag or verification step to reject it.  This script
quantifies how far the silently accepted result is from the untampered one.

Tamper cases (z_ano = predecessor key applied to a real E2.3 latent, d = 65536):
  flip_one_sign        flip the sign of one coordinate of z_ano
  gaussian_noise       add N(0, 0.05^2) noise to z_ano
  storage_range_hi     z_ano[0,0] stored as range_preserving_png/v1; the stored
                       `hi` bit pattern in the PNG text chunk is scaled by 1.05
  storage_pixel        same record, one quantized pixel changed (q -> q+128 mod 256)

The storage cases are lossy (8 bit), so their clean reference is the recovery
from the UNTAMPERED stored record; latent-level cases are compared with the
original latent (the untampered round trip is bit-exact).  The contrast with
S2's transform-then-MAC is a note only (not implemented here).

CPU only, seed 1911, deterministic.

    scripts/run_cfg_ddim.sh python -m scripts.a4_5_tamper
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# CPU only: hide every GPU before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.env_guard import enforce_active_prefix  # noqa: E402
from scripts.e1_ddim_runner import git_value, sha256_file  # noqa: E402
from scripts.e2_legacy_wrapper import apply_legacy_key  # noqa: E402
from scripts.m1_storage import (  # noqa: E402
    HI_KEY, LO_KEY, _f32_bits, _f32_from_bits,
    decode_range_preserving_png, encode_range_preserving_png,
)
from scripts.output_guard import add_force_flag, refuse_overwrite  # noqa: E402

enforce_active_prefix({"numpy": np, "torch": torch})
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "A4.5 must run on CPU only"

SEED = 1911
TASK_ID = "A4.5"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.a4_5_tamper"
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/A4.5.json"
TABLE = ROOT / "paper_assets/tables/table_tamper_baseline.csv"
SOURCE_ROOT = ROOT / "past/SourceCode"
ANON_PATH = SOURCE_ROOT / "guided_diffusion/anonymization.py"
LATENT_CACHE = ROOT / "artifacts/runs/E2.3/E2.3_20261009T183736363217Z_84cf642a/latents.npy"

LATENT_INDICES = (0, 1, 2)
SCHEMES = ("S0", "S1")
NOISE_SIGMA = 0.05
HI_SCALE = 1.05
CASES = ("flip_one_sign", "gaussian_noise", "storage_range_hi", "storage_pixel")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def legacy(x: torch.Tensor, scheme: str, mode: str) -> torch.Tensor:
    return apply_legacy_key(x, SOURCE_ROOT, scheme, mode)


def metrics(recovered: torch.Tensor, reference: torch.Tensor) -> dict:
    a = recovered.double().reshape(-1)
    b = reference.double().reshape(-1)
    diff = a - b
    cos = float((a @ b) / (a.norm() * b.norm()))
    return {
        "cosine": cos,
        "rmse": float(torch.sqrt(torch.mean(diff * diff))),
        "max_abs": float(diff.abs().max()),
        "n_changed_coordinates": int((recovered.reshape(-1) != reference.reshape(-1)).sum()),
    }


def run_inverse(x: torch.Tensor, scheme: str) -> tuple[bool, torch.Tensor | None, str | None]:
    """Run the legacy inverse-key step; ``accepted`` means it ran and returned a finite tensor."""
    try:
        out = legacy(x, scheme, "deanonymize")
    except Exception as exc:  # noqa: BLE001 - any rejection must be visible
        return False, None, f"{type(exc).__name__}: {exc}"
    return bool(torch.isfinite(out).all()), out, None


def rewrite_png(payload: bytes, hi_scale: float | None = None, pixel: tuple[int, int] | None = None) -> bytes:
    """Return the PNG with the stored `hi` scaled and/or one quantized pixel changed (text chunks otherwise kept)."""
    with Image.open(io.BytesIO(payload)) as handle:
        text = dict(handle.text)
        q = np.array(handle, dtype=np.uint8)
    if hi_scale is not None:
        hi = _f32_from_bits(text[HI_KEY])
        text[HI_KEY] = _f32_bits(np.float32(hi) * np.float32(hi_scale))
    if pixel is not None:
        q[pixel] = np.uint8((int(q[pixel]) + 128) % 256)
    info = PngInfo()
    for key, value in text.items():
        info.add_text(key, value)
    buffer = io.BytesIO()
    Image.fromarray(q, mode="L").save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def strip_range_text(payload: bytes) -> bytes:
    with Image.open(io.BytesIO(payload)) as handle:
        q = np.array(handle, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(q, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def predecessor_integrity_scan() -> dict:
    """Static look at the predecessor's key module: is there any check on the data it inverts?"""
    source = ANON_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    inverse_names = {"deanonymize_latent", "apply_inverse", "apply_key", "anonymize_latent", "apply"}
    guards = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in inverse_names:
            key = node.name
            count = sum(isinstance(n, (ast.Raise, ast.Assert)) for n in ast.walk(node))
            guards[key] = guards.get(key, 0) + count
    lowered = source.lower()
    verify_functions = sorted(
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("verify_")
    )
    # verify_* take the original latent as an argument: they are developer self-tests of the key holder, not
    # input validation of a stored record.
    takes_original = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name.startswith("verify_"):
            takes_original[n.name] = [a.arg for a in n.args.args]
    return {
        "raise_or_assert_count_in_data_path_functions": guards,
        "contains_hmac_text": "hmac" in lowered,
        "contains_mac_or_tag_verification_text": any(w in lowered for w in ("authenticat", "checksum", "integrity")),
        "verify_functions_are_developer_self_tests": verify_functions,
        "verify_function_arguments": takes_original,
        "hashlib_used_only_for_key_derivation": "hashlib.sha256(password" in source,
    }


def tamper_rows(cache, scheme: str, idx: int, rng: np.random.Generator) -> tuple[list[dict], dict]:
    z = torch.from_numpy(np.array(cache[idx], dtype=np.float32))[None]  # (1,1,256,256)
    z_ano = legacy(z, scheme, "anonymize")
    ok_clean, clean_rec, _ = run_inverse(z_ano, scheme)
    clean_round_trip_bit_exact = bool(ok_clean and torch.equal(clean_rec, z))
    rows: list[dict] = []

    def add(case: str, tampered: torch.Tensor, reference: torch.Tensor, ref_name: str, extra: dict | None = None):
        accepted, rec, error = run_inverse(tampered, scheme)
        row = {"scheme": scheme, "latent_index": idx, "tamper_case": case, "accepted": accepted,
               "error": error, "reference": ref_name}
        if accepted:
            row.update(metrics(rec, reference))
        if extra:
            row.update(extra)
        rows.append(row)

    # (a) one sign flip
    j = int(rng.integers(0, z_ano.numel()))
    flipped = z_ano.clone().reshape(-1)
    flipped[j] = -flipped[j]
    add("flip_one_sign", flipped.reshape(z_ano.shape), z, "original_latent",
        {"tampered_coordinates": 1})

    # (b) additive Gaussian noise
    noise = torch.from_numpy(rng.standard_normal(z_ano.numel()).astype(np.float32) * np.float32(NOISE_SIGMA))
    add("gaussian_noise", z_ano + noise.reshape(z_ano.shape), z, "original_latent",
        {"noise_sigma": NOISE_SIGMA})

    # (c) the stored range-preserving PNG record of z_ano[0,0]
    payload, record = encode_range_preserving_png(z_ano[0, 0].numpy())
    stored = torch.from_numpy(decode_range_preserving_png(payload))[None, None]
    ok_s, stored_rec, _ = run_inverse(stored, scheme)
    assert ok_s
    lossy_vs_original = metrics(stored_rec, z)
    hi_payload = rewrite_png(payload, hi_scale=HI_SCALE)
    px = (int(rng.integers(0, 256)), int(rng.integers(0, 256)))
    pixel_payload = rewrite_png(payload, pixel=px)
    for case, tampered_payload in (("storage_range_hi", hi_payload), ("storage_pixel", pixel_payload)):
        try:
            decoded = torch.from_numpy(decode_range_preserving_png(tampered_payload))[None, None]
            decode_error = None
        except Exception as exc:  # noqa: BLE001
            decoded, decode_error = None, f"{type(exc).__name__}: {exc}"
        if decoded is None:
            rows.append({"scheme": scheme, "latent_index": idx, "tamper_case": case, "accepted": False,
                         "error": decode_error, "reference": "untampered_stored_record"})
            continue
        add(case, decoded, stored_rec, "untampered_stored_record",
            {"record_png_sha256_matches_payload": hashlib.sha256(tampered_payload).hexdigest() == record["png_sha256"],
             "decoder_consults_record_sha256": False})
    info = {
        "scheme": scheme, "latent_index": idx,
        "untampered_latent_round_trip_bit_exact": clean_round_trip_bit_exact,
        "untampered_storage_round_trip_vs_original_latent": lossy_vs_original,
        "tampered_pixel_position": list(px),
    }
    return rows, info


def fmt(value: float) -> str:
    return f"{value:.9g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--table", type=Path, default=TABLE)
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([args.output, args.table], args.force)

    cache = np.load(LATENT_CACHE, mmap_mode="r")
    assert cache.shape == (200, 1, 256, 256) and cache.dtype == np.float32
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(8)[0])

    rows: list[dict] = []
    controls: list[dict] = []
    for scheme in SCHEMES:
        for idx in LATENT_INDICES:
            r, info = tamper_rows(cache, scheme, idx, rng)
            rows.extend(r)
            controls.append(info)

    # positive control for the harness itself: a malformed record (range text stripped) IS rejected by the
    # decoder (format check), so ``accepted=False`` is observable when something really raises.
    z0 = torch.from_numpy(np.array(cache[0], dtype=np.float32))[None]
    payload, _ = encode_range_preserving_png(legacy(z0, "S0", "anonymize")[0, 0].numpy())
    try:
        decode_range_preserving_png(strip_range_text(payload))
        harness_detects_rejection = False
    except ValueError:
        harness_detects_rejection = True

    scan = predecessor_integrity_scan()
    by_case = lambda c, s=None: [r for r in rows if r["tamper_case"] == c and (s is None or r["scheme"] == s)]  # noqa: E731
    flips = by_case("flip_one_sign")
    checks = {
        "all_tamper_cases_accepted": all(r["accepted"] for r in rows),
        "tamper_row_count": len(rows) == len(SCHEMES) * len(LATENT_INDICES) * len(CASES),
        "untampered_latent_round_trip_bit_exact": all(c["untampered_latent_round_trip_bit_exact"] for c in controls),
        "every_tamper_changes_the_recovered_latent": all(r.get("max_abs", 0.0) > 0.0 for r in rows),
        "flip_one_sign_changes_exactly_one_coordinate": all(r["n_changed_coordinates"] == 1 for r in flips),
        "all_metrics_finite": all(math.isfinite(r.get(k, float("nan"))) for r in rows for k in ("cosine", "rmse", "max_abs")),
        "harness_detects_rejection_when_decoder_raises": harness_detects_rejection,
        "predecessor_data_path_has_no_raise_or_assert": all(v == 0 for v in scan["raise_or_assert_count_in_data_path_functions"].values()),
        "predecessor_has_no_hmac_or_tag_verification": not scan["contains_hmac_text"] and not scan["contains_mac_or_tag_verification_text"],
        "decoder_ignores_record_sha256_on_modified_payload": all(
            r["record_png_sha256_matches_payload"] is False for r in rows if r["tamper_case"].startswith("storage_")),
    }
    if not all(checks.values()):
        raise SystemExit(f"A4.5 checks failed: {[k for k, v in checks.items() if not v]}")

    ranges = {}
    for case in CASES:
        sel = by_case(case)
        ranges[case] = {
            "cosine": [min(r["cosine"] for r in sel), max(r["cosine"] for r in sel)],
            "rmse": [min(r["rmse"] for r in sel), max(r["rmse"] for r in sel)],
            "max_abs": [min(r["max_abs"] for r in sel), max(r["max_abs"] for r in sel)],
        }
    conclusion = (
        f"[事實] 前作 S0/S1 沒有任何完整性檢查：{len(rows)} 個竄改案例（{len(SCHEMES)} 種方案 x {len(LATENT_INDICES)} 張真實 latent x {len(CASES)} 種竄改）"
        "全部被接受（accepted=True）——逆金鑰步驟照常執行、輸出皆為有限值、沒有任何例外或拒絕，而恢復結果與未竄改版本不同。"
        f"影響範圍：翻轉單一座標符號只改變恢復 latent 的 1 個座標（max_abs {ranges['flip_one_sign']['max_abs'][0]:.3g}–{ranges['flip_one_sign']['max_abs'][1]:.3g}，"
        f"cosine {ranges['flip_one_sign']['cosine'][0]:.9f}–{ranges['flip_one_sign']['cosine'][1]:.9f}）；"
        f"加入 σ={NOISE_SIGMA} 高斯雜訊使 RMSE 約 {ranges['gaussian_noise']['rmse'][0]:.3g}–{ranges['gaussian_noise']['rmse'][1]:.3g}；"
        f"改動儲存紀錄中的 hi 值（x{HI_SCALE}）使 RMSE {ranges['storage_range_hi']['rmse'][0]:.3g}–{ranges['storage_range_hi']['rmse'][1]:.3g}；"
        f"改動單一儲存像素使 max_abs {ranges['storage_pixel']['max_abs'][0]:.3g}–{ranges['storage_pixel']['max_abs'][1]:.3g}。"
        "未竄改的 latent 層往返為 bit-exact（對照），且解碼器對格式錯誤的紀錄（缺少 lo/hi 文字）仍會拋例外，表示 accepted=False 在機制上可被觀察，"
        "而內容被竄改時卻沒有任何拒絕。靜態檢查：前作金鑰模組的資料路徑函式沒有 raise/assert，沒有 HMAC／認證字樣；其 verify_* 是持有原始 latent 的開發者自我測試，不是對儲存紀錄的驗證。"
        "[推定] S2 的 transform-then-MAC 會在驗證階段拒絕這些竄改（D5 實作、AF-024/AF-030），本表是無完整性保護的 baseline，S2 的拒絕行為需另以 D5 的測試驗證，本腳本不實作 HMAC。"
        "[假設] 這裡的 accepted 指金鑰步驟與儲存解碼器層級（無 GPU）；完整管線（含擴散生成）的下游表現不在本表範圍。"
    )

    created_at = datetime.now(timezone.utc).isoformat()
    payload_json = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "created_at": created_at,
        "command": COMMAND,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "git_commit": git_value("rev-parse", "HEAD"),
        "seed": SEED,
        "seed_streams": "np.random.default_rng(np.random.SeedSequence(1911).spawn(8)[0]); draw order: scheme S0 then S1, latent 0,1,2; per latent: flipped coordinate, noise, pixel row/col",
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cpu (CUDA_VISIBLE_DEVICES='' set before importing torch)",
        "inputs": {
            "predecessor_source": "past/SourceCode/guided_diffusion/anonymization.py",
            "predecessor_source_sha256": sha256_file(ANON_PATH),
            "real_latent_cache": str(LATENT_CACHE.relative_to(ROOT)),
            "real_latent_cache_sha256": sha256_file(LATENT_CACHE),
            "latent_indices": list(LATENT_INDICES),
            "key": "legacy S0/S1 full-mask key via scripts.e2_legacy_wrapper.apply_legacy_key (d=65536); key material not written",
            "storage_protocol": "scripts.m1_storage range_preserving_png/v1",
            "noise_sigma": NOISE_SIGMA, "hi_scale": HI_SCALE,
        },
        "raw_key_material_recorded": False,
        "acceptance_definition": "accepted=True iff the legacy inverse-key step (and the storage decoder for storage cases) ran without raising and returned a finite tensor",
        "predecessor_integrity_scan": scan,
        "impact_ranges_over_all_rows": ranges,
        "untampered_controls": controls,
        "rows": rows,
        "checks": checks,
        "conclusion": conclusion,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["scheme", "tamper_case", "accepted", "recovered_latent_cosine_vs_clean",
                         "recovered_latent_rmse_vs_clean", "recovered_latent_max_abs_vs_clean",
                         "latent_index", "clean_reference"])
        for r in rows:
            writer.writerow([r["scheme"], r["tamper_case"], r["accepted"], fmt(r["cosine"]), fmt(r["rmse"]),
                             fmt(r["max_abs"]), r["latent_index"], r["reference"]])
    print(json.dumps({"checks": checks, "impact_ranges": ranges, "conclusion": conclusion}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
