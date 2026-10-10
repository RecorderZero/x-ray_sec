#!/usr/bin/env python3
"""A3.1 effective key-space audit of the predecessor's S0 / S1 key objects.

Question: the predecessor claims key spaces of 2^d (Rademacher, S0) and
2^d * d! (Signed Permutation, S1) with d = 65536.  How many bits of key does
the code path actually use?

The script (1) anchors every file:line citation to the predecessor source
(it fails if a cited line no longer contains the cited text), (2) runs a set
of numerical checks on the *predecessor's own key classes*, and (3) writes
the claimed-vs-actual table and a method note.

It is an independent re-implementation of the arithmetic in the exploratory
attack/attack_keyspace.py (which it does not import).  CPU only; no
checkpoint is loaded; the GPU is hidden from torch before import.

    scripts/run_cfg_ddim.sh python -m scripts.a3_keyspace_audit

Outputs (deterministic for a fixed source tree, torch and numpy version):
    paper_assets/tables/table_keyspace.csv
    reports/keyspace_method_note.md
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
from pathlib import Path

# CPU only: hide every GPU before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from scripts.output_guard import add_force_flag, refuse_overwrite
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.env_guard import enforce_active_prefix  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    from env_guard import enforce_active_prefix  # type: ignore  # noqa: E402

enforce_active_prefix({"numpy": np, "torch": torch})
torch.set_num_threads(1)

SOURCE_ROOT = ROOT / "past/SourceCode"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from guided_diffusion.anonymization import (  # noqa: E402
    PermutationKey,
    RademacherKey,
    SignedPermutationKey,
)

SEED = 1911
TASK_ID = "A3.1"
COMMAND = "scripts/run_cfg_ddim.sh python -m scripts.a3_keyspace_audit"
ANON = "past/SourceCode/guided_diffusion/anonymization.py"
ANON_PATH = ROOT / ANON
CSV_OUT = ROOT / "paper_assets/tables/table_keyspace.csv"
NOTE_OUT = ROOT / "reports/keyspace_method_note.md"

SHAPE = (1, 256, 256)
D = 256 * 256  # 65536
LEGACY_PASSWORD = "Ki@13579"
LEGACY_SEED_ARG = 42
PERM_OFFSET = 1_000_000  # SignedPermutationKey seed mode: perm_seed = seed + 1000000

# (id, file, line, substring) -- every citation used below is verified at run time.
CRYPTO_SH = "past/SourceCode/run_cfg_inference_Crypto.sh"
SAMPLER = "past/SourceCode/scripts/cfg_image_sample_anonymization.py"
ANCHORS = [
    ("claim_module_docstring", ANON, 20, "Key space expands from 2^d to 2^d × d!"),
    ("claim_class_docstring", ANON, 305, "Key space expands from 2^d to 2^d × d!"),
    ("rad_password_branch", ANON, 70, "elif password is not None:"),
    ("rad_seed_branch", ANON, 73, "elif seed is not None:"),
    ("rad_manual_seed", ANON, 85, "generator.manual_seed(seed)"),
    ("rad_bernoulli", ANON, 87, "torch.bernoulli(torch.ones(shape) * 0.5, generator=generator)"),
    ("rad_sha256", ANON, 99, "hashlib.sha256(password_bytes).digest()"),
    ("rad_seed_slice", ANON, 100, "seed = int.from_bytes(hash_digest[:4], byteorder='big')"),
    ("perm_password_branch", ANON, 164, "elif password is not None:"),
    ("perm_seed_branch", ANON, 167, "elif seed is not None:"),
    ("perm_manual_seed", ANON, 179, "generator.manual_seed(seed)"),
    ("perm_randperm", ANON, 180, "torch.randperm(size, generator=generator)"),
    ("perm_suffix", ANON, 189, '(password + "_permutation")'),
    ("perm_sha256", ANON, 190, "hashlib.sha256(password_bytes).digest()"),
    ("perm_seed_slice", ANON, 191, "seed = int.from_bytes(hash_digest[:4], byteorder='big')"),
    ("sp_rademacher_ctor", ANON, 344, "self.rademacher_key = RademacherKey("),
    ("sp_perm_offset", ANON, 358, "perm_seed = seed + 1000000"),
    ("sp_perm_ctor", ANON, 359, "self.permutation_key = PermutationKey("),
    ("sampler_log_claim", SAMPLER, 505, "key space = 2^d × d! (astronomical)"),
    ("sampler_sp_ctor", SAMPLER, 506, "key = SignedPermutationKey("),
    ("e22_wrapper_ctor", "scripts/e2_legacy_wrapper.py", 73, "password=LEGACY_PASSWORD,"),
    ("shell_flags", CRYPTO_SH, 7, "--anonymization_key_seed 42 --anonymization_key_password Ki@13579 --use_signed_permutation True"),
]
PAPER_PDF = "past/Privacy-Preserving and Reversible Diffusion Models for Lesion Localization in Chest X-Rays.pdf"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_prefix_seed(text: str) -> tuple[int, str]:
    """Reproduce the predecessor's seed derivation (anonymization.py:99-100, 190-191)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big"), digest.hex()


def verify_anchors() -> list[dict]:
    cache: dict[str, list[str]] = {}
    out = []
    for aid, rel, line, needle in ANCHORS:
        lines = cache.setdefault(rel, (ROOT / rel).read_text(encoding="utf-8").splitlines())
        ok = line <= len(lines) and needle in lines[line - 1]
        if not ok:
            raise SystemExit(f"anchor {aid} failed: {rel}:{line} does not contain {needle!r}")
        out.append({"id": aid, "file": rel, "line": line, "text": lines[line - 1].strip()})
    return out


def frac_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a == b).float().mean())


def find_collision(suffix: str, limit: int = 1 << 22) -> dict:
    """First pair of distinct passwords 'collide-<i><suffix-free>' with equal sha256(pw+suffix)[:4]."""
    seen: dict[int, str] = {}
    for i in range(limit):
        pw = f"collide-{i}"
        seed, digest = sha_prefix_seed(pw + suffix)
        if seed in seen:
            other = seen[seed]
            _, other_digest = sha_prefix_seed(other + suffix)
            return {
                "password_a": other, "password_b": pw, "seed": seed, "hashes_computed": i + 1,
                "digest_a": other_digest, "digest_b": digest,
                "digests_differ": other_digest != digest,
                "same_first_4_bytes": other_digest[:8] == digest[:8],
                "differing_hex_chars": sum(x != y for x, y in zip(other_digest, digest)),
            }
        seen[seed] = pw
    raise SystemExit("no sha256[:4] collision found within the search limit")


# ---------------------------------------------------------------------------
# numerical checks on the predecessor's key classes
# ---------------------------------------------------------------------------
def run_checks() -> dict:
    checks: dict[str, dict] = {}

    # V1: which bits of the SHA-256 digest are used.
    sign_seed, sign_digest = sha_prefix_seed(LEGACY_PASSWORD)
    perm_seed, perm_digest = sha_prefix_seed(LEGACY_PASSWORD + "_permutation")
    checks["V1"] = {
        "sign_password": LEGACY_PASSWORD, "sign_digest": sign_digest, "sign_seed": sign_seed,
        "perm_password": LEGACY_PASSWORD + "_permutation", "perm_digest": perm_digest, "perm_seed": perm_seed,
        "bytes_used": 4, "bits_used": 32, "bits_unused": 224,
        "sign_seed_fraction_of_2_32": sign_seed / 2**32,
        "perm_seed_fraction_of_2_32": perm_seed / 2**32,
    }

    # V2: the full 65536-dim key is reproduced from the 32-bit integer alone.
    k_pw = RademacherKey(shape=SHAPE, password=LEGACY_PASSWORD).key
    k_seed = RademacherKey(shape=SHAPE, seed=sign_seed).key
    p_pw = PermutationKey(size=D, password=LEGACY_PASSWORD).permutation
    p_seed = PermutationKey(size=D, seed=perm_seed).permutation
    sp_pw = SignedPermutationKey(shape=SHAPE, password=LEGACY_PASSWORD)
    sp_ints = SignedPermutationKey(
        shape=SHAPE,
        rademacher_key=RademacherKey(shape=SHAPE, seed=sign_seed),
        permutation_key=PermutationKey(size=D, seed=perm_seed),
    )
    z = torch.from_numpy(np.random.default_rng(SEED).standard_normal(SHAPE).astype(np.float32))
    checks["V2"] = {
        "d": D,
        "rademacher_equal": bool(torch.equal(k_pw, k_seed)),
        "rademacher_coords_equal": int((k_pw == k_seed).sum()),
        "permutation_equal": bool(torch.equal(p_pw, p_seed)),
        "permutation_positions_equal": int((p_pw == p_seed).sum()),
        "sp_sign_equal": bool(torch.equal(sp_pw.rademacher_key.key, sp_ints.rademacher_key.key)),
        "sp_perm_equal": bool(torch.equal(sp_pw.permutation_key.permutation, sp_ints.permutation_key.permutation)),
        "sp_ciphertext_equal": bool(torch.equal(sp_pw.apply(z), sp_ints.apply(z))),
        "key_values_are_pm1": bool(((k_pw == 1) | (k_pw == -1)).all()),
        "perm_is_valid_permutation": bool(torch.equal(torch.sort(p_pw).values, torch.arange(D))),
    }

    # V3: password takes precedence over seed (the legacy shell passes both).
    k_both = RademacherKey(shape=SHAPE, seed=LEGACY_SEED_ARG, password=LEGACY_PASSWORD).key
    k_seed42 = RademacherKey(shape=SHAPE, seed=LEGACY_SEED_ARG).key
    sp_both = SignedPermutationKey(shape=SHAPE, seed=LEGACY_SEED_ARG, password=LEGACY_PASSWORD)
    sp_seed42 = SignedPermutationKey(shape=SHAPE, seed=LEGACY_SEED_ARG)
    checks["V3"] = {
        "s0_both_equals_password_only": bool(torch.equal(k_both, k_pw)),
        "s0_both_equals_seed_only": bool(torch.equal(k_both, k_seed42)),
        "s0_sign_agreement_with_seed42": frac_equal(k_both, k_seed42),
        "s1_both_sign_equals_password_only": bool(torch.equal(sp_both.rademacher_key.key, sp_pw.rademacher_key.key)),
        "s1_both_perm_equals_password_only": bool(torch.equal(sp_both.permutation_key.permutation, sp_pw.permutation_key.permutation)),
        "s1_both_sign_equals_seed_only": bool(torch.equal(sp_both.rademacher_key.key, sp_seed42.rademacher_key.key)),
        "s1_both_perm_equals_seed_only": bool(torch.equal(sp_both.permutation_key.permutation, sp_seed42.permutation_key.permutation)),
        "s1_perm_position_agreement_with_seed_only": frac_equal(sp_both.permutation_key.permutation, sp_seed42.permutation_key.permutation),
    }

    # V4: seed mode of SignedPermutationKey: sign and permutation are one 32-bit value.
    rows = []
    for s in (0, LEGACY_SEED_ARG, SEED, 123456789):
        sp = SignedPermutationKey(shape=SHAPE, seed=s)
        rows.append({
            "seed": s, "perm_seed": s + PERM_OFFSET,
            "sign_equals_rademacher_of_seed": bool(torch.equal(sp.rademacher_key.key, RademacherKey(shape=SHAPE, seed=s).key)),
            "perm_equals_permutation_of_seed_plus_offset": bool(
                torch.equal(sp.permutation_key.permutation, PermutationKey(size=D, seed=s + PERM_OFFSET).permutation)
            ),
        })
    wrap_seed = 2**32 - PERM_OFFSET
    sp_wrap = SignedPermutationKey(shape=SHAPE, seed=wrap_seed)
    checks["V4"] = {
        "rows": rows,
        "all_coupled": all(r["sign_equals_rademacher_of_seed"] and r["perm_equals_permutation_of_seed_plus_offset"] for r in rows),
        "wrap_seed": wrap_seed, "wrap_perm_seed": wrap_seed + PERM_OFFSET,
        "wrap_perm_equals_permutation_of_seed_0": bool(
            torch.equal(sp_wrap.permutation_key.permutation, PermutationKey(size=D, seed=0).permutation)
        ),
    }

    # V5: torch's CPU generator keeps only the low 32 bits of an integer seed.
    trunc_rows = []
    for s in (0, LEGACY_SEED_ARG, 1234567, sign_seed):
        trunc_rows.append({
            "seed": s,
            "rademacher_equal_to_seed_plus_2_32": bool(
                torch.equal(RademacherKey._generate_key(SHAPE, s), RademacherKey._generate_key(SHAPE, s + 2**32))
            ),
            "permutation_equal_to_seed_plus_2_32": bool(
                torch.equal(PermutationKey._generate_permutation(D, s), PermutationKey._generate_permutation(D, s + 2**32))
            ),
        })
    try:
        RademacherKey._generate_key(SHAPE, 2**64)
        overflow = "no error"
    except Exception as exc:  # noqa: BLE001 - recording the exception type is the point
        overflow = type(exc).__name__
    checks["V5"] = {
        "rows": trunc_rows,
        "all_truncated": all(r["rademacher_equal_to_seed_plus_2_32"] and r["permutation_equal_to_seed_plus_2_32"] for r in trunc_rows),
        "seed_2_64_raises": overflow,
        "torch_version": torch.__version__,
    }

    # V6: two different passwords that give the identical 65536-dim key.
    coll_sign = find_collision("")
    ka = RademacherKey(shape=SHAPE, password=coll_sign["password_a"]).key
    kb = RademacherKey(shape=SHAPE, password=coll_sign["password_b"]).key
    coll_sign["keys_equal"] = bool(torch.equal(ka, kb))
    coll_sign["coords_equal"] = int((ka == kb).sum())
    coll_perm = find_collision("_permutation")
    pa = PermutationKey(size=D, password=coll_perm["password_a"]).permutation
    pb = PermutationKey(size=D, password=coll_perm["password_b"]).permutation
    coll_perm["keys_equal"] = bool(torch.equal(pa, pb))
    coll_perm["positions_equal"] = int((pa == pb).sum())
    checks["V6"] = {
        "sign": coll_sign, "perm": coll_perm,
        "birthday_expected_hashes_32bit": math.sqrt(math.pi / 2 * 2**32),
    }

    for name, body in checks.items():
        if name == "V1":
            continue
        _assert_checks(name, body)
    return checks


def _assert_checks(name: str, body: dict) -> None:
    expected_true = {
        "V2": ["rademacher_equal", "permutation_equal", "sp_sign_equal", "sp_perm_equal", "sp_ciphertext_equal",
               "key_values_are_pm1", "perm_is_valid_permutation"],
        "V3": ["s0_both_equals_password_only", "s1_both_sign_equals_password_only", "s1_both_perm_equals_password_only"],
        "V4": ["all_coupled", "wrap_perm_equals_permutation_of_seed_0"],
        "V5": ["all_truncated"],
    }.get(name, [])
    for key in expected_true:
        if not body[key]:
            raise SystemExit(f"check {name}.{key} failed")
    if name == "V3" and (body["s0_both_equals_seed_only"] or body["s1_both_sign_equals_seed_only"]
                         or body["s1_both_perm_equals_seed_only"]):
        raise SystemExit("check V3: seed unexpectedly influenced the key")
    if name == "V6":
        for part in ("sign", "perm"):
            if not (body[part]["keys_equal"] and body[part]["digests_differ"] and body[part]["same_first_4_bytes"]):
                raise SystemExit(f"check V6.{part} failed")


# ---------------------------------------------------------------------------
# claimed vs actual table
# ---------------------------------------------------------------------------
def log2_factorial(n: int) -> float:
    return math.lgamma(n + 1) / math.log(2)


def build_rows() -> list[dict]:
    claim_s0 = float(D)
    claim_s1 = D + log2_factorial(D)
    rows = [
        {
            "scheme": "S0 Rademacher", "mode": "password (legacy configuration)",
            "claim_expression": "2^d", "claim_bits": claim_s0, "actual_bits": 32, "gap_bits": claim_s0 - 32,
            "known_plaintext_search_bits": 32,
            "basis": (f"{ANON}:99-100 seed=int.from_bytes(sha256(pw)[:4]); :85-87 manual_seed(seed)+bernoulli -> k=f(32-bit seed); "
                      "checks V1,V2,V5,V6"),
        },
        {
            "scheme": "S0 Rademacher", "mode": "integer seed",
            "claim_expression": "2^d", "claim_bits": claim_s0, "actual_bits": 32, "gap_bits": claim_s0 - 32,
            "known_plaintext_search_bits": 32,
            "basis": f"{ANON}:73-75,85-87 key=bernoulli(generator.manual_seed(seed)); torch CPU generator keeps low 32 bits of seed; check V5",
        },
        {
            "scheme": "S1 Signed Permutation", "mode": "integer seed (sign and permutation coupled)",
            "claim_expression": "2^d * d!", "claim_bits": claim_s1, "actual_bits": 32, "gap_bits": claim_s1 - 32,
            "known_plaintext_search_bits": 32,
            "basis": f"{ANON}:358 perm_seed = seed + 1000000, :344-348 sign from the same seed -> (k,pi)=f(32-bit seed); checks V4,V5",
        },
        {
            "scheme": "S1 Signed Permutation", "mode": "password (legacy configuration; seed argument ignored)",
            "claim_expression": "2^d * d!", "claim_bits": claim_s1, "actual_bits": 64, "gap_bits": claim_s1 - 64,
            "known_plaintext_search_bits": 33,
            "basis": (f"{ANON}:99-100 sign seed from sha256(pw)[:4] and {ANON}:189-191 permutation seed from sha256(pw+'_permutation')[:4] "
                      "-> two independent 32-bit seeds (upper bound 64 bits; 33 bits if the two halves are searched separately, "
                      "see method note section 5); checks V1,V2,V3,V6"),
        },
    ]
    return rows


def write_csv(rows: list[dict]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scheme", "mode", "claim_expression", "claim_bits", "actual_bits", "gap_bits",
              "known_plaintext_search_bits", "basis"]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("claim_bits", "gap_bits"):
                out[key] = f"{row[key]:.1f}"
            writer.writerow(out)


# ---------------------------------------------------------------------------
# method note
# ---------------------------------------------------------------------------
def yes(flag: bool) -> str:
    return "是" if flag else "否"


def write_note(rows: list[dict], checks: dict, anchors: list[dict], script_sha: str) -> None:
    v1, v2, v3, v4, v5, v6 = (checks[k] for k in ("V1", "V2", "V3", "V4", "V5", "V6"))
    ln = {a["id"]: f"{Path(a['file']).name}:{a['line']}" for a in anchors}
    s1_bits = rows[2]["claim_bits"]
    log2_fact = log2_factorial(D)
    table = ["| 方案 | 模式 | 宣稱 | claim_bits | actual_bits（上界） | 已知明文下搜尋 bits | gap_bits |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['scheme']} | {r['mode']} | {r['claim_expression']} | {r['claim_bits']:,.1f} | "
                     f"{r['actual_bits']} | {r['known_plaintext_search_bits']} | {r['gap_bits']:,.1f} |")

    lines = [
        "# 前作金鑰空間審計：方法說明（A3.1）",
        "",
        f"- 任務：WORKFLOW A3.1 effective key-space audit；seed = {SEED}；CPU only，未載入擴散模型 checkpoint。",
        f"- 產生方式：`{COMMAND}`（由腳本寫出本檔與 `paper_assets/tables/table_keyspace.csv`，內容對固定原始碼、torch、numpy 版本為確定性）。",
        f"- 被審計原始碼：`{ANON}`（sha256 = `{sha256_file(ANON_PATH)}`）；torch {torch.__version__}；numpy {np.__version__}。",
        f"- 產生腳本 sha256：`{script_sha}`。",
        "- 標記說明：[事實] = 逐行讀碼或本腳本實測（附來源）；[推定] = 由 [事實] 推導，未獨立量測；[假設] = 尚未驗證的前提。",
        "",
        "## 1. 結論",
        "",
        "- [事實] 前作的 Rademacher 金鑰 k（d = 65536 維、值域 ±1）完全由一個 32-bit 整數 seed 經 `torch.Generator.manual_seed` 與 "
        f"`torch.bernoulli` 決定（{ln['rad_seed_slice']}、{ln['rad_manual_seed']}、{ln['rad_bernoulli']}）。"
        f"`actual_bits` 上界為 32，宣稱為 {D:,}，差 {D - 32:,} bits。",
        f"- [事實] Signed Permutation 在整數 seed 模式下，置換的 seed 為 `seed + 1000000`（{ln['sp_perm_offset']}），符號與置換由同一個 32-bit 值決定，"
        f"`actual_bits` 上界為 32，宣稱 2^d·d!（約 {s1_bits:,.1f} bits）。",
        "- [事實] 前作實際執行的 S1 設定（`run_cfg_inference_Crypto.sh:7`：`--anonymization_key_password Ki@13579 --use_signed_permutation True`）走 password 模式："
        "符號 seed 與置換 seed 是兩個獨立的 32-bit 值，因此 `actual_bits` 上界是 64，不是 32。這是對「32 bits」說法的重要修正，見第 5 節。",
        "- [推定] 上述上界均為 bits 數的上限；「搜尋成本」尚須 oracle，見 A3.2（實測吞吐與外推）。本檔不宣稱任何實際攻擊成功率。",
        "",
        "## 2. 程式碼路徑（[事實]，每個 file:line 都在執行時逐行驗證）",
        "",
        "| 步驟 | 位置 | 內容 |",
        "|---|---|---|",
    ]
    step_ids = [
        ("宣稱（模組 docstring）", "claim_module_docstring"),
        ("宣稱（SignedPermutationKey docstring）", "claim_class_docstring"),
        ("宣稱（取樣腳本日誌）", "sampler_log_claim"),
        ("S0：password 分支優先於 seed 分支", "rad_password_branch"),
        ("S0：SHA-256", "rad_sha256"),
        ("S0：只取前 4 bytes", "rad_seed_slice"),
        ("S0：seed 餵給 Generator", "rad_manual_seed"),
        ("S0：Bernoulli(0.5) 產生金鑰", "rad_bernoulli"),
        ("S1 置換：password 分支優先於 seed 分支", "perm_password_branch"),
        ("S1 置換：password + `_permutation` 後綴", "perm_suffix"),
        ("S1 置換：SHA-256", "perm_sha256"),
        ("S1 置換：只取前 4 bytes", "perm_seed_slice"),
        ("S1 置換：randperm", "perm_randperm"),
        ("S1 整數 seed：置換 seed 為固定偏移", "sp_perm_offset"),
        ("S1：符號金鑰以 seed/password 建構", "sp_rademacher_ctor"),
        ("S1：置換金鑰以 seed+偏移/password 建構", "sp_perm_ctor"),
        ("取樣腳本：S1 以 seed 與 password 同時建構", "sampler_sp_ctor"),
        ("啟動命令同時傳 seed 42 與 password", "shell_flags"),
        ("E2.2 wrapper 以 password 與 seed 同時建構（與前作一致）", "e22_wrapper_ctor"),
    ]
    by_id = {a["id"]: a for a in anchors}
    for label, aid in step_ids:
        a = by_id[aid]
        text = a["text"].replace("|", "\\|")
        lines.append(f"| {label} | `{a['file']}:{a['line']}` | `{text}` |")
    lines += [
        "",
        "- [事實] 論文本身的宣稱：§4.3.1 寫「The key space of this scheme is 2^d」，§4.3.2 寫「a key space of up to 2^d × d!」"
        f"（`{PAPER_PDF}`，PDF 第 25、26 頁，由 pdftotext 取得）。",
        "",
        "## 3. 宣稱 vs 實際（對應 `paper_assets/tables/table_keyspace.csv`）",
        "",
        *table,
        "",
        f"- [事實] d = {D:,}；claim_bits 的 S0 = d，S1 = d + log2(d!) = {D:,} + {log2_fact:,.1f}（以 `math.lgamma` 計算）。",
        "- [事實] `actual_bits` 欄是「金鑰可由多少個獨立 bits 決定」的上界（由程式路徑得出），不是攻擊成本；"
        "`known_plaintext_search_bits` 是在攻擊者持有一對明文/密文 latent 時、對 seed 做窮舉所需的 bits（第 5 節）。",
        "",
        "## 4. 數值驗證（皆以前作自己的 `RademacherKey`、`PermutationKey`、`SignedPermutationKey` 實測）",
        "",
        f"- V1 [事實] `sha256(\"{LEGACY_PASSWORD}\")` = `{v1['sign_digest']}`，只用前 4 bytes → 符號 seed = {v1['sign_seed']}"
        f"（位於 [0, 2^32) 的 {v1['sign_seed_fraction_of_2_32']:.4f} 處）；其餘 224 bits 不影響金鑰。"
        f"`sha256(\"{LEGACY_PASSWORD}_permutation\")` = `{v1['perm_digest']}` → 置換 seed = {v1['perm_seed']}。",
        f"- V2 [事實] 只用整數 seed 重建金鑰：`RademacherKey(seed={v1['sign_seed']}).key` 與 `RademacherKey(password=...)` 在 {v2['rademacher_coords_equal']:,}/{D:,} 個座標相同"
        f"（torch.equal = {yes(v2['rademacher_equal'])}）；`PermutationKey(seed={v1['perm_seed']})` 與 password 版在 {v2['permutation_positions_equal']:,}/{D:,} 個位置相同"
        f"（torch.equal = {yes(v2['permutation_equal'])}）；以兩個整數 seed 組成的 `SignedPermutationKey` 對同一個 N(0,1) latent（seed {SEED}）的密文與 password 版逐位元相同"
        f"（{yes(v2['sp_ciphertext_equal'])}）。即整把 {D:,} 維金鑰只由 32-bit 整數決定。",
        f"- V3 [事實] password 優先於 seed：`RademacherKey(seed=42, password=\"{LEGACY_PASSWORD}\")` 與只給 password 的金鑰完全相同（{yes(v3['s0_both_equals_password_only'])}），"
        f"與只給 seed=42 的金鑰不同（符號一致率 {v3['s0_sign_agreement_with_seed42']:.4f}）；`SignedPermutationKey` 同理，符號與置換皆與只給 password 者相同，"
        f"與只給 seed=42 者不同（置換位置一致率 {v3['s1_perm_position_agreement_with_seed_only']:.6f}）。"
        "因此啟動命令的 `--anonymization_key_seed 42` 對實際金鑰沒有任何作用。",
        "- V4 [事實] 整數 seed 模式下 S1 的耦合：" + "；".join(
            f"seed={r['seed']} → 符號 = Rademacher(seed)（{yes(r['sign_equals_rademacher_of_seed'])}）、置換 = Permutation(seed+1000000)（{yes(r['perm_equals_permutation_of_seed_plus_offset'])}）"
            for r in v4["rows"]
        ) + f"。另 seed = 2^32 − 1000000 = {v4['wrap_seed']} 時 perm_seed = 2^32，其置換與 seed 0 的置換相同（{yes(v4['wrap_perm_equals_permutation_of_seed_0'])}），見 V5。",
        f"- V5 [事實] torch {v5['torch_version']} 的 CPU generator 只保留整數 seed 的低 32 bits：" + "；".join(
            f"seed={r['seed']} 與 seed+2^32 產生相同的金鑰（{yes(r['rademacher_equal_to_seed_plus_2_32'])}）與置換（{yes(r['permutation_equal_to_seed_plus_2_32'])}）"
            for r in v5["rows"]
        ) + f"；seed = 2^64 會丟出 `{v5['seed_2_64_raises']}`。所以即使呼叫端傳入 64-bit 整數，金鑰也至多 32 bits（上界與 SHA-256 截斷無關，獨立成立）。",
        f"- V6 [事實] 兩個不同密碼得到完全相同的金鑰：`{v6['sign']['password_a']}` 與 `{v6['sign']['password_b']}` 的 SHA-256 前 4 bytes 相同"
        f"（seed = {v6['sign']['seed']}，共計算 {v6['sign']['hashes_computed']:,} 次雜湊；32-bit 生日界預期約 {v6['birthday_expected_hashes_32bit']:,.0f} 次），完整 digest 卻有 {v6['sign']['differing_hex_chars']} 個十六進位字元不同；"
        f"兩者的 `RademacherKey` 在 {v6['sign']['coords_equal']:,}/{D:,} 個座標相同（{yes(v6['sign']['keys_equal'])}）。"
        f"置換同理：`{v6['perm']['password_a']}` 與 `{v6['perm']['password_b']}`（後綴 `_permutation`，seed = {v6['perm']['seed']}，{v6['perm']['hashes_computed']:,} 次雜湊）的 `PermutationKey` 在 "
        f"{v6['perm']['positions_equal']:,}/{D:,} 個位置相同（{yes(v6['perm']['keys_equal'])}）。",
        "",
        "## 5. 對「32 bits」說法的限定與細節",
        "",
        "1. [事實] 預設 legacy 設定（password 模式）下，S0 的 `actual_bits` 上界是 32；S1 的上界是 64，因為符號與置換各由一個獨立的 32-bit 前綴決定"
        f"（{ln['rad_seed_slice']} 與 {ln['perm_seed_slice']}，且置換的 password 分支先於 seed 分支，{ln['perm_password_branch']}，所以 `seed + 1000000` 的耦合在 password 模式不會發生）。",
        "2. [事實] 只有整數 seed 模式才有 `seed + 1000000` 耦合（V4），此時 S1 的上界是 32。本專案 legacy 重現（E2.2，`e2_legacy_wrapper.py:70-74`）與前作一樣同時給 seed 與 password，此時 seed 被忽略（V3），走 password 模式。",
        "3. [推定] 在攻擊者持有一對明文/密文 latent 時，S1 的置換可由 |z| 排序配對、符號可由逐元素比較直接讀出（連續分布下幅值理論上互異），"
        "因此符號 seed 與置換 seed 可各自獨立窮舉：候選數約 2·2^32 = 2^33，而非 2^64。此推論的 S0 部分與縮小空間 S1 positive control 見 A3.2，完整 S1 KPA 屬 A4.3，不在本任務內。",
        "   [事實] 但 float32 latent 並非幅值互異：A3.3 的 10 個 d=65536 latent 的相異幅值數皆小於 65536（`results/A3.3.json` 的 `inputs.*.stats.n_distinct_magnitudes`，即有座標共用相同幅值），"
        "僅憑幅值排序時這些座標的配對有歧義；其處理屬 A4.3，不影響本檔的 bits 上界。",
        "4. [推定] 在僅有密文、且 oracle 無法分開評估符號與置換的情況下，搜尋空間上界仍是 2^64（S1 password 模式）。是否存在這樣的 oracle 不在本任務檢驗範圍。",
        "5. [假設] 若密碼本身是人類可記憶字串（例如本案的 `Ki@13579`），實際熵可能遠低於 32 bits（字典或規則式猜測）；本檔只量測由 SHA-256 前綴施加的 32-bit 上界，不量測密碼熵。",
        "6. [推定] 32 是上界而非精確值：seed → 金鑰的映射在 2^32 個 seed 上是否全為單射本腳本未窮舉驗證（V6 只顯示不同密碼可共用同一 seed）。",
        "7. [事實] 金鑰由 password 確定性導出（V2），且啟動命令寫死單一 password（`run_cfg_inference_Crypto.sh:7`）；[推定] 因此同一 password 下的所有影像共用同一把金鑰，恢復一把即可還原全部——這屬於金鑰重用問題，與位元數是兩個獨立議題。",
        "",
        "## 6. 與後續任務的接口",
        "",
        "- [事實] A3.2（`results/A3.2.json`）用同一批前作金鑰類別在縮小空間（2^20）上做窮舉 positive control，並量測吞吐後外推到 2^32；本檔不含任何計時。",
        "- [事實] A3.3 驗證 S0 逐元素 |z_enc| = |z| 與 S1 排序幅值不變，這是不需要金鑰就能比對 gallery 的依據；本檔只處理金鑰空間。",
        "- [推定] 修補方向（D5.1）應使用 256-bit 隨機主金鑰，並由 CSPRNG／KDF 展開到 d 維，使每一位元都來自完整熵；該設計的檢驗不在本任務內。",
        "",
        "## 7. 重現",
        "",
        "```",
        COMMAND,
        "```",
        "",
        f"- 驗證項共 {len(anchors)} 個 file:line 錨點與 V1–V6 六組數值檢查；任何一項失敗腳本即以非零狀態結束，不寫出檔案。",
    ]
    NOTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    NOTE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=(__doc__ or '').strip().splitlines()[0])
    add_force_flag(parser)
    args = parser.parse_args()
    refuse_overwrite([CSV_OUT, NOTE_OUT], args.force)
    anchors = verify_anchors()
    checks = run_checks()
    rows = build_rows()
    script_sha = sha256_file(Path(__file__).resolve())
    write_csv(rows)
    write_note(rows, checks, anchors, script_sha)
    print(f"wrote {CSV_OUT.relative_to(ROOT)}")
    print(f"wrote {NOTE_OUT.relative_to(ROOT)}")
    for row in rows:
        print(f"{row['scheme']:<24} {row['mode']:<56} claim={row['claim_bits']:>12,.1f} actual={row['actual_bits']:>3} kpa_search={row['known_plaintext_search_bits']}")


if __name__ == "__main__":
    main()
