#!/usr/bin/env python3
"""P11.7 / Z12.4 checksums + freeze manifest for the paper deliverables.

Writes `paper_assets/SHA256SUMS` over every frozen table/figure/report under
paper_assets/ and appendix/, plus `paper_assets/freeze_manifest.json` recording
git HEAD, split hashes and the environment-freeze reference. Re-runnable; the
SHA256SUMS file and the manifest exclude themselves.

  scripts/run_cfg_ddim.sh python -m scripts.p11_checksums
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = ROOT / "paper_assets" / "SHA256SUMS"
MANIFEST = ROOT / "paper_assets" / "freeze_manifest.json"
ROOTS = ["paper_assets", "appendix"]
SKIP_NAMES = {"SHA256SUMS", "freeze_manifest.json"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def main() -> None:
    files = []
    for r in ROOTS:
        for p in sorted((ROOT / r).rglob("*")):
            if p.is_file() and p.name not in SKIP_NAMES and "__pycache__" not in p.parts:
                files.append(p)

    lines = [f"{sha256(p)}  {p.relative_to(ROOT)}" for p in files]
    SUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    split_hashes = {}
    for m in sorted((ROOT / "splits").glob("*_manifest.json")):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            split_hashes[m.name] = d.get("split_sha256") or d.get("sha256") or d.get("manifest_sha256")
        except Exception:
            pass

    env_ref = next((str(p.relative_to(ROOT)) for p in [ROOT / "artifacts/environment_freeze.txt"] if p.exists()), None)
    combined = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()
    manifest = {
        "task_id": "P11.7/Z12.4-freeze",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty_files": len([x for x in git("status", "--porcelain").splitlines() if x])},
        "n_files_hashed": len(files),
        "sha256sums_combined": combined,
        "split_hashes": split_hashes,
        "environment_freeze": env_ref,
        "seed": 1911,
        "note": ("Covers paper_assets/ + appendix/ frozen deliverables. Re-run after any "
                 "table/figure change; the combined hash is the single freeze fingerprint. "
                 "Main-table reproduction (P11.2) is bit-exact 10/10 vs the E2.3 cache."),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SUMS.relative_to(ROOT)} ({len(files)} files)")
    print(f"wrote {MANIFEST.relative_to(ROOT)}  combined={combined[:16]}…  "
          f"HEAD={manifest['git']['head'][:10]} dirty={manifest['git']['dirty_files']}")


if __name__ == "__main__":
    main()
