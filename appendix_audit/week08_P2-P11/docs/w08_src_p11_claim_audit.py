#!/usr/bin/env python3
"""P11.6 claim audit: scan paper-facing text for over-claim phrases.

Flags every occurrence of a banned/overclaim phrase in the deliverable docs so a
human can confirm each is NEGATED or scoped (e.g. "we do NOT claim IND-CPA"),
not an unsupported assertion. Auditor files (AUDIT*), the PROGRESS log and the
read-only predecessor tree are excluded.

  scripts/run_cfg_ddim.sh python -m scripts.p11_claim_audit
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# phrase -> why it is sensitive
BANNED = {
    r"perfect(ly)?\s+secur": "perfect security",
    r"perfect\s+secrecy": "perfect secrecy",
    r"\bIND-?CPA\b": "IND-CPA",
    r"\bIND-?CCA\b": "IND-CCA",
    r"semantic(ally)?\s+secur": "semantic security",
    r"provabl[ey]\s+secur": "provable security",
    r"unbreakable": "unbreakable",
    r"unconditional(ly)?\s+secur": "unconditional security",
    r"information[- ]theoretic": "information-theoretic security",
    r"\blossless\b": "lossless",
    r"無損": "lossless(zh)",
    r"完美(安全|保密)": "perfect security(zh)",
    r"100\s*%\s*(secure|安全|恢復|還原|reversib)": "100% claim",
    r"zero[- ]distortion": "zero-distortion",
    # AF-043: the project's key distinguishing phrase -- image-space "exact recovery"
    # is only true at the latent/crypto layer or via the M2 AEAD-ref payload, never
    # for the M1 image hand-off (~38-42 dB). "bit-exact" is deliberately NOT banned:
    # it is the precise term the project uses correctly, not an over-claim word.
    r"精[確準](還原|恢復)": "exact-recovery(zh)",
    r"exact(ly)?[- ]recover": "exact-recovery",
}

# search these globs; exclude auditor/log/predecessor
INCLUDE = ["reports/*.md", "paper_assets/**/*.md", "PROPOSAL.md", "README.md"]
NEG_HINT = re.compile(r"(不|非|並非|未|no\b|not\b|never|без|避免|不宣稱|不等同|不以|不得|排除|without|≠)", re.I)


def main() -> None:
    files: list[Path] = []
    for g in INCLUDE:
        files += [p for p in ROOT.glob(g) if p.is_file()]
    files = sorted(set(files))

    hits = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            for pat, label in BANNED.items():
                if re.search(pat, line, re.I):
                    scoped = bool(NEG_HINT.search(line))
                    hits.append((f.relative_to(ROOT), i, label, scoped, line.strip()[:160]))

    print(f"scanned {len(files)} paper-facing files; {len(hits)} phrase hits\n")
    unscoped = [h for h in hits if not h[3]]
    for f, i, label, scoped, text in hits:
        tag = "SCOPED  " if scoped else "REVIEW→ "
        print(f"{tag}{f}:{i}  [{label}]\n          {text}")
    print(f"\n{len(hits)-len(unscoped)}/{len(hits)} occurrences are negated/scoped on the same line; "
          f"{len(unscoped)} need manual confirmation (REVIEW→).")


if __name__ == "__main__":
    main()
