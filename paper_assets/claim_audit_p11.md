# P11.6 Claim audit

> Scanner: `scripts/p11_claim_audit.py` (re-runnable). Scans paper-facing docs
> (`reports/*.md`, `paper_assets/**/*.md`, `PROPOSAL.md`, `README.md`) for
> over-claim phrases; auditor files, the PROGRESS log and `past/` are excluded.
> Pairs with the 10-claim `claim_matrix/index.md` + its Section-5 limits.

## Verdict: PASS — 0 unsupported over-claims (after AF-043 fix)

50 phrase hits; disposition is by **per-hit manual review**, not the same-line
negation heuristic (which auditor AUD-20261023-01 correctly flagged as unreliable
in both directions — a nearby "AEAD"/negation can mis-scope a real claim, and a
real negation can be missed). The heuristic's SCOPED/REVIEW→ tag is only a sort
order; every REVIEW→ hit below is confirmed by hand.

**AF-043 fix (2026-10-23):** the scanner previously lacked the project's key
distinguishing phrase `精確還原 / exact-recover`, so one image-space over-claim
escaped it — `method_comparison.md:10` claimed the key-holder "精確還原原圖"
(exact image recovery), contradicting our own frozen evidence (M2 exact-latent
ceiling 41.87 dB, Q11 recovery ~38 dB — neither is pixel bit-exact). The phrase
is now scanned; the three affected spots (`method_comparison.md:10, :20 table,
:35`) are reworded to "金鑰控制高保真還原 (latent/crypto exact; image ~38–42 dB;
byte-exact original needs M2 AEAD-ref)". `bit-exact` is deliberately **not** banned
— it is used correctly throughout to mark the latent/crypto layer, not an over-claim.

| file:line | phrase | why it is OK |
|---|---|---|
| contribution.md:14, :73 | 無損 / lossless | inside「證偽『0.9989 = 無損』」— **refuting** the predecessor's lossless claim |
| contribution.md:17, :54 | 無損 | scoped to the **crypto transform only**, stated with the exact residual (latent cosine 1.0, max_abs ~8.3e-7); matches audited H3c / AF-040 (CLOSED). "無損" here = lossless to float precision, residual shown adjacent |
| deployment_modes.md:13, :64 | 100% / 無損再解碼 | the **M2 latent-payload** path: AEAD decryption returns the stored latent bytes bit-exact (true by construction, Q10.4) — not a pixel-space or pipeline-wide claim |
| metric_semantics.md:34 | lossless / zero-distortion | the correction doc **listing the predecessor's over-claim phrases to avoid** (quoting their `.tex`), not asserting them |
| **README.md:72** | **精確還原 (exact-recovery)** | "一對精確還原**金鑰**" = KPA recovers the cipher **key** from one plaintext pair (A4.3) — not image recovery; supported |
| **method_comparison.md:27** | **精確還原 (exact-recovery)** | the **external** Secret-key-diffusion paper's own claim (arXiv 2510.01031 row, source cited) — reported, not ours |
| PROPOSAL.md:252 | perfect secrecy / IND-CPA | under the heading「不正確說法:」— **negative enumeration** of what NOT to claim |
| results_index.md:31, claim_audit_p11.md:* | IND-CPA / lossless / etc. | this report + the results index **listing the avoided phrases** (meta); not assertions |

## Guarantees explicitly NOT claimed (Section-5, enforced)

- No "perfect security / IND-CPA / IND-CCA / semantic / provable / unconditional /
  information-theoretic security" is asserted anywhere; where these terms appear they
  are negated (`reports/evidence_reset.md`, `reports/t2wb_protocol.md`, `metric_semantics.md`).
- No pixel-space "lossless / zero-distortion / bit-exact" is claimed for M1 image hand-off
  (end-to-end PSNR ~30–39 dB, L∞ 0.4–0.6 reported instead).
- The structured transform is called lossless only at the latent-transform step, always
  with the max_abs ~8.3e-7 residual shown (never for the whole pipeline).

## How to re-run
`scripts/run_cfg_ddim.sh python -m scripts.p11_claim_audit` — any new `REVIEW→` line
in a paper-facing doc must be scoped/negated or carry adjacent residual evidence before freeze.
