# P11.6 Claim audit

> Scanner: `scripts/p11_claim_audit.py` (re-runnable). Scans paper-facing docs
> (`reports/*.md`, `paper_assets/**/*.md`, `PROPOSAL.md`, `README.md`) for
> over-claim phrases; auditor files, the PROGRESS log and `past/` are excluded.
> Pairs with the 10-claim `claim_matrix/index.md` + its Section-5 limits.

## Verdict: PASS — 0 unsupported over-claims

30 phrase hits; 20 negated/scoped on the same line, the remaining 10 reviewed
by hand and all legitimate:

| file:line | phrase | why it is OK |
|---|---|---|
| contribution.md:14, :73 | 無損 / lossless | inside「證偽『0.9989 = 無損』」— **refuting** the predecessor's lossless claim |
| contribution.md:17, :54 | 無損 | scoped to the **crypto transform only**, stated with the exact residual (latent cosine 1.0, max_abs ~8.3e-7); matches audited H3c / AF-040 (CLOSED). "無損" here = lossless to float precision, residual shown adjacent |
| deployment_modes.md:13, :64 | 100% / 無損再解碼 | the **M2 latent-payload** path: AEAD decryption returns the stored latent bytes bit-exact (true by construction, Q10.4) — not a pixel-space or pipeline-wide claim |
| metric_semantics.md:34 | lossless / zero-distortion | the correction doc **listing the predecessor's over-claim phrases to avoid** (quoting their `.tex`), not asserting them |

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
