# E0 Evidence Reset

## Canonical evidence

- Generator: `attack/validate_direction_candidates.py`
- Command: `conda run -n CFG_DDIM python attack/validate_direction_candidates.py --dimension 65536 --gallery-size 100 --output artifacts/preflight/canonical_direction_candidates_d65536_n100.json`
- Seed: `1911`
- Source JSON SHA-256: `20b413d87317c7f0daaef967d5b278de95bcf650963b5958545acfd76beb5168`
- Tracked summary: `canonical_preflight.csv`

The JSON above and `canonical_preflight.csv` are the only canonical synthetic preflight for the current code revision. Earlier `direction_candidates_cpu*.json` files are **historical, non-canonical artifacts** because their gallery sizes differ. Their 31/32, 98/100, or similar bit-exact counts must not be mixed with the current N=100 result.

## Current conclusions

- SOT-WHT float32 bit-exact vectors: `97/100`. This is an implementation diagnostic, not a security or end-to-end image-quality guarantee.
- SOT-WHT preserves L2 norm (`r=1`), so norm-only exact-pair linkage remains possible (Top-1 `1`).
- SOT-WHT elementwise-magnitude gallery Top-1 is `0.01` for this synthetic gallery.
- Fresh-pad CDF removes the same simple linkage in this run (norm correlation `-0.200824`, norm-only Top-1 `0`), but a reused pad is recovered by one known pair and decrypts a second message with cosine `1`.
- KCI is only a toy algebraic cancellation test here: correct-key max error `7.46625e-15`; it is not evidence for O-BELM image quality or IND-CPA security.

## Withdrawn or bounded claims

- Withdraw any unqualified statement that SOT-WHT is always float32 bit-exact; the rate depends on dimension, implementation, and sampled vectors.
- Do not treat cosine 0.9989 as byte equality or lossless recovery.
- Do not cite these CPU Gaussian tests as checkpoint/CXR evidence, diagnostic-utility evidence, or a complete cryptographic proof.
