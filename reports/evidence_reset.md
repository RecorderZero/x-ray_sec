# E0 Evidence Reset

## Canonical evidence

All commands must run through `scripts/run_cfg_ddim.sh`, which disables user-site packages.
Stable hashes below exclude runtime/timing fields but retain every scientific value.

| Source | Command summary | Stable SHA-256 |
|---|---|---|
| `artifacts/preflight/canonical_direction_candidates_d4096_n100.json` | `validate_direction_candidates.py --dimension 4096 --gallery-size 100` | `8f9345a896a420533fb3c1f7fcb4369d8f7a4599672982e758d0bf4aa6dd18f3` |
| `artifacts/preflight/canonical_direction_candidates_d65536_n100.json` | `validate_direction_candidates.py --dimension 65536 --gallery-size 100` | `cff4835a9f4410e4631b27adc928000cef5d12a024a9083334627165768d5a11` |
| `artifacts/preflight/canonical_sot_seed_sweep_d65536_n100.json` | `validate_sot_bitexact_sweep.py --dimension 65536 --seed-start 0 --seed-stop 100 --rounds 1 2 4` | `722e81f3e14009fa7a16d6162f8fbcb26268003cb7e91a80a91ce4ca4627538e` |

## Referenced-number reconciliation

| Prior reference | Current reconstruction | Status |
|---|---|---|
| d=65,536, one canonical gallery, 97/100 | `97/100` (seed 1911 gallery) | REPRODUCED; this is a gallery-level rate |
| d=65,536, seeds 0–99, R=2, 98/100 | `98/100` | REPRODUCED; distinct seed-sweep protocol |
| d=4,096 bit-exact 100% | `100/100` (N=100) | REPRODUCED for current N=100 protocol |
| R=1/2/4 seed sweep | `98/100`, `98/100`, `98/100` | REPRODUCED |
| 2026-09-10 gallery=2000 Householder/SOT table in `defense_design_check.py` | no current script reproduces its full Householder table | HISTORICAL / WITHDRAWN from canonical claims |

The apparent 97/100 versus 98/100 difference is not a contradiction: the first tests 100 vectors in one seeded gallery with fixed transform keys; the second tests one vector under each of 100 RNG seeds. They are different estimands.

## Current conclusions

- d=4,096 SOT-WHT float32 bit-exact rate is `1`; d=65,536 is `0.97` for the gallery protocol.
- d=65,536 SOT KS statistic/p-value are `0.00509392` / `0.0106447`; d=4,096 values are `0.00227454` / `0.663631`. A single p-value is reported, not used as proof of normality.
- SOT preserves L2 norm (`r=1`), so exact-pair norm linkage remains possible.
- Fresh-pad CDF has d=65,536 KS statistic/p-value `0.00252384` / `0.541027`, but reused pads remain broken by one known pair in this synthetic test.

## Claim boundaries

- SOT-WHT is not universally float32 bit-exact; the rate depends on dimension, rounds, implementation and sampled vectors.
- Cosine similarity is not byte equality or lossless recovery.
- CPU Gaussian tests do not establish checkpoint/CXR quality, diagnostic utility, IND-CPA, or IND-CCA security.
- The old gallery=2000 table is retained only as history; it is not cited as current evidence.
