# E2–F9 Formal Compute Budget

> Decision date: 2026-10-08. The user fixed the current formal split at
> `security_v1`, N=200. Expansion is considered only after smoke/pilot runs pass;
> any expansion creates `security_v2` rather than modifying v1 in place.
>
> Revised 2026-10-09 (AF-021, AUD-20261009-01 §8-Q2): costs are now measured on
> the predecessor's **anonymization pipeline** (`ddim_sample_loop_anonymization`)
> that P/S0/S1 actually share (AF-017), with generation at `guidance_scale=-1`
> (one model call per step, as hard-coded by the predecessor). The earlier
> version used the E1 inference-flow cycle with guidance 0 (two calls per
> generation step) and did not charge P's anonymous generation or the T2-WB
> positive control on P.

## Basis and unit

Measured by `scripts/e2_anonymization_runner.py benchmark`
(`results/AF021_anonymization_benchmark.json`, managed run
`AF021_ANON_BENCH_20261009T152410677702Z_a5f3dcdc`), timestep 250, 10 steps × 3
repeats, per image, scaled by 499 steps (noise level 500):

| Batch | Forward half-cycle F (null=True) | Generation half-cycle G (−1) | Generation at guidance 0 | Full cycle F+G (−1) | Full cycle at guidance 0 |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.73 s | 7.82 s | 15.40 s | 15.55 s | 23.13 s |
| 4 | 6.31 s | 6.33 s | 12.59 s | 12.64 s | 18.90 s |
| 8 | 5.54 s | 5.52 s | 11.01 s | 11.07 s | 16.55 s |

- Cross-checks: batch 1 matches the auditor's 15.9 s (−1) and 23.0 s (0) per
  image (AUD-20261009-02); the batch-8 guidance-0 cycle (16.55 s) matches the
  E1.2 basis used previously (16.6 s). The P smoke measured 15.67 s for one
  complete batch-1 anonymization (`results/AF017_P_anonymization_smoke.csv`).
- Planning below is given for batch 8 (F = 5.54 s, G = 5.52 s per image) and
  batch 1 (F = 7.73 s, G = 7.82 s). Model loading, I/O, validation and
  contention are covered by a 25% reserve.
- `[事實]` Batch composition changes the GPU result: an E2.3 smoke that cached
  four `security_v1` latents at batch 4 and recomputed them at batch 1 gave
  MaxAbs 6.93e-4 (> the E2.3 tolerance 1e-5, 0/4 bit-exact), while caching and
  recomputing at batch 1 gave MaxAbs 0 (4/4 bit-exact).
- `[決定]` (implementer, 2026-10-10) Formal passes run at **batch 1**: the E2.3
  tolerance, the T1 reuse of P outputs and every bit-exact regression test are
  defined at batch 1. Switching downstream generation to batch 8 would save
  about 29% but requires recording the batch composition and redoing the
  recomputation checks at that composition; it is not adopted unless the user
  asks for it.
- T1/KPA/CPA/norm/linkage and tamper checks reuse cached tensors and are not
  charged a diffusion pass; their small costs are measured in the runtime table.

## Pass matrix (half-cycles per image)

The forward pass is key-independent in the legacy pipeline, so one cached
`x_T` per image serves all schemes.

| Stage | P | S0 | S1 | S2a | S2 | F | G | N=200 GPU-hours (batch 8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E2.3 forward cache (shared by all schemes) | F | — | — | — | — | 1 | 0 | 0.31 |
| Anonymous image generation from keyed latent | G | G | G | G | G | 0 | 5 | 1.53 |
| T2-WB re-inversion of the stored anonymous PNG (P = positive control) | F | F | F | F | F | 5 | 0 | 1.54 |
| M1 legitimate recovery generation after inverse key | G | G | G | G | G | 0 | 5 | 1.53 |
| T1 recovery from the exact decrypted latent | 0¹ | 0² | 0² | G | G | 0 | 2 | 0.61 |
| Q8 wrong-key / wrong-nonce generation | — | G | G | G | G | 0 | 4 | 1.23 |
| F9 one independent critical-path reproduction (cache + anonymous + T2-WB + M1) | | | | | | 6 | 10 | 4.92 |
| **Core total** | | | | | | **12** | **26** | **11.67** |

¹ For P the decrypted latent is the cached `x_T`, so T1 recovery is the P anonymous output itself.
² S0/S1 transforms round-trip bit-exactly (E2.2), so T1 recovery equals the P output; this is verified per sample and charged one G only if a sample is not bit-exact (covered by the contingency below).

## Scale comparison and decision

| Formal N | Batch | Core | Core +25% | Core +25% +2 full-cycle contingency |
|---:|---:|---:|---:|---:|
| 200 | 8 | 11.67 | 14.59 | 15.82 |
| 200 | **1 (adopted)** | **16.45** | **20.56** | **22.29** |
| 500 | 8 | 29.18 | 36.48 | 39.55 |
| 500 | 1 | 41.13 | 51.41 | 55.73 |
| 1,000 | 8 | 58.37 | 72.96 | 79.11 |
| 1,000 | 1 | 82.26 | 102.82 | 111.46 |

N=200 is retained (user decision 2026-10-08). At batch 8 the revision would
lower the N=200 cap from 18.45 to 15.82 GPU-hours (one model call per
generation step, shared forward cache); at the adopted batch 1 the cap is
22.29 GPU-hours, still within the eight-week schedule. Automatic retries remain limited
by the loop specification and cannot change seed, split, metric or method
parameters; the existing `security_v1` membership and hash remain frozen.
