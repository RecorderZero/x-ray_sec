# E2–F9 Formal Compute Budget

> Decision date: 2026-10-08. The user fixed the current formal split at
> `security_v1`, N=200. Expansion is considered only after smoke/pilot runs pass;
> any expansion creates `security_v2` rather than modifying v1 in place.

## Basis and unit

- Measured planning rate: batch 8, guidance 0, approximately 16.6 seconds per
  image per complete DDIM inversion/reconstruction cycle (`E1.2`).
- One N=200 cycle therefore costs `200 × 16.6 s = 3,320 s = 0.922 GPU-hours`.
- A half-cycle is one inversion or one generation pass and is budgeted as
  0.461 GPU-hours for N=200.
- Times below are planning estimates, not runtime results. Add 25% for model
  loading, I/O, validation, contention, and failed-run recovery.
- T1/KPA/CPA/norm/linkage calculations reuse cached latent tensors and are not
  charged an additional diffusion cycle. Their smaller CPU/GPU tensor costs
  must still be measured in the final runtime table.

## Scheme × attack/pass matrix

| Stage | P | S0 | S1 | S2a | S2 | Total equivalent cycles | N=200 GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2.3 canonical inversion cache | 1 | 0 | 0 | 0 | 0 | 1.0 | 0.92 |
| Anonymous-image generation from cached latent | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 2.0 | 1.84 |
| T2-WB image re-inversion positive control/attack | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 2.0 | 1.84 |
| Correct-key recovered-image generation | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 2.0 | 1.84 |
| T1 magnitude/sorted-magnitude/norm/linkage | 0 | 0 | 0 | 0 | 0 | 0 | cache only |
| T3 known/chosen-plaintext key recovery | 0 | 0 | 0 | 0 | 0 | 0 | cache/synthetic only |
| T4 tamper rejection | 0 | 0 | 0 | 0 | 0 | 0 | must reject before diffusion |
| Q8 wrong-key/quality diagnostic generation | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 2.0 | 1.84 |
| F9 one independent critical-path reproduction | 1 | 1 | 1 | 1 | 1 | 5.0 | 4.61 |
| **Core subtotal** | **2** | **2.5** | **2.5** | **2.5** | **2.5** | **14.0** | **12.91** |

With the 25% operational reserve, the formal N=200 core budget is approximately
`16.14 GPU-hours`. A two-cycle contingency for selective reruns raises the cap
to approximately `18.45 GPU-hours`. Automatic retries remain limited by the
loop specification and cannot change seed, split, metric, or method parameters.

## Scale comparison and decision

| Formal N | Hours per cycle | 14-cycle core | Core +25% | Core +25% +2-cycle contingency |
|---:|---:|---:|---:|---:|
| 200 | 0.92 | 12.91 | 16.14 | 18.45 |
| 500 | 2.31 | 32.28 | 40.35 | 46.11 |
| 1,000 | 4.61 | 64.56 | 80.69 | 92.22 |

N=200 is retained because it supports every P0 scheme/attack cell while keeping
the formal run recoverable within the eight-week schedule. Increasing N before
the cache, T2-WB positive control, lifecycle logging, and attack schemas pass
would multiply failures rather than improve evidence. After those pilots pass,
the decision can be revisited using measured end-to-end time; the existing
`security_v1` membership and hash remain frozen in all cases.
