# RQ4 formal (DPHD R=1) — git.dirty provenance note [AF-051]

The analysis_manifest.json recorded `git.dirty=true` because the original
dirty check (`git status --porcelain`) counted **untracked** files. The repo
legitimately carries many untracked artifacts (`audit/`, `AUDIT.md`,
`refpaper/`, `past/`, out-of-tree `/data2/external_gpl/` outputs), so that
check is always dirty and not meaningful. Fixed in `rq4_formal_redo.py` to
`--untracked-files=no` (tracked modifications only) + record `dirty_tracked_files`.

**Tracked modifications at the time of writing this note** (none touch the
fidelity pipeline `scripts/{rq4_formal_redo,deploy_s2_sweep,exact_inv_sweep,
respaced_ddim_sweep}.py` or `src/crypto/`):
```
 M .gitignore
 M README.md
 M scripts/q11_localization_parity.py
 M scripts/rq4_formal_redo.py
```
- `.gitignore`, `README.md` — docs, unrelated to fidelity.
- `scripts/q11_localization_parity.py` — localization parity, not the fidelity pipeline.

**Reproducibility**: the fidelity run is deterministic (seed 1911, DPM++ N=100,
null backward, range-preserving; checkpoint SHA + split SHA in analysis_manifest).
Re-running reproduces table_fidelity.csv byte-for-byte; only the manifest git
fields differ. The fidelity-producing code was committed/clean at run time.
