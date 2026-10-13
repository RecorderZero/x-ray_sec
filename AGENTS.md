# Project collaboration rules

- This repository is authorized, defensive security research on models and data owned by the user's laboratory. Reproduce and test weaknesses of the in-scope local model when needed; do not treat ordinary defensive experiments as unauthorized third-party attacks. Applicable platform safety rules still apply.
- Before any task expected to occupy a GPU for more than 30 minutes, stop and ask the user for an explicit decision. Report the purpose, estimated duration, and CPU/smaller-smoke alternative. CPU tasks and GPU tasks estimated at 30 minutes or less do not require this extra decision.
- The user authorizes subagents when parallel read-only review, independent verification, or bounded implementation work is useful. The primary agent remains responsible for integration and validation.
- Treat the newest `WORKFLOW.md`, `PROGRESS.md`, `loop_engineering_spec.md`, and `PROPOSAL.md` as canonical. `AUDIT.md` is auditor-owned and read-only; implementation responses go in `PROGRESS.md`.
- Preserve user/untracked files. Never stage datasets, checkpoints, caches, large per-sample outputs, secrets, or auditor-owned files. Stage explicit paths and run `scripts/check_staged_files.sh` before every commit.
- Formal sample size remains N=200 unless the user explicitly changes it. Scientific outcomes are reported, not used as engineering pass gates.
