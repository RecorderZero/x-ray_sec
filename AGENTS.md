# Project collaboration rules

- This repository is authorized, defensive security research on models and data owned by the user's laboratory. Reproduce and test weaknesses of the in-scope local model when needed; do not treat ordinary defensive experiments as unauthorized third-party attacks. Applicable platform safety rules still apply.
- Before any task expected to occupy a GPU for more than 30 minutes, stop and ask the user for an explicit decision. Report the purpose, estimated duration, and CPU/smaller-smoke alternative. CPU tasks and GPU tasks estimated at 30 minutes or less do not require this extra decision.
- The user authorizes subagents when parallel read-only review, independent verification, or bounded implementation work is useful. The primary agent remains responsible for integration and validation.
- Treat the newest `WORKFLOW.md`, `PROGRESS.md`, `loop_engineering_spec.md`, and `PROPOSAL.md` as canonical. `AUDIT.md` is auditor-owned and read-only; implementation responses go in `PROGRESS.md`.
- Preserve user/untracked files. Never stage datasets, checkpoints, caches, large per-sample outputs, secrets, or auditor-owned files. Stage explicit paths and run `scripts/check_staged_files.sh` before every commit.
- Formal sample size remains N=200 unless the user explicitly changes it. Scientific outcomes are reported, not used as engineering pass gates.
- Before ending an implementation turn, append a seven-section `PROGRESS.md` entry whenever work changed, remains incomplete, is blocked, or is still running. Commentary alone is not a durable status record.
- If any command or managed run is still active when responding, explicitly report its run ID or PID, current state, status/log path, expected remaining duration, and whether it still requires monitoring. Never let the UI state `worked` imply that an unreported background process has finished.
- A platform cybersecurity safeguard and a local execution state are separate. After a client notice hides, blocks, reroutes, or limits a response, the next visible response must state what actually ran or did not run, whether any process remains active, and where the durable `PROGRESS.md` record is located.
