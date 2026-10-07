#!/usr/bin/env bash
set -euo pipefail

# Keep ~/.local packages out of the research environment. Usage:
#   scripts/run_cfg_ddim.sh python -m pytest -q tests/unit
export PYTHONNOUSERSITE=1
exec conda run --no-capture-output -n CFG_DDIM "$@"
