#!/usr/bin/env bash
set -euo pipefail

visdom_data_dir="${VISDOM_DATA_DIR:-/data2/paper/.visdom}"
mkdir -p "$visdom_data_dir"

exec conda run --no-capture-output -n CFG_DDIM \
    python -m visdom.server \
    -port 8850 \
    -bind_local \
    -env_path "$visdom_data_dir"
