#!/usr/bin/env bash
set -euo pipefail

# Keep a margin below GitHub's 100 MiB hard object limit.
limit_bytes="${STAGED_FILE_LIMIT_BYTES:-94371840}"
failed=0

while IFS= read -r -d '' path; do
    case "$path" in
        dataset/*|artifacts/runs/*|smoke_results/*|past/SourceCode/results/Model/*|*/__pycache__/*|*.pyc|*.pyo|*/per_sample.csv|*/per_sample.jsonl|*/per_sample.parquet)
            echo "ERROR: forbidden generated/data path is staged: $path" >&2
            failed=1
            ;;
        .env|.env.*|*/.env|*/.env.*|*.pem|*.key|id_rsa|id_rsa.pub|id_ed25519|id_ed25519.pub|*/id_rsa|*/id_rsa.pub|*/id_ed25519|*/id_ed25519.pub|secrets.*|credentials.*|*/secrets.*|*/credentials.*)
            echo "ERROR: possible secret or credential is staged: $path" >&2
            failed=1
            ;;
    esac

    size="$(git cat-file -s ":$path" 2>/dev/null || printf '0')"
    if [[ "$size" =~ ^[0-9]+$ ]] && (( size > limit_bytes )); then
        echo "ERROR: staged file exceeds ${limit_bytes} bytes: $path ($size bytes)" >&2
        failed=1
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

if (( failed != 0 )); then
    echo "Staged-file safety check failed; commit was not performed." >&2
    exit 1
fi

echo "Staged-file safety check passed."
