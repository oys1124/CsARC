#!/usr/bin/env bash
set -euo pipefail
CSA_SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$CSA_SCRIPTS/train_csa.sh" --config "$CSA_SCRIPTS/../configs/paper_proj256_align.json" "$@"
