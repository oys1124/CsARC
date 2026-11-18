#!/usr/bin/env bash
set -euo pipefail
CSA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CSA_ROOT"
exec "${PYTHON:-python}" eval_scale_stream.py "$@"
