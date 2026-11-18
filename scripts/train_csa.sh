#!/usr/bin/env bash
# Training options are configured in configs/ and can be overridden on the CLI.
set -euo pipefail
CSA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CSA_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec torchrun \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29501}" \
  train.py "$@"
