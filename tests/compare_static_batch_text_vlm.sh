#!/usr/bin/env bash
# Same three datasets/checkpoints, true static batches instead of round-robin C.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BENCHMARK_MODE=static_batch
export BATCH_SIZES="${BATCH_SIZES:-1 8}"
exec bash "$HERE/compare_text_vlm_16.sh"
