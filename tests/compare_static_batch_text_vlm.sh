#!/usr/bin/env bash
# Select legacy groups or named vision datasets with the same model selectors.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BENCHMARK_MODE=static_batch
export BATCH_SIZES="${BATCH_SIZES:-1 8}"
exec bash "$HERE/compare_text_vlm_16.sh"
