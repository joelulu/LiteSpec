#!/usr/bin/env bash
# Foreground, read-only checkpoint diagnostics. GPU defaults to physical 7.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
cd "$REPO_ROOT"
RUN_GROUP="${1:-${RUN_GROUP:-}}"
if [[ -z "$RUN_GROUP" ]]; then
    RUN_GROUP="$($PYTHON_BIN - "$WORKSPACE" <<'PY'
from pathlib import Path
import sys
roots=[p for p in (Path(sys.argv[1])/'runs').glob('vlm_gdn_valid20k_*')
       if all((p/arm/'training_complete.json').is_file() for arm in ('dflash','dlite','dflash_gdn','dlite_gdn'))]
if len(roots)!=1:
    raise SystemExit('Supply the completed RUN_GROUP explicitly. Candidates:\n'+'\n'.join(str(p) for p in sorted(roots)))
print(roots[0])
PY
)"
fi
export CUDA_VISIBLE_DEVICES="${DIAG_GPU:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT="${DIAG_OUTPUT:-$RUN_GROUP/gdn_diagnostics_$(date +%Y%m%d_%H%M%S)_$$}"
echo "Run group: $RUN_GROUP"
echo "GPU: $CUDA_VISIBLE_DEVICES; diagnostics: $OUTPUT"
exec "$PYTHON_BIN" tests/diagnose_gdn.py --run-group "$RUN_GROUP" --output-dir "$OUTPUT" \
    --samples-per-dataset "${DIAG_SAMPLES:-8}" --reference-tokens "${DIAG_REFERENCE_TOKENS:-64}"
