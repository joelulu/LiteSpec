#!/usr/bin/env bash
# One entry: eight replicas -> health/model checks -> generation -> owned cleanup.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$HERE/LiteSpec" ]]; then
    REPO_ROOT="$HERE/LiteSpec"
else
    REPO_ROOT="$(cd -- "$HERE/../.." && pwd)"
fi
export CLIENT_PYTHON="${CLIENT_PYTHON:-$REPO_ROOT/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "$CLIENT_PYTHON" -m litespec.vlm.data_pipeline
