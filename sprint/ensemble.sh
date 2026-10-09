#!/bin/bash
# ensemble.sh - per-model contrast stretch over the swept depths, then the z-score mean.
#
#   ensemble.sh <render.zarr> <maps_dir> <out_dir> [STAT_DEPTHS] [WRITE_DEPTHS]
#
# Depth defaults follow the render's layer count. Fixing one stretch per model across
# all depths before averaging is the step that separates this from a single-checkpoint
# run, and it is what the PHerc0343 win turned on.
set -euo pipefail
[[ $# -ge 3 ]] || { awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2; }
Z=$1; MAPS=$2; OUT=$3
ROOT=${ROOT:-/data}
PY=${PY:-$ROOT/venv/bin/python}
S=${SWEEP_SCRIPTS:-$ROOT/pherc343/ink/scripts}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
eval "$("$PY" "$HERE/render_info.py" "$Z" --sh)"
export N_LAYERS
STAT=${4:-$D_MIN..$D_MAX}; WRITE=${5:--1 0 1}
mkdir -p "$OUT"
NAME=$(basename "$Z" .zarr)
cd "$OUT"
"$PY" "$S/ensemble_maps.py" --surface "$Z" --name "$NAME" --maps "$MAPS" --out . \
    --stat-depths $STAT --write-depths $WRITE
ls "$OUT" | head -30
