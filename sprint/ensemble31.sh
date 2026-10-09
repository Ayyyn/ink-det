#!/bin/bash
# ensemble31.sh - turn a finished sweep into the four-model z-score ensemble images.
#
#   ensemble31.sh <render.zarr> <maps_dir> <out_dir> [STAT_DEPTHS] [WRITE_DEPTHS]
#
# Defaults to the statistics over every depth of the sweep (-7..7) and writes images at
# d = -1, 0, +1. One contrast stretch per model is fixed across all depths first, then
# the four models are averaged as z-scores: that is what separates real ink from a single
# model's noise, and it is the step that distinguishes the winning submission from the
# single-checkpoint runs everyone else had already done on these renders.
#
# Outputs <name>__ensemble__d+NN.png (reading side), _blind variants, per-model maps,
# the CT layer, and <name>__stats.json with every number used.
set -euo pipefail
[[ $# -ge 3 ]] || { awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2; }
Z=$1; MAPS=$2; OUT=$3; STAT=${4:--7..7}; WRITE=${5:--1 0 1}
ROOT=${ROOT:-/data}
PY=${PY:-$ROOT/venv/bin/python}
S=${SWEEP_SCRIPTS:-$ROOT/pherc343/ink/scripts}
export N_LAYERS=31
mkdir -p "$OUT"
NAME=$(basename "$Z" .zarr)
cd "$OUT"
"$PY" "$S/ensemble_maps.py" --surface "$Z" --name "$NAME" --maps "$MAPS" --out . \
    --stat-depths $STAT --write-depths $WRITE
ls -la "$OUT" | head -30
