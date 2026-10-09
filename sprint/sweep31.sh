#!/bin/bash
# sweep31.sh - run the four-model ink sweep over a 31-layer published surface volume.
#
#   sweep31.sh <render.zarr> <out_dir> [D_MIN D_MAX]        (default -7 +7)
#
# This is Erwin Nieuwlaar's run_depth_sweep.sh with three changes, and nothing else:
#
#   N_LAYERS=31      the published renders are 31 layers, not the 66 he rendered himself.
#                    The villa models take a 17-layer window s..s+16 with s = N/2 + d - 8,
#                    so 31 layers admit d = -7..+7 and no more. Hecate centres on
#                    (N-1)/2 + d = 15 + d.
#   no padded run    the tile-phase trick exists because his segment is two windings
#                    joined at a seam; a single auto-grown segment has no seam, so the
#                    second (zero-padded) run and merge_tile_phase.py are not needed.
#                    That halves the GPU time.
#   d = 0 outwards   unchanged from his script, and worth restating: a partial run is
#                    still useful because the central depths land first.
#
# Both directions are always computed (--direction both inside run_ink.sh): on an unread
# scroll we do not know which face of the sheet carries the ink.
set -euo pipefail
[[ $# -ge 2 ]] || { awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2; }
Z=$1; OUT=$2; DMIN=${3:--7}; DMAX=${4:-7}
ROOT=${ROOT:-/data}
export PY=${PY:-$ROOT/venv/bin/python}
export CKPT_DIR=${CKPT_DIR:-$ROOT/ckpt}
export HECATE_PY=${HECATE_PY:-$CKPT_DIR/hecate.py}
export N_LAYERS=31
export NATIVE_UM=${NATIVE_UM:-8.64}
export TMPDIR=${TMPDIR:-$ROOT/tmp}
mkdir -p "$TMPDIR" "$OUT"
S=${SWEEP_SCRIPTS:-$ROOT/pherc343/ink/scripts}

(( DMIN >= -7 && DMAX <= 7 )) || { echo "31 layers admit only d = -7..+7 (got $DMIN..$DMAX)"; exit 2; }

# P_VILLA/P_HEC: he ran 2+2 concurrently on 24 GB (~7.5 GB peak each on a 4090).
# On a 16 GB card leave these at 1.
export P_VILLA=${P_VILLA:-1} P_HEC=${P_HEC:-1}
export MODELS=${MODELS:-"dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43 hecate"}

echo "sweep: $(basename "$Z")  d=$DMIN..$DMAX  models: $MODELS  parallel: villa $P_VILLA / hecate $P_HEC"
time bash "$S/run_depth_sweep.sh" "$Z" "$OUT" "$DMIN" "$DMAX"
