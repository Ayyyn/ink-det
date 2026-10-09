#!/bin/bash
# sweep.sh - four-model ink sweep over any surface volume, whatever its depth.
#
#   sweep.sh <render.zarr> <out_dir> [D_MIN D_MAX]
#
# Layer count and pixel size are read from the render, not assumed. That matters:
# Erwin Nieuwlaar's own renders are 66 layers at 8.64 um, the published PHerc0800
# segments are 31 at 8.64, and the PHerc0139 w035 positive control is 28 at 9.362.
# Passing the wrong pixel size silently resamples hecate onto the wrong grid.
#
# The villa models take a 17-layer window s..s+16 with s = N/2 + d - 8, so the legal
# depth range follows from N; asking for more is refused rather than clipped.
#
# No padded second run: that exists for the seam in a two-winding segment, and these
# are single patches.
set -euo pipefail
[[ $# -ge 2 ]] || { awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2; }
Z=$1; OUT=$2
ROOT=${ROOT:-/data}
export PY=${PY:-$ROOT/venv/bin/python}
export CKPT_DIR=${CKPT_DIR:-$ROOT/ckpt}
export HECATE_PY=${HECATE_PY:-$CKPT_DIR/hecate.py}
export TMPDIR=${TMPDIR:-$ROOT/tmp}
S=${SWEEP_SCRIPTS:-$ROOT/pherc343/ink/scripts}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

INFO=$("$PY" "$HERE/render_info.py" "$Z" --sh) || { echo "cannot read render metadata: $Z" >&2; exit 1; }
eval "$INFO"   # N_LAYERS PX_UM D_MIN D_MAX
DMIN=${3:-$D_MIN}; DMAX=${4:-$D_MAX}
(( DMIN >= D_MIN && DMAX <= D_MAX )) || {
  echo "$N_LAYERS layers admit only d = $D_MIN..$D_MAX (asked $DMIN..$DMAX)"; exit 2; }

export N_LAYERS NATIVE_UM=${NATIVE_UM:-$PX_UM}
export P_VILLA=${P_VILLA:-2} P_HEC=${P_HEC:-2}
export MODELS=${MODELS:-"dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43 hecate"}
mkdir -p "$TMPDIR" "$OUT"

echo "sweep $(basename "$Z"): ${N_LAYERS} layers at ${NATIVE_UM} um, d=$DMIN..$DMAX"
echo "      models: $MODELS   parallel: villa $P_VILLA / hecate $P_HEC"
time bash "$S/run_depth_sweep.sh" "$Z" "$OUT" "$DMIN" "$DMAX"
