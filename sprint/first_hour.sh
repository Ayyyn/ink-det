#!/usr/bin/env bash
# first_hour.sh - everything to run on a fresh rented box BEFORE anything expensive starts.
#
# Each step here is cheap and each one checks something that otherwise fails hours in,
# after you have paid for those hours. Nothing in this script costs more than ~20 minutes,
# and it stops before the first long job.
#
#   git clone https://github.com/Ayyyn/ink-det.git /data/ink-det
#   cd /data/ink-det && SCROLL=PHerc0268 bash sprint/first_hour.sh /data
#
# When it finishes clean you will know: the environment builds, the checkpoints are what
# we think they are, fit_spiral's real flags, the scroll's assets resolve, and which z band
# is worth fitting. Then and only then start the chain.
set -euo pipefail
ROOT=${1:-/data}
SCROLL=${SCROLL:-PHerc0268}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$ROOT/venv/bin/python

# Keep every cache on local disk. setup_gpu.sh explains why (the rented box points these
# at a FUSE volume that refuses chmod); repeated here because the steps below run in this
# shell, not setup_gpu.sh's, and would otherwise inherit the box's own values.
export UV_CACHE_DIR="$ROOT/cache/uv" PIP_CACHE_DIR="$ROOT/cache/pip" \
       HF_HOME="$ROOT/cache/hf" VIRTUALENV_OVERRIDE_APP_DATA="$ROOT/cache/virtualenv" \
       XDG_CACHE_HOME="$ROOT/cache"

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }

step "0a. idle guard (start this BEFORE anything else)"
# A rented pod bills wall-clock time, not GPU time. An idle pod costs the same as a busy
# one, and this project has already burned a full set of credits on a box left up after
# its last job finished. Start the guard first so that can only happen once.
if tmux has-session -t guard 2>/dev/null; then
  echo "guard already running"
else
  tmux new-session -d -s guard "bash '$HERE/idle_guard.sh' ${IDLE_MIN:-20}" \
    && echo "guard started: pod stops after ${IDLE_MIN:-20} idle minutes" \
    || echo "*** could not start the guard - stop the pod yourself when the last job ends"
fi

step "0. the box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "cores: $(nproc)   ram: $(free -g | awk '/^Mem:/{print $2}') GB   disk: $(df -h "$ROOT" | awk 'NR==2{print $4}') free"
# The chain needs 24 GB VRAM, and the SDT stage wants cores and RAM more than it wants GPU.
# If this says <24 GB VRAM or <32 GB RAM, stop and move to another host now, not later.

step "1. environment (~15 min)"
# villa pinned to 0fb38c45, Python 3.14, the four sha256-verified checkpoints
bash "$HERE/setup_gpu.sh" "$ROOT"

step "2. what the checkpoints actually expect"
# Never yet run on a real box. villa centre-pads a SHORT layer selection with ZEROS and
# says nothing, so a model wanting >17 layers would have silently degraded every map we
# have produced so far. This is the check that confirms it did not.
"$PY" "$HERE/checkpoint_info.py" "$ROOT/ckpt"

step "3. fit_spiral's real flags"
# The one step the local clone could not verify: our checkout is not the pinned commit, so
# the invocation in spiral_chain.sh is written from the winning entry's README. Confirm it
# here, for free, instead of discovering it after the SDT build.
FIT=$(find "$ROOT/villa" -name fit_spiral.py | head -1)
echo "fit_spiral: ${FIT:-NOT FOUND}"
[ -n "$FIT" ] && "$PY" "$FIT" --help 2>&1 | head -40 || echo "*** find it before going further"

step "4. the scroll's assets"
"$PY" "$HERE/spiral_preflight.py" --scroll "$SCROLL" --villa "$ROOT/villa" \
    --json "$ROOT/work/preflight.json"

step "5. choose the z band (the decision that sets the whole run)"
# Do NOT copy the winning entry's z 9500-11000: that band was chosen for PHerc0343. Run the
# umbilicus over the WHOLE scroll at a coarse group first - it is cheap - and read its
# per-block report. Blocks with many samples and a low residual are where the sheets are
# well-formed and worth fitting; blocks that come back UNDETERMINED are crushed or empty.
"$PY" "$HERE/make_umbilicus.py" \
    --lasagna "$("$PY" -c "import json;print(json.load(open('$ROOT/work/preflight.json',encoding='utf-8'))['lasagna'])")" \
    --scroll "$SCROLL" --group 4 --out "$ROOT/work/umbilicus_survey.json"
echo
echo "Pick Z0/Z1 from that table: a 1500-voxel band over the best-conditioned run of blocks."
echo "Then re-run make_umbilicus at --group 2 over just that band for the fit itself."

step "6. dry run the chain"
SCROLL=$SCROLL VILLA=$ROOT/villa PY=$PY bash "$HERE/spiral_chain.sh" "$ROOT/work" --dry-run

cat <<'NEXT'

=== first hour clean. Nothing expensive has run yet. ===

Start the real chain inside tmux, with the band you chose:

  tmux new -s chain
  SCROLL=PHerc0268 VILLA=/data/villa VC3D=/data/VC3D-41701aa.AppImage \
    PY=/data/venv/bin/python Z0=<z0> Z1=<z1> \
    bash sprint/spiral_chain.sh /data/work

Watch for, in order:
  * the SDT finishing at roughly the size preflight predicted - if it is far larger, the
    band did not apply and the disk will fill;
  * snap_report.tsv - a winding with a low snapped fraction is one the surface prediction
    does not support, and is not worth rendering;
  * the ink score against the calibrated yardstick: the control reads row_periodicity
    0.497 at 4.8 mm. Below ~0.25 is a null.

And on OUR renders the face convention is Erwin's (forward = plain), NOT the inverted one
that applies to published team renders. Score both ways before trusting either.
NEXT
