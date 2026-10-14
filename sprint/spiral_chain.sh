#!/usr/bin/env bash
# spiral_chain.sh - grow new geometry on a scroll nobody has read, then look for ink on it.
#
# This is Erwin Nieuwlaar's PHerc0343 chain (reproduce.sh, villa 0fb38c45, VC3D 41701aa)
# retargeted at a scroll with NO published segments, which is the whole point: PHerc0800's
# geometry was handed to us and produced a null, and the thesis this project runs on says
# the bottleneck is geometry, not ink. PHerc0268 and PHerc1218 are the two eligible 8.64 um
# scrolls that publish both inputs the chain consumes (m7 surface + lasagna) and have zero
# segments, so nobody has a head start.
#
#   SCROLL=PHerc0268 VILLA=/data/villa VC3D=/data/VC3D-41701aa.AppImage \
#   PY=/data/venv/bin/python CT=/data/ct.zarr bash sprint/spiral_chain.sh /data/work
#
# Phases, in order. Each is resumable and skipped if its output exists:
#
#   preflight  resolve S3 assets, check the eligible volume, read lasagna_scale   (seconds)
#   umbilicus  estimate the scroll axis by converging lasagna normals             (minutes)
#   dataset    assemble the conventional fit_spiral dataset root                  (seconds)
#   sdt        make_surf_sdt.py: capped signed-distance volume from m7            (~1 h CPU)
#   fit        fit_spiral: 30 000 steps, clockwise, patches disabled              (hours, GPU)
#   snap       snap_meshes.py: nodes onto the predicted sheet                     (minutes)
#   flatten    lasagna flatten per winding                                        (minutes, GPU)
#   render     VC3D 66 layers, one voxel apart                                    (~1 h)
#   ink        sweep_dual.py + ensemble.sh + score_maps.py                        (~1 h, GPU)
#
# Run `--dry-run` first. It prints every command with its real resolved paths and touches
# nothing, which is how you find a wrong path for free instead of 40 minutes into a rental.
#
# WHAT IS NOT VERIFIED HERE, AND WHY: the fit and flatten invocations are written from the
# winning entry's own README and from a *different* villa commit than the pinned one, as
# the local clone is not 0fb38c45. `preflight` locates each tool in the actual checkout and
# fails if one is missing, but the exact flag spellings of fit_spiral at 0fb38c45 must be
# confirmed on the box with --help before the first paid run. Treat the fit phase as the
# one step that may need a correction; everything around it is checked.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-$PWD/work}
: "${SCROLL:?set SCROLL, e.g. PHerc0268}"
PY=${PY:-python}
DRY=${DRY:-0}
ONLY=${ONLY:-}
Z0=${Z0:-}            # z band, level-0 voxels; empty = whole scroll (expensive)
Z1=${Z1:-}
STEPS=${STEPS:-30000}
SENSE=${SENSE:-clockwise}
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --only=*)  ONLY="${arg#--only=}" ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run()  { if [ "$DRY" = 1 ]; then printf '   $ %s\n' "$*"; else printf '   $ %s\n' "$*"; eval "$@"; fi; }
skip() { printf '   (exists, skipping: %s)\n' "$1"; }
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

mkdir -p "$WORK"
MANIFEST=$WORK/preflight.json

# ---------------------------------------------------------------- preflight
if want preflight; then
  say "preflight: $SCROLL"
  "$PY" "$HERE/spiral_preflight.py" --scroll "$SCROLL" \
      ${VILLA:+--villa "$VILLA"} ${VC3D:+--vc3d "$VC3D"} --json "$MANIFEST"
fi
[ -f "$MANIFEST" ] || { echo "no $MANIFEST; run the preflight phase first" >&2; exit 1; }

read_json() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get(sys.argv[2]) or '')" "$MANIFEST" "$1"; }
SURF=$(read_json surface)
LAS=$(read_json lasagna)
VOL=$(read_json volume_store)
LAS_GROUP=$(read_json lasagna_group)
LAS_SCALE=$(read_json lasagna_scale)
CT=${CT:-$VOL}
echo "   surface  $SURF"
echo "   lasagna  $LAS  (group $LAS_GROUP, scale $LAS_SCALE)"
echo "   ct       $CT"

# ---------------------------------------------------------------- umbilicus
UMB=$WORK/umbilicus.json
if want umbilicus; then
  say "umbilicus: the scroll axis"
  if [ -f "$UMB" ]; then skip "$UMB"; else
    # The centroid of surface voxels is NOT the axis of a crushed spiral - that mistake
    # already killed this project's region scoreboard. make_umbilicus.py converges the
    # published lasagna normals instead, which is the geometry the fitter actually wants.
    run "\"$PY\" \"$HERE/make_umbilicus.py\" --lasagna \"$LAS\" --scroll \"$SCROLL\" --out \"$UMB\" --group \"$LAS_GROUP\""
  fi
  echo "   REVIEW THIS IN VC3D before a submission run. The winning entry used a"
  echo "   hand-made umbilicus; a wrong one yields a plausible but wrong spiral."
fi

# ---------------------------------------------------------------- dataset root
DS=$WORK/dataset
if want dataset; then
  say "dataset: the conventional fit_spiral layout"
  # fit_spiral takes --dataset, not loose paths, and resolves inputs by fixed relative
  # names (fit_session.py: _CONVENTIONAL_INPUT_RELATIVES). Build exactly that tree.
  run "mkdir -p \"$DS/lasagna_inputs\""
  run "cp \"$UMB\" \"$DS/umbilicus.json\""
  # These must be LOCAL paths: fit_spiral opens them as plain stores, and an s3:// string
  # cannot be symlinked. Syncing a prediction is fine - the "never download a volume" rule
  # is about the multi-TB CT, not these. Expect tens of GB; check the size before pulling.
  for c in nx ny grad_mag; do
    D="$DS/lasagna_inputs/las_008_${c}.ome.zarr"
    if [ -d "$D" ]; then skip "$D"; else
      run "aws s3 sync --no-sign-request \"$LAS${SCROLL}_${c}.ome.zarr/$LAS_GROUP/\" \"$D/$LAS_GROUP/\""
      run "aws s3 cp --no-sign-request \"$LAS${SCROLL}_${c}.ome.zarr/.zattrs\" \"$D/.zattrs\""
      run "aws s3 cp --no-sign-request \"$LAS${SCROLL}_${c}.ome.zarr/.zgroup\" \"$D/.zgroup\""
    fi
  done
  echo "   only group $LAS_GROUP is synced - that is the one fit_spiral reads at lasagna_scale $LAS_SCALE"
  # Erwin ran with spiral tracks; neither PHerc0268 nor PHerc1218 publishes any, and their
  # patch makes the outer shell optional, so the fit runs without both. Expect a weaker fit
  # than the winning one had - this is the single biggest known difference from that run.
  echo "   note: no tracks/ and no outer_shell/ - both optional, both absent for this scroll"
fi

# ---------------------------------------------------------------- SDT
SDT=$WORK/surf_sdt.zarr
if want sdt; then
  say "sdt: capped signed distance from the m7 surface prediction"
  if [ -d "$SDT" ]; then skip "$SDT"; else
    ZR=""
    [ -n "$Z0" ] && ZR="--z-range $Z0:$Z1"
    # level 1 at threshold 128, CT-masked with a 1-voxel dilation: the winning settings
    run "\"$PY\" \"\$(\"$PY\" -c \"import json;print(json.load(open('$MANIFEST',encoding='utf-8'))['tools']['make_surf_sdt.py'])\")\" \
        --surf \"$SURF\" --group 1 --threshold 128 --out \"$SDT\" \
        --ct \"$CT\" --ct-dilate 1 $ZR --resume"
  fi
fi

# ---------------------------------------------------------------- fit
FIT=$WORK/fit
if want fit; then
  say "fit: spiral, $STEPS steps, sense $SENSE, patches disabled"
  cat > "$WORK/fit_overrides.json" <<JSON
{
  "input_disable_patches": true,
  "loss_weight_shell_outer": 0,
  "loss_weight_shell_patch_radius": 0,
  "dense_spacing_mode": "phase",
  "optimizer_num_training_steps": $STEPS,
  "lasagna_scale": $LAS_SCALE,
  "sense": "$SENSE"${Z0:+,
  "z_begin": $Z0,
  "z_end": $Z1}
}
JSON
  echo "   overrides -> $WORK/fit_overrides.json"
  run "cat \"$WORK/fit_overrides.json\""
  echo "   CONFIRM FLAG SPELLINGS FIRST:  \$PY \$(fit_spiral.py) --help"
  echo "   fit_spiral reads config overrides from the environment (get_env_config_overrides)."
  run "FIT_SPIRAL_CONFIG=\"$WORK/fit_overrides.json\" \"$PY\" \"\$(\"$PY\" -c \"import json;print(json.load(open('$MANIFEST',encoding='utf-8'))['tools']['fit_spiral.py'])\")\" --dataset \"$DS\""
fi

# ---------------------------------------------------------------- snap
SNAP=$WORK/snapped
if want snap; then
  say "snap: mesh nodes onto the predicted sheet (the winner's own contribution)"
  run "ls \"$FIT\" > \"$WORK/windings.txt\" || true"
  run "SURF_ZARR=\"$SURF\" \"$PY\" \"\$(\"$PY\" -c \"import json;print(json.load(open('$MANIFEST',encoding='utf-8'))['tools']['snap_meshes.py'])\")\" \
      \"$FIT\" \"$SCROLL\" \"$SNAP\" --windings \"$WORK/windings.txt\" --level 0"
  echo "   read $SNAP/snap_report.tsv: a winding with a low snapped fraction is one the"
  echo "   surface prediction does not support, and is not worth rendering."
fi

# ---------------------------------------------------------------- flatten
if want flatten; then
  say "flatten: lasagna, per winding"
  echo "   \$PY \$VILLA/lasagna/fit.py configs/flatten_fast_nofilter.json <w>_flatten_input.json --device cuda"
  echo "   (one call per snapped winding; the config is copied into each mesh's meta.json)"
fi

# ---------------------------------------------------------------- render
if want render; then
  say "render: 66 layers, one voxel apart"
  : "${VC3D:?set VC3D to the 41701aa AppImage for this phase}"
  echo "   66 layers gives d = -25..+24 for the 17-layer model window; the published 31-layer"
  echo "   renders only allow d = -7..+7. Depth coverage is a reason to render our own."
  run "bash \"$HERE/../tools/pherc343/mesh/render_66_layers.sh\" \"$VC3D\" \"$CT\" \"<mesh>.tifxyz\" \"$WORK/render_66.zarr\""
fi

# ---------------------------------------------------------------- ink
if want ink; then
  say "ink: the same sweep that is already validated at AUC 0.972 on the control"
  run "\"$PY\" \"$HERE/sweep_dual.py\" --render \"$WORK/render_66.zarr\" --out \"$WORK/maps\" --dmin -3 --dmax 3"
  run "bash \"$HERE/ensemble.sh\" \"$WORK/render_66.zarr\" \"$WORK/maps\" \"$WORK/images\" '-3..3' '-1 0 1'"
  echo "   score against the calibrated yardstick: control row_periodicity 0.497 at 4.8 mm."
  echo "   NOTE: this render is ours, not a published team render, so the face convention"
  echo "   is Erwin's (forward = plain), NOT the inverted one. Score WITHOUT --face blind"
  echo "   first and compare both ways before trusting either."
fi

say "done${DRY:+ (dry run - nothing was executed)}"
