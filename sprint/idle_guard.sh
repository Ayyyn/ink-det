#!/usr/bin/env bash
# idle_guard.sh - stop the pod when nothing is running. Run it FIRST, in its own tmux.
#
#   tmux new -d -s guard 'bash sprint/idle_guard.sh 20'
#
# A rented pod bills WALL-CLOCK TIME, not GPU time. A pod with nothing running costs
# exactly as much as one mid-fit. This project has already lost a full set of credits to a
# pod left up overnight after its last job finished, so treat this as part of starting a
# box, not as an optional extra.
#
# It watches for any tmux session other than itself. When none has existed for IDLE_MIN
# consecutive minutes it stops the pod. Stopping (not terminating) keeps any network
# volume; the container disk is erased either way, so anything worth keeping must already
# be on the volume or pulled off the box.
set -uo pipefail
IDLE_MIN=${1:-20}
SELF=${TMUX_SESSION:-guard}
idle=0
echo "idle guard: will stop the pod after ${IDLE_MIN} min with no other tmux session"
while true; do
  busy=$(tmux ls 2>/dev/null | grep -cv "^${SELF}:" || true)
  if [ "${busy:-0}" -gt 0 ]; then
    idle=0
  else
    idle=$((idle + 1))
    echo "$(date -u +%H:%M) idle ${idle}/${IDLE_MIN} min"
  fi
  if [ "$idle" -ge "$IDLE_MIN" ]; then
    echo "$(date -u +%H:%M) idle limit reached - stopping pod ${RUNPOD_POD_ID:-?}"
    if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
      runpodctl stop pod "$RUNPOD_POD_ID"
    else
      echo "runpodctl or RUNPOD_POD_ID missing - CANNOT self-stop; stop it from the dashboard" >&2
    fi
    exit 0
  fi
  sleep 60
done
