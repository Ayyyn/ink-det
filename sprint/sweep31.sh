#!/bin/bash
# superseded by sweep.sh (layer count is detected, not assumed)
exec "$(dirname "$0")/sweep.sh" "$@"
