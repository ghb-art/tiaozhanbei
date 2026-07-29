#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

THRESHOLD_PERCENT="${MEMORY_GUARD_THRESHOLD_PERCENT:-60}"
INTERVAL_SECONDS="${MEMORY_GUARD_INTERVAL_SECONDS:-2}"
CONSECUTIVE_SAMPLES="${MEMORY_GUARD_CONSECUTIVE_SAMPLES:-2}"
GRACE_SECONDS="${MEMORY_GUARD_GRACE_SECONDS:-10}"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_with_memory_guard.sh COMMAND [ARG ...]" >&2
  echo "Example: bash scripts/run_with_memory_guard.sh .venv/bin/python your_long_task.py" >&2
  exit 2
fi

exec "$PYTHON_BIN" "$ROOT/scripts/memory_watchdog.py" run \
  --threshold-percent "$THRESHOLD_PERCENT" \
  --interval-seconds "$INTERVAL_SECONDS" \
  --consecutive-samples "$CONSECUTIVE_SAMPLES" \
  --grace-seconds "$GRACE_SECONDS" \
  -- "$@"
