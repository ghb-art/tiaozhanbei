#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
mkdir -p runtime logs
exec 9>runtime/p0b1_continuation.lock
if ! flock -n 9; then
  echo "[$(date -Is)] another P0-B1 continuation process already owns the lock" >&2
  exit 3
fi

status_is_passed() {
  local path="$1"
  [[ -f "$path" ]] && "$PYTHON_BIN" - "$path" >/dev/null 2>&1 <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('status')=='passed' else 1)
PY
}

training_is_active() {
  pgrep -f 'model_compression/train_p0a5_lora.py.*configs/p0b1_converged_shared.json' \
    >/dev/null 2>&1
}

require_free_disk_gb() {
  local minimum_gb="$1" available_kb
  available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
  if (( available_kb < minimum_gb * 1024 * 1024 )); then
    echo "[$(date -Is)] insufficient free disk: require ${minimum_gb}GB before merge/quantize" >&2
    exit 4
  fi
}

echo "[$(date -Is)] waiting for P0-B1 verified data"
while ! status_is_passed reports/audit/gate_p0b1_code_data.json || \
      ! status_is_passed reports/audit/gate_p0b1_nlp_teacher_data.json; do
  sleep 30
done

echo "[$(date -Is)] verified data ready; starting converged training pipeline"
bash scripts/run_p0b1.sh data-build
if status_is_passed reports/audit/gate_p0b1_train_shared.json; then
  echo "[$(date -Is)] training audit already passed"
elif training_is_active; then
  echo "[$(date -Is)] current P0-B1 training detected; waiting without starting a duplicate"
  while training_is_active && ! status_is_passed reports/audit/gate_p0b1_train_shared.json; do
    sleep 60
  done
  if ! status_is_passed reports/audit/gate_p0b1_train_shared.json; then
    echo "[$(date -Is)] training exited without a passed audit; continuation stopped" >&2
    exit 5
  fi
else
  bash scripts/run_p0b1.sh train
fi

require_free_disk_gb 15
bash scripts/run_p0b1.sh merge
bash scripts/run_p0b1.sh imatrix-corpus
bash scripts/run_p0b1.sh quantize

gate_rc=0
bash scripts/run_p0b1.sh gate300 || gate_rc=$?
echo "[$(date -Is)] 300-item test finished rc=$gate_rc; formal full runs regardless"

full_rc=0
bash scripts/run_p0b1.sh full || full_rc=$?
echo "[$(date -Is)] P0-B1 terminal full finished rc=$full_rc"
exit "$full_rc"
