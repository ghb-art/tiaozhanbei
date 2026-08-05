#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG="configs/p0b2_code_grpo.json"
GPUS="${P0B2_GPUS:-0,1,2,3}"
BASE="models/checkpoints/p0a4/student-shared-merged"
ADAPTER="models/checkpoints/p0b2/code-grpo"
DATA_AUDIT="reports/audit/gate_p0b2_code_data.json"
PREFLIGHT_AUDIT="reports/audit/gate_p0b2_code_preflight.json"
TRAIN_AUDIT="reports/audit/gate_p0b2_code_train.json"

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not p.is_file(): raise SystemExit(f'Missing audit: {p}')
status=json.loads(p.read_text(encoding='utf-8')).get('status')
if status not in allowed: raise SystemExit(f'Audit rejected: {p} status={status} allowed={sorted(allowed)}')
print(f'Audit guard passed: {p} status={status}')
PY
}

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[value.strip() for value in sys.argv[1].split(',') if value.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(value.isdigit() for value in values):
 raise SystemExit(f'P0-B2 requires four distinct GPU ids: {values}')
print('P0-B2 GPU group:', ','.join(values))
PY
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0b2_code_data.py --config "$CONFIG"
  require_status "$DATA_AUDIT" passed
}

preflight() {
  data_build
  validate_gpus
  [[ -d "$BASE" ]] || { echo "Missing base model: $BASE" >&2; return 1; }
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0b2_code_grpo.py \
      --config "$CONFIG" --smoke \
      --audit "$PREFLIGHT_AUDIT" \
    2>&1 | tee logs/p0b2_code_grpo_preflight.log
  require_status "$PREFLIGHT_AUDIT" dry_run_passed
}

latest_checkpoint() {
  local output="$1" checkpoint
  [[ -d "$output" ]] || return 0
  while IFS= read -r checkpoint; do
    if [[ -s "$checkpoint/adapter_config.json" \
       && -s "$checkpoint/adapter_model.safetensors" \
       && -s "$checkpoint/optimizer.pt" ]]; then
      printf '%s\n' "$checkpoint"
    fi
  done < <(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' | sort -V)
}

train() {
  local resume=() checkpoint=""
  preflight
  if [[ -f "$TRAIN_AUDIT" ]] && \
     [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$TRAIN_AUDIT")" == passed ]]; then
    echo "P0-B2 Code GRPO training already complete."
    return 0
  fi
  checkpoint="$(latest_checkpoint "$ADAPTER" | tail -n 1)"
  if [[ -n "$checkpoint" ]]; then
    resume=(--resume-from-checkpoint "$checkpoint")
    echo "Resuming P0-B2 Code GRPO from $checkpoint"
  fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0b2_code_grpo.py \
      --config "$CONFIG" \
      --audit "$TRAIN_AUDIT" \
      "${resume[@]}" \
    2>&1 | tee -a logs/p0b2_code_grpo_train.log
  require_status "$TRAIN_AUDIT" passed
}

status() {
  echo "P0-B2 Code GRPO status:"
  for audit in "$DATA_AUDIT" "$PREFLIGHT_AUDIT" "$TRAIN_AUDIT"; do
    if [[ -f "$audit" ]]; then
      "$PYTHON_BIN" - "$audit" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
print(f"  {sys.argv[1]}: status={data.get('status')}")
PY
    else
      echo "  $audit: missing"
    fi
  done
}

case "${1:-}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  status) status ;;
  *)
    echo "Usage: $0 {data-build|preflight|train|status}" >&2
    return 1
    ;;
esac
