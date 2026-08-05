#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A8_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A8_SERVE_GPU:-0}"
PORT="${P0A8_PORT:-18462}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
OUTPUT_DIR="models/checkpoints/p0a8/code-specialist"
AUDIT_ROOT="reports/audit/p0a8"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; return 1; }
}

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not path.is_file(): raise SystemExit(f"Missing audit: {path}")
status=json.loads(path.read_text(encoding='utf-8')).get('status')
if status not in allowed: raise SystemExit(f"Audit rejected: {path} status={status}")
print(f"Audit guard passed: {path} status={status}")
PY
}

validate_gpu_group() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[x.strip() for x in sys.argv[1].split(',') if x.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(x.isdigit() for x in values):
    raise SystemExit(f"P0-A8 requires four distinct GPU ids: {values}")
print("P0-A8 GPU group:", ','.join(values))
PY
}

require_checkpoint() {
  local step="$1" checkpoint="$OUTPUT_DIR/checkpoint-$1"
  require_file "$checkpoint/trainer_state.json"
  require_file "$checkpoint/adapter_config.json"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" && ! -s "$checkpoint/adapter_model.bin" ]]; then
    echo "Missing adapter weights: $checkpoint" >&2
    return 1
  fi
}

wait_endpoint() {
  local pid="$1" ids="$2" log="$3" attempt
  for attempt in $(seq 1 240); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Model service exited before readiness; see $log" >&2
      return 1
    fi
    if "$PYTHON_BIN" - "$ENDPOINT" "$ids" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models', timeout=2) as response:
    actual={str(x.get('id')) for x in json.load(response).get('data', [])}
required={x for x in sys.argv[2].split(',') if x}
raise SystemExit(0 if required.issubset(actual) else 1)
PY
    then
      echo "Model service is ready: $ids"
      return 0
    fi
    sleep 2
  done
  echo "Model service readiness timeout; see $log" >&2
  return 1
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a8_code_validation.py
  require_status reports/audit/gate_p0a8_code_validation.json passed
}

preflight() {
  require_status reports/audit/gate_p0a6_data.json passed
  require_status reports/audit/gate_p0a8_code_validation.json passed
  require_dir "$BASE_DIR"
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a8_code_preflight.json \
    --max-steps 128 --checkpoint-steps 64 \
    --focus-domain code --learning-rate 0.00001 --dry-run
  require_status reports/audit/gate_p0a8_code_preflight.json dry_run_passed
}

train() {
  validate_gpu_group
  preflight
  if [[ -f reports/audit/gate_p0a8_train_code.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a8_train_code.json")).get("status", ""))')"
    if [[ "$status" == passed ]]; then
      require_checkpoint 64; require_checkpoint 128
      echo "P0-A8 Code training is already complete."
      return 0
    fi
  fi
  local resume=()
  if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a8_train_code.json \
    --max-steps 128 --checkpoint-steps 64 \
    --focus-domain code --learning-rate 0.00001 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume[@]}"
  require_status reports/audit/gate_p0a8_train_code.json passed
  require_checkpoint 64; require_checkpoint 128
}

evaluate_code_once() {
  local model="$1" name="${1//-/_}"
  if [[ -f "$AUDIT_ROOT/${name}.json" ]]; then
    require_status "$AUDIT_ROOT/${name}.json" passed
    return 0
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a8_code.py \
    --manifest data/p0a8/code_internal_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model" --candidate-name "$model" \
    --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" \
    --audit "$AUDIT_ROOT/${name}.json"
}

validation() {
  require_checkpoint 64; require_checkpoint 128
  require_file data/p0a8/code_internal_validation.jsonl
  require_file models/checkpoints/p0a6/student-pilot/checkpoint-200/adapter_config.json
  require_file models/checkpoints/p0a7/nlp-mmlu-aux-specialist/checkpoint-188/adapter_config.json
  mkdir -p logs runtime "$AUDIT_ROOT" reports/audit/p0a6
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a8-base \
    --lora-module "p0a6-step-200=$ROOT/models/checkpoints/p0a6/student-pilot/checkpoint-200" \
    --lora-module "p0a7-nlp-188=$ROOT/models/checkpoints/p0a7/nlp-mmlu-aux-specialist/checkpoint-188" \
    --lora-module "p0a8-code-64=$ROOT/$OUTPUT_DIR/checkpoint-64" \
    --lora-module "p0a8-code-128=$ROOT/$OUTPUT_DIR/checkpoint-128" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a8_validation_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a8_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_validation EXIT INT TERM
  wait_endpoint "$server_pid" \
    "p0a8-base,p0a6-step-200,p0a7-nlp-188,p0a8-code-64,p0a8-code-128" \
    logs/p0a8_validation_server.log

  evaluate_code_once p0a8-base
  evaluate_code_once p0a6-step-200
  evaluate_code_once p0a8-code-64
  evaluate_code_once p0a8-code-128
  "$PYTHON_BIN" scripts/select_p0a8_code.py \
    --base-audit "$AUDIT_ROOT/p0a8_base.json" \
    --candidate "64=$AUDIT_ROOT/p0a8_code_64.json" \
    --candidate "128=$AUDIT_ROOT/p0a8_code_128.json" \
    --minimum-gain 0.03 --output "$AUDIT_ROOT/checkpoint_selection.json"

  local step
  step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a8/checkpoint_selection.json"))["selected_step"])')"
  require_status reports/audit/p0a6/base_quick.json passed
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/quick_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id p0a8-base --model-id-math p0a8-base \
    --model-id-code "p0a8-code-$step" --model-id-nlp p0a7-nlp-188 \
    --candidate-name "p0a8-router-step-$step-quick" \
    --output-trace reports/audit/p0a6/p0a8_router_quick_trace.jsonl \
    --audit reports/audit/p0a6/p0a8_router_quick.json
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit reports/audit/p0a6/base_quick.json \
    --candidate "$step=reports/audit/p0a6/p0a8_router_quick.json" \
    --output reports/audit/p0a6/p0a8_quick_selection.json

  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id p0a8-base --candidate-name p0a8-base-full \
    --output-trace reports/audit/p0a6/p0a8_base_full_trace.jsonl \
    --audit reports/audit/p0a6/p0a8_base_full.json
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id p0a8-base --model-id-math p0a8-base \
    --model-id-code "p0a8-code-$step" --model-id-nlp p0a7-nlp-188 \
    --candidate-name "p0a8-router-step-$step-full" \
    --output-trace reports/audit/p0a6/p0a8_router_full_trace.jsonl \
    --audit reports/audit/p0a6/p0a8_router_full.json
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --validation-manifest data/p0a6/full_validation.jsonl \
    --base-audit reports/audit/p0a6/p0a8_base_full.json \
    --candidate "$step=reports/audit/p0a6/p0a8_router_full.json" \
    --output reports/audit/p0a6/p0a8_full_selection.json
  cleanup_validation
  trap - EXIT INT TERM
}

status() {
  local path
  for path in reports/audit/gate_p0a8_code_validation.json \
    reports/audit/gate_p0a8_code_preflight.json \
    reports/audit/gate_p0a8_train_code.json \
    reports/audit/p0a8/checkpoint_selection.json \
    reports/audit/p0a6/p0a8_quick_selection.json \
    reports/audit/p0a6/p0a8_full_selection.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"))' "$path"
    else
      echo "$path missing"
    fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  validation) validation ;;
  auto) data_build; preflight; train; validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a8.sh
    "$PYTHON_BIN" -m py_compile \
      model_compression/build_p0a8_code_validation.py \
      model_compression/train_p0a6_student.py \
      scripts/evaluate_p0a8_code.py scripts/select_p0a8_code.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a8.sh <command>"
    echo "Commands: data-build | preflight | train | validation | auto | status | structural-check"
    ;;
esac
