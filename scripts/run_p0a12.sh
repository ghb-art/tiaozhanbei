#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A12_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A12_SERVE_GPU:-0}"
PORT="${P0A12_PORT:-18466}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
MATH_DIR="models/checkpoints/p0a12/math-specialist"
CODE_DIR="models/checkpoints/p0a11/code-specialist/checkpoint-250"
NLP_DIR="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
AUDIT_ROOT="reports/audit/p0a12"

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; return 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; return 1; }; }
require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not p.is_file(): raise SystemExit(f"Missing audit: {p}")
s=json.loads(p.read_text(encoding='utf-8')).get('status')
if s not in allowed: raise SystemExit(f"Audit rejected: {p} status={s}")
print(f"Audit guard passed: {p} status={s}")
PY
}

validate_gpu_group() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[x.strip() for x in sys.argv[1].split(',') if x.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(x.isdigit() for x in values):
 raise SystemExit(f"P0-A12 requires four distinct GPU ids: {values}")
print("P0-A12 GPU group:",','.join(values))
PY
}

require_checkpoint() {
  local step="$1" dir="$MATH_DIR/checkpoint-$1"
  require_file "$dir/trainer_state.json"
  require_file "$dir/adapter_config.json"
  [[ -s "$dir/adapter_model.safetensors" || -s "$dir/adapter_model.bin" ]] || {
    echo "Missing adapter weights: $dir" >&2; return 1;
  }
}

wait_endpoint() {
  local pid="$1" ids="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "Service exited before readiness" >&2; return 1; }
    if "$PYTHON_BIN" - "$ENDPOINT" "$ids" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 actual={str(x.get('id')) for x in json.load(response).get('data',[])}
needed=set(sys.argv[2].split(','))
raise SystemExit(0 if needed.issubset(actual) else 1)
PY
    then return 0; fi
    sleep 2
  done
  echo "Timed out waiting for $ENDPOINT" >&2
  return 1
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a12_data.py build
  require_status reports/audit/gate_p0a12_data.json passed
}

asset_guards() {
  require_status reports/audit/p0a11/code_selection.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  require_dir "$CODE_DIR"; require_dir "$NLP_DIR"
  "$PYTHON_BIN" - <<'PY'
import json
code=json.load(open('reports/audit/p0a11/code_selection.json'))
nlp=json.load(open('reports/audit/p0a10/nlp_selection.json'))
if code.get('selected_step') != 250: raise SystemExit('Frozen Code selection is not step 250')
if nlp.get('selected_step') != 136: raise SystemExit('Frozen NLP selection is not step 136')
print('Frozen router assets passed: code=250 nlp=136')
PY
}

preflight() {
  require_status reports/audit/gate_p0a12_data.json passed
  require_dir "$BASE_DIR"
  asset_guards
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data data/p0a12/math_train.jsonl \
    --output-dir "$MATH_DIR" --audit reports/audit/gate_p0a12_math_preflight.json \
    --max-steps 576 --checkpoint-steps 288 --focus-domain math \
    --learning-rate 0.000002 --lora-rank 16 --lora-alpha 32 --dry-run
  require_status reports/audit/gate_p0a12_math_preflight.json dry_run_passed
}

train() {
  validate_gpu_group
  preflight
  if [[ -f reports/audit/gate_p0a12_train_math.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a12_train_math.json")).get("status",""))')"
    if [[ "$status" == passed ]]; then
      require_checkpoint 288; require_checkpoint 576
      echo "P0-A12 Math training already complete."
      return 0
    fi
  fi
  local resume=()
  if [[ -d "$MATH_DIR" ]] && find "$MATH_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data data/p0a12/math_train.jsonl \
    --output-dir "$MATH_DIR" --audit reports/audit/gate_p0a12_train_math.json \
    --max-steps 576 --checkpoint-steps 288 --focus-domain math \
    --learning-rate 0.000002 --lora-rank 16 --lora-alpha 32 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume[@]}"
  require_status reports/audit/gate_p0a12_train_math.json passed
  require_checkpoint 288; require_checkpoint 576
}

evaluate_once() {
  local model="$1" name="${1//-/_}_math"
  if [[ -f "$AUDIT_ROOT/${name}.json" ]]; then
    require_status "$AUDIT_ROOT/${name}.json" passed
    return 0
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a12 --domain math --manifest data/p0a12/math_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model" --candidate-name "$model" \
    --expected-rows 1000 --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" \
    --audit "$AUDIT_ROOT/${name}.json" --workers "${P0A12_EVAL_WORKERS:-8}"
}

select_math() {
  "$PYTHON_BIN" scripts/select_p0a11_domain.py \
    --protocol p0a12 --domain math --steps 288,576 \
    --base-audit "$AUDIT_ROOT/p0a12_base_math.json" \
    --candidate "288=$AUDIT_ROOT/p0a12_math_288_math.json" \
    --candidate "576=$AUDIT_ROOT/p0a12_math_576_math.json" \
    --minimum-gain 0.02 --output "$AUDIT_ROOT/math_selection.json"
}

run_gate300_once() {
  local math_step="$1"
  local trace="data/eval/p0a12_final_router_hf_gate300.jsonl"
  local eval_audit="reports/audit/gate_p0a12_final_router_hf_gate300_eval.json"
  local retention="reports/audit/gate_p0a12_final_router_hf_gate300_retention.json"
  if [[ -e "$trace" || -e "$eval_audit" || -e "$retention" ]]; then
    echo "P0-A12 gate300 artifacts already exist; refusing a repeated run." >&2
    return 1
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a9_router_gate.py \
    --endpoint "$ENDPOINT" --model-id-math "p0a12-math-$math_step" \
    --model-id-code p0a12-code-250 --model-id-nlp p0a12-nlp-136 \
    --candidate-name p0a12-final-router-hf --output-trace "$trace" --audit "$eval_audit"
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$trace" --candidate-name p0a12-final-router-hf --output "$retention"
}

validation() {
  require_status reports/audit/gate_p0a12_train_math.json passed
  asset_guards
  require_checkpoint 288; require_checkpoint 576
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a12-base \
    --lora-module "p0a12-math-288=$ROOT/$MATH_DIR/checkpoint-288" \
    --lora-module "p0a12-math-576=$ROOT/$MATH_DIR/checkpoint-576" \
    --lora-module "p0a12-code-250=$ROOT/$CODE_DIR" \
    --lora-module "p0a12-nlp-136=$ROOT/$NLP_DIR" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a12_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a12-base,p0a12-math-288,p0a12-math-576,p0a12-code-250,p0a12-nlp-136"
  evaluate_once p0a12-base
  evaluate_once p0a12-math-288
  evaluate_once p0a12-math-576
  if ! select_math; then
    echo "P0-A12 Math selection failed; gate300 remains closed." >&2
    cleanup; trap - EXIT INT TERM
    return 1
  fi
  local math_step
  math_step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a12/math_selection.json"))["selected_step"])')"
  run_gate300_once "$math_step"
  local result=$?
  cleanup; trap - EXIT INT TERM
  return "$result"
}

status() {
  local path
  for path in reports/audit/gate_p0a12_data.json \
    reports/audit/gate_p0a12_math_preflight.json reports/audit/gate_p0a12_train_math.json \
    reports/audit/p0a12/math_selection.json \
    reports/audit/gate_p0a12_final_router_hf_gate300_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("ratios","")))' "$path"
    else echo "$path missing"; fi
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
    bash -n scripts/run_p0a12.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a12_data.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a11_domain.py \
      model_compression/train_p0a6_student.py
    ;;
  *) echo "Usage: bash scripts/run_p0a12.sh <data-build|preflight|train|validation|auto|status|structural-check>" ;;
esac
