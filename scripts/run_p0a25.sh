#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A25_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A25_SERVE_GPU:-0}"
PORT="${P0A25_PORT:-18479}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
INIT_ADAPTER="models/checkpoints/p0a23/code-continuation/checkpoint-96"
OUTPUT_DIR="models/checkpoints/p0a25/code-failure-repair"
TRAIN_DATA="data/p0a25/code_curriculum.jsonl"
TRAIN_AUDIT="reports/audit/gate_p0a25_train_code.json"
AUDIT_ROOT="reports/audit/p0a25"

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
 raise SystemExit(f"P0-A25 requires four distinct GPU ids: {values}")
print('P0-A25 GPU group:',','.join(values))
PY
}

require_checkpoint() {
  local step="$1" directory="$OUTPUT_DIR/checkpoint-$1"
  [[ -f "$directory/trainer_state.json" && -f "$directory/adapter_config.json" ]] || {
    echo "Missing complete P0-A25 checkpoint-$step" >&2; return 1;
  }
  [[ -s "$directory/adapter_model.safetensors" || -s "$directory/adapter_model.bin" ]] || {
    echo "Missing adapter weights in checkpoint-$step" >&2; return 1;
  }
}

data_build() {
  if [[ -f reports/audit/gate_p0a25_data.json ]]; then
    require_status reports/audit/gate_p0a25_data.json passed
    return 0
  fi
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    "$PYTHON_BIN" model_compression/build_p0a25_data.py build \
    --workers "${P0A25_DATA_WORKERS:-16}"
  require_status reports/audit/gate_p0a25_data.json passed
}

wait_endpoint() {
  local pid="$1" required="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$required" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
needed=set(sys.argv[2].split(','))
raise SystemExit(0 if needed.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

mining() {
  require_status reports/audit/gate_p0a25_data.json passed
  require_status reports/audit/p0a23/code_selection.json passed
  if [[ -f reports/audit/gate_p0a25_curriculum.json ]]; then
    require_status reports/audit/gate_p0a25_curriculum.json passed
    echo "P0-A25 failure curriculum already complete."
    return 0
  fi
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a25-base \
    --lora-module "p0a25-miner=$ROOT/$INIT_ADAPTER" \
    --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a25_mining_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a25-miner
  if [[ ! -f "$AUDIT_ROOT/mining.json" ]]; then
    "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
      --protocol p0a25_mining --domain code \
      --manifest data/p0a25/code_mining_manifest.jsonl \
      --endpoint "$ENDPOINT" --model-id p0a25-miner --candidate-name p0a25-miner \
      --expected-rows 6000 --workers "${P0A25_MINING_WORKERS:-16}" \
      --thinking off --max-tokens 768 --timeout-sec 120 --code-timeout-sec 5 \
      --output-trace "$AUDIT_ROOT/mining_trace.jsonl" \
      --audit "$AUDIT_ROOT/mining.json"
  fi
  require_status "$AUDIT_ROOT/mining.json" passed
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" model_compression/build_p0a25_curriculum.py
  require_status reports/audit/gate_p0a25_curriculum.json passed
}

guarded_mining() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a25.sh mining
}

preflight() {
  require_status reports/audit/gate_p0a25_data.json passed
  require_status reports/audit/gate_p0a25_curriculum.json passed
  require_status reports/audit/gate_p0a24_code_retention.json failed
  [[ -d "$BASE_DIR" && -s "$INIT_ADAPTER/adapter_model.safetensors" ]] || {
    echo "Missing P0-A25 base or initial adapter" >&2; return 1;
  }
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a25_code_preflight.json \
    --max-steps 192 --checkpoint-steps 96 --focus-domain code \
    --learning-rate 0.0000006 --lora-rank 16 --lora-alpha 32 \
    --init-adapter "$INIT_ADAPTER" --dry-run
  require_status reports/audit/gate_p0a25_code_preflight.json dry_run_passed
}

train() {
  validate_gpu_group
  preflight
  if [[ -f "$TRAIN_AUDIT" ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$TRAIN_AUDIT")"
    if [[ "$status" == passed ]]; then
      require_checkpoint 96; require_checkpoint 192
      echo "P0-A25 failure-repair training already complete."
      return 0
    fi
  fi
  local resume=() init=(--init-adapter "$INIT_ADAPTER")
  if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
    init=()
  fi
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --audit "$TRAIN_AUDIT" --max-steps 192 --checkpoint-steps 96 \
    --focus-domain code --learning-rate 0.0000006 --lora-rank 16 --lora-alpha 32 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${init[@]}" "${resume[@]}"
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 96; require_checkpoint 192
}

evaluate_once() {
  local model_id="$1" label="$2" audit="$AUDIT_ROOT/$2.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a25 --domain code --manifest data/p0a25/code_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 1000 --workers "${P0A25_EVAL_WORKERS:-8}" \
    --thinking off --max-tokens 768 --timeout-sec 120 --code-timeout-sec 5 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$audit"
  require_status "$audit" passed
}

validation() {
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 96; require_checkpoint 192
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a25-base \
    --lora-module "p0a25-initial=$ROOT/$INIT_ADAPTER" \
    --lora-module "p0a25-code-96=$ROOT/$OUTPUT_DIR/checkpoint-96" \
    --lora-module "p0a25-code-192=$ROOT/$OUTPUT_DIR/checkpoint-192" \
    --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a25_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a25-base,p0a25-initial,p0a25-code-96,p0a25-code-192
  evaluate_once p0a25-base base_code
  evaluate_once p0a25-initial initial_code
  evaluate_once p0a25-code-96 code_96
  evaluate_once p0a25-code-192 code_192
  set +e
  "$PYTHON_BIN" scripts/select_p0a25_code.py \
    --base-audit "$AUDIT_ROOT/base_code.json" \
    --initial-audit "$AUDIT_ROOT/initial_code.json" \
    --candidate "96=$AUDIT_ROOT/code_96.json" \
    --candidate "192=$AUDIT_ROOT/code_192.json" \
    --output "$AUDIT_ROOT/code_selection.json"
  local selection_rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$selection_rc"
}

guarded_validation() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a25.sh validation
}

status() {
  local path
  for path in reports/audit/gate_p0a25_data.json \
    "$AUDIT_ROOT/mining.json" reports/audit/gate_p0a25_curriculum.json \
    reports/audit/gate_p0a25_code_preflight.json "$TRAIN_AUDIT" \
    "$AUDIT_ROOT/base_code.json" "$AUDIT_ROOT/initial_code.json" \
    "$AUDIT_ROOT/code_96.json" "$AUDIT_ROOT/code_192.json" \
    "$AUDIT_ROOT/code_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("selected_step",d.get("student_accuracy",""))))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  mining) mining ;;
  guarded-mining) guarded_mining ;;
  preflight) preflight ;;
  train) train ;;
  validation) validation ;;
  guarded-validation) guarded_validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a25.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a25_data.py \
      model_compression/build_p0a25_curriculum.py \
      model_compression/train_p0a6_student.py scripts/evaluate_p0a11_domain.py \
      scripts/select_p0a25_code.py
    ;;
  *) echo "Usage: bash scripts/run_p0a25.sh <data-build|guarded-mining|preflight|train|guarded-validation|status|structural-check>" ;;
esac
