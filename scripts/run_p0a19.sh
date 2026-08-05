#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A19_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A19_SERVE_GPU:-0}"
PORT="${P0A19_PORT:-18473}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
OUTPUT_DIR="models/checkpoints/p0a19/code-mixed-specialist"
TRAIN_DATA="data/p0a19/code_train.jsonl"
TRAIN_AUDIT="reports/audit/gate_p0a19_train_code.json"
AUDIT_ROOT="reports/audit/p0a19"

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
 raise SystemExit(f"P0-A19 requires four distinct GPU ids: {values}")
print('P0-A19 GPU group:',','.join(values))
PY
}

require_checkpoint() {
  local step="$1" directory="$OUTPUT_DIR/checkpoint-$1"
  [[ -f "$directory/trainer_state.json" && -f "$directory/adapter_config.json" ]] || {
    echo "Missing complete P0-A19 checkpoint-$step" >&2; return 1;
  }
  [[ -s "$directory/adapter_model.safetensors" || -s "$directory/adapter_model.bin" ]] || {
    echo "Missing adapter weights in checkpoint-$step" >&2; return 1;
  }
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a19_data.py build
  require_status reports/audit/gate_p0a19_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a19_data.json passed
  require_status reports/audit/p0a18/code_selection.json failed
  [[ -d "$BASE_DIR" ]] || { echo "Missing shared base: $BASE_DIR" >&2; return 1; }
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a19_code_preflight.json \
    --max-steps 256 --checkpoint-steps 128 --focus-domain code \
    --learning-rate 0.000001 --lora-rank 16 --lora-alpha 32 --dry-run
  require_status reports/audit/gate_p0a19_code_preflight.json dry_run_passed
}

train() {
  validate_gpu_group
  preflight
  if [[ -f "$TRAIN_AUDIT" ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$TRAIN_AUDIT")"
    if [[ "$status" == passed ]]; then
      require_checkpoint 128; require_checkpoint 256
      echo "P0-A19 Code training already complete."
      return 0
    fi
  fi
  local resume=()
  if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --audit "$TRAIN_AUDIT" --max-steps 256 --checkpoint-steps 128 \
    --focus-domain code --learning-rate 0.000001 --lora-rank 16 --lora-alpha 32 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume[@]}"
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 128; require_checkpoint 256
}

wait_endpoint() {
  local pid="$1" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "P0-A19 vLLM exited before readiness" >&2; return 1; }
    if "$PYTHON_BIN" - "$ENDPOINT" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
needed={'p0a19-base','p0a19-code-128','p0a19-code-256'}
raise SystemExit(0 if needed.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

evaluate_once() {
  local model_id="$1" label="$2" audit="$AUDIT_ROOT/$2.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a19 --domain code --manifest data/p0a19/code_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 255 --workers "${P0A19_WORKERS:-8}" \
    --thinking off --max-tokens 768 --timeout-sec 120 --code-timeout-sec 5 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$audit"
  require_status "$audit" passed
}

validation() {
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 128; require_checkpoint 256
  mkdir -p logs runtime "$AUDIT_ROOT"
  # One frozen validation prompt is 897 tokens; with the preregistered
  # 768-token generation budget the serving context must exceed 1536.
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a19-base \
    --lora-module "p0a19-code-128=$ROOT/$OUTPUT_DIR/checkpoint-128" \
    --lora-module "p0a19-code-256=$ROOT/$OUTPUT_DIR/checkpoint-256" \
    --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a19_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  evaluate_once p0a19-base base_code
  evaluate_once p0a19-code-128 code_128
  evaluate_once p0a19-code-256 code_256
  set +e
  "$PYTHON_BIN" scripts/select_p0a19_code.py \
    --base-audit "$AUDIT_ROOT/base_code.json" \
    --candidate "128=$AUDIT_ROOT/code_128.json" \
    --candidate "256=$AUDIT_ROOT/code_256.json" \
    --output "$AUDIT_ROOT/code_selection.json"
  local selection_rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$selection_rc"
}

guarded_validation() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a19.sh validation
}

status() {
  local path
  for path in reports/audit/gate_p0a19_data.json \
    reports/audit/gate_p0a19_code_preflight.json "$TRAIN_AUDIT" \
    "$AUDIT_ROOT/base_code.json" "$AUDIT_ROOT/code_128.json" \
    "$AUDIT_ROOT/code_256.json" "$AUDIT_ROOT/code_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("selected_step","")))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  validation) validation ;;
  guarded-validation) guarded_validation ;;
  repair-infra-http400) "$PYTHON_BIN" scripts/repair_p0a19_infra.py ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a19.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a19_data.py \
      model_compression/train_p0a6_student.py scripts/evaluate_p0a11_domain.py \
      scripts/select_p0a19_code.py scripts/repair_p0a19_infra.py
    ;;
  *) echo "Usage: bash scripts/run_p0a19.sh <data-build|preflight|train|validation|guarded-validation|repair-infra-http400|status|structural-check>" ;;
esac
