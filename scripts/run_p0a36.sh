#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A36_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A36_SERVE_GPU:-0}"
TEACHER_PORT="${P0A36_TEACHER_PORT:-18498}"
VALIDATION_PORT="${P0A36_VALIDATION_PORT:-18499}"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
INIT_ADAPTER="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
OUTPUT_DIR="models/checkpoints/p0a36/nlp-balanced-mcq"
TRAIN_DATA="data/p0a36/train.jsonl"
TRAIN_AUDIT="reports/audit/gate_p0a36_train.json"
AUDIT_ROOT="reports/audit/p0a36"

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
ids=[item.strip() for item in sys.argv[1].split(',') if item.strip()]
if len(ids)!=4 or len(set(ids))!=4 or not all(item.isdigit() for item in ids):
 raise SystemExit(f"P0-A36 requires four distinct GPU ids: {ids}")
print("P0-A36 GPU group:",','.join(ids))
PY
}

require_checkpoint() {
  local step="$1" path="$OUTPUT_DIR/checkpoint-$1"
  [[ -s "$path/trainer_state.json" && -s "$path/adapter_config.json" ]] || {
    echo "Missing complete checkpoint: $path" >&2; return 1;
  }
  [[ -s "$path/adapter_model.safetensors" || -s "$path/adapter_model.bin" ]] || {
    echo "Missing checkpoint weights: $path" >&2; return 1;
  }
}

wait_endpoint() {
  local pid="$1" endpoint="$2" model_id="$3" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$endpoint" "$model_id" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(row.get('id')) for row in json.load(response).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

requests_build() {
  if [[ -f reports/audit/gate_p0a36_human_requests.json ]]; then
    require_status reports/audit/gate_p0a36_human_requests.json passed
    return 0
  fi
  "$PYTHON_BIN" model_compression/build_p0a36_openqa_requests.py
  require_status reports/audit/gate_p0a36_human_requests.json passed
}

teacher_generate() {
  require_status reports/audit/gate_p0a36_human_requests.json passed
  if [[ -f reports/audit/gate_p0a36_teacher_data.json ]]; then
    require_status reports/audit/gate_p0a36_teacher_data.json passed
    return 0
  fi
  validate_gpu_group
  mkdir -p logs data/p0a36
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port "$TEACHER_PORT" \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ \
    --quantization awq --tensor-parallel-size 4 \
    --served-model-name p0a36-teacher14b --max-model-len 2048 \
    --gpu-memory-utilization 0.80 >logs/p0a36_teacher_server.log 2>&1 &
  local server_pid=$! endpoint="http://127.0.0.1:$TEACHER_PORT"
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$endpoint" p0a36-teacher14b
  "$PYTHON_BIN" model_compression/generate_p0a36_openqa_mcq.py \
    --endpoint "$endpoint" --model-id p0a36-teacher14b \
    --workers "${P0A36_TEACHER_WORKERS:-8}" --timeout-sec 180
  require_status reports/audit/gate_p0a36_teacher_data.json passed
  cleanup; trap - EXIT INT TERM
}

train_data_build() {
  require_status reports/audit/gate_p0a36_teacher_data.json passed
  "$PYTHON_BIN" model_compression/build_p0a36_train.py
  require_status reports/audit/gate_p0a36_train_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a36_train_data.json passed
  [[ -d "$BASE_DIR" && -s "$INIT_ADAPTER/adapter_model.safetensors" ]] || {
    echo "Missing P0-A36 base or P0-A10 initial adapter" >&2; return 1;
  }
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a36_preflight.json \
    --max-steps 128 --checkpoint-steps 64 --focus-domain nlp \
    --learning-rate 0.0000004 --lora-rank 16 --lora-alpha 32 \
    --init-adapter "$INIT_ADAPTER" --dry-run
  require_status reports/audit/gate_p0a36_preflight.json dry_run_passed
}

train() {
  validate_gpu_group
  preflight
  if [[ -f "$TRAIN_AUDIT" ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$TRAIN_AUDIT")"
    if [[ "$status" == passed ]]; then
      require_checkpoint 64; require_checkpoint 128
      echo "P0-A36 training already complete."
      return 0
    fi
  fi
  local resume=() init=(--init-adapter "$INIT_ADAPTER")
  if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
      -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
    init=()
  fi
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --audit "$TRAIN_AUDIT" --max-steps 128 --checkpoint-steps 64 \
    --focus-domain nlp --learning-rate 0.0000004 \
    --lora-rank 16 --lora-alpha 32 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${init[@]}" "${resume[@]}"
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 64; require_checkpoint 128
}

evaluate_once() {
  local endpoint="$1" model_id="$2" label="$3" audit="$AUDIT_ROOT/$3.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a36 --domain nlp --manifest data/p0a36/nlp_validation.jsonl \
    --endpoint "$endpoint" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 256 --workers "${P0A36_EVAL_WORKERS:-8}" \
    --thinking off --max-tokens 256 --timeout-sec 120 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$audit"
  require_status "$audit" passed
}

validation() {
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 64; require_checkpoint 128
  mkdir -p logs "$AUDIT_ROOT"
  local endpoint="http://127.0.0.1:$VALIDATION_PORT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$VALIDATION_PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a36-base \
    --lora-module "p0a36-initial=$ROOT/$INIT_ADAPTER" \
    --lora-module "p0a36-nlp-64=$ROOT/$OUTPUT_DIR/checkpoint-64" \
    --lora-module "p0a36-nlp-128=$ROOT/$OUTPUT_DIR/checkpoint-128" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a36_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$endpoint" p0a36-base
  evaluate_once "$endpoint" p0a36-initial initial_nlp
  evaluate_once "$endpoint" p0a36-nlp-64 nlp_64
  evaluate_once "$endpoint" p0a36-nlp-128 nlp_128
  set +e
  "$PYTHON_BIN" scripts/select_p0a36_nlp.py \
    --initial-audit "$AUDIT_ROOT/initial_nlp.json" \
    --candidate "64=$AUDIT_ROOT/nlp_64.json" \
    --candidate "128=$AUDIT_ROOT/nlp_128.json" \
    --output "$AUDIT_ROOT/nlp_selection.json"
  local rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$rc"
}

case "${1:-help}" in
  requests-build) requests_build ;;
  teacher-generate) teacher_generate ;;
  guarded-teacher-generate)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
      bash scripts/run_p0a36.sh teacher-generate
    ;;
  train-data-build) train_data_build ;;
  preflight) preflight ;;
  train) train ;;
  validation) validation ;;
  guarded-validation)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
      bash scripts/run_p0a36.sh validation
    ;;
  status)
    for path in reports/audit/gate_p0a36_human_requests.json \
      reports/audit/gate_p0a36_teacher_data.json \
      reports/audit/gate_p0a36_train_data.json reports/audit/gate_p0a36_preflight.json \
      "$TRAIN_AUDIT" "$AUDIT_ROOT/initial_nlp.json" "$AUDIT_ROOT/nlp_64.json" \
      "$AUDIT_ROOT/nlp_128.json" "$AUDIT_ROOT/nlp_selection.json"; do
      if [[ -f "$path" ]]; then
        "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("request_count",d.get("train_rows","")),d.get("accepted_count",""))' "$path"
      else echo "$path missing"; fi
    done
    if [[ -f data/distill/p0a36_openqa_mcq_trace.jsonl ]]; then
      wc -l data/distill/p0a36_openqa_mcq_trace.jsonl
    fi
    ;;
  structural-check)
    bash -n scripts/run_p0a36.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a36_openqa_requests.py \
      model_compression/generate_p0a36_openqa_mcq.py \
      model_compression/build_p0a36_train.py model_compression/train_p0a6_student.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a36_nlp.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a36.sh <requests-build|guarded-teacher-generate|train-data-build|preflight|train|guarded-validation|status|structural-check>"
    ;;
esac
