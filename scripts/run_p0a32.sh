#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A32_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A32_SERVE_GPU:-0}"
TEACHER_PORT="${P0A32_TEACHER_PORT:-18492}"
VALIDATION_PORT="${P0A32_VALIDATION_PORT:-18493}"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
INIT_ADAPTER="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
OUTPUT_DIR="models/checkpoints/p0a32/nlp-continuation"
TRAIN_DATA="data/p0a32/train.jsonl"
TRAIN_AUDIT="reports/audit/gate_p0a32_train.json"
AUDIT_ROOT="reports/audit/p0a32"

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
 raise SystemExit(f"P0-A32 requires four distinct GPU ids: {ids}")
print("P0-A32 GPU group:",','.join(ids))
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
  local pid="$1" endpoint="$2" ids="$3" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$endpoint" "$ids" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 actual={str(row.get('id')) for row in json.load(response).get('data',[])}
raise SystemExit(0 if set(sys.argv[2].split(',')).issubset(actual) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

requests_build() {
  if [[ -f reports/audit/gate_p0a32_requests.json ]]; then
    require_status reports/audit/gate_p0a32_requests.json passed
    return 0
  fi
  "$PYTHON_BIN" model_compression/build_p0a32_mmlu_requests.py
  require_status reports/audit/gate_p0a32_requests.json passed
}

teacher_generate() {
  require_status reports/audit/gate_p0a32_requests.json passed
  if [[ -f reports/audit/gate_p0a32_teacher_data.json ]]; then
    require_status reports/audit/gate_p0a32_teacher_data.json passed
    return 0
  fi
  validate_gpu_group
  mkdir -p logs data/p0a32
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port "$TEACHER_PORT" \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ \
    --quantization awq --tensor-parallel-size 4 \
    --served-model-name p0a32-teacher14b --max-model-len 2048 \
    --gpu-memory-utilization 0.80 >logs/p0a32_teacher_server.log 2>&1 &
  local server_pid=$! endpoint="http://127.0.0.1:$TEACHER_PORT"
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$endpoint" p0a32-teacher14b
  "$PYTHON_BIN" model_compression/generate_p0a7_mmlu_chinese.py generate \
    --requests data/distill/p0a32_nlp_teacher_requests.jsonl \
    --trace data/distill/p0a32_nlp_teacher_trace.jsonl \
    --train-output data/p0a32/nlp_train.jsonl \
    --validation-output data/p0a32/nlp_validation.jsonl \
    --audit reports/audit/gate_p0a32_teacher_data.json \
    --train-target 4000 --validation-target 500 \
    --endpoint "$endpoint" --model-id p0a32-teacher14b \
    --fallback-model-id p0a32-teacher14b --workers 8 --retries 2 \
    --timeout-sec 180 --seed 20260802 \
    --minimum-domain-equal-quota-ratio 0.05 --minimum-domains 6
  require_status reports/audit/gate_p0a32_teacher_data.json passed
  cleanup; trap - EXIT INT TERM
}

guarded_teacher_generate() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    bash scripts/run_p0a32.sh teacher-generate
}

train_data_build() {
  require_status reports/audit/gate_p0a32_teacher_data.json passed
  "$PYTHON_BIN" model_compression/build_p0a32_train.py
  require_status reports/audit/gate_p0a32_train_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a32_train_data.json passed
  [[ -d "$BASE_DIR" && -s "$INIT_ADAPTER/adapter_model.safetensors" ]] || {
    echo "Missing P0-A32 base or initial NLP adapter" >&2; return 1;
  }
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$TRAIN_DATA" --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a32_preflight.json \
    --max-steps 256 --checkpoint-steps 128 --focus-domain nlp_mixed_mcq \
    --learning-rate 0.0000008 --lora-rank 16 --lora-alpha 32 \
    --init-adapter "$INIT_ADAPTER" --dry-run
  require_status reports/audit/gate_p0a32_preflight.json dry_run_passed
}

train() {
  validate_gpu_group
  preflight
  if [[ -f "$TRAIN_AUDIT" ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$TRAIN_AUDIT")"
    if [[ "$status" == passed ]]; then
      require_checkpoint 128; require_checkpoint 256
      echo "P0-A32 NLP continuation training already complete."
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
    --audit "$TRAIN_AUDIT" --max-steps 256 --checkpoint-steps 128 \
    --focus-domain nlp_mixed_mcq --learning-rate 0.0000008 \
    --lora-rank 16 --lora-alpha 32 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${init[@]}" "${resume[@]}"
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 128; require_checkpoint 256
}

evaluate_once() {
  local endpoint="$1" model_id="$2" label="$3" audit="$AUDIT_ROOT/$3.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a32 --domain nlp \
    --manifest data/p0a32/nlp_internal_validation.jsonl \
    --endpoint "$endpoint" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 500 --workers "${P0A32_EVAL_WORKERS:-8}" \
    --thinking off --max-tokens 256 --timeout-sec 120 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$audit"
  require_status "$audit" passed
}

validation() {
  require_status "$TRAIN_AUDIT" passed
  require_checkpoint 128; require_checkpoint 256
  mkdir -p logs "$AUDIT_ROOT"
  local endpoint="http://127.0.0.1:$VALIDATION_PORT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$VALIDATION_PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a32-base \
    --lora-module "p0a32-initial=$ROOT/$INIT_ADAPTER" \
    --lora-module "p0a32-nlp-128=$ROOT/$OUTPUT_DIR/checkpoint-128" \
    --lora-module "p0a32-nlp-256=$ROOT/$OUTPUT_DIR/checkpoint-256" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a32_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$endpoint" p0a32-base,p0a32-initial,p0a32-nlp-128,p0a32-nlp-256
  evaluate_once "$endpoint" p0a32-initial initial_nlp
  evaluate_once "$endpoint" p0a32-nlp-128 nlp_128
  evaluate_once "$endpoint" p0a32-nlp-256 nlp_256
  set +e
  "$PYTHON_BIN" scripts/select_p0a32_nlp.py \
    --initial-audit "$AUDIT_ROOT/initial_nlp.json" \
    --candidate "128=$AUDIT_ROOT/nlp_128.json" \
    --candidate "256=$AUDIT_ROOT/nlp_256.json" \
    --output "$AUDIT_ROOT/nlp_selection.json"
  local selection_rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$selection_rc"
}

guarded_validation() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    bash scripts/run_p0a32.sh validation
}

status() {
  local path
  for path in reports/audit/gate_p0a32_requests.json \
    reports/audit/gate_p0a32_teacher_data.json \
    reports/audit/gate_p0a32_train_data.json reports/audit/gate_p0a32_preflight.json \
    "$TRAIN_AUDIT" "$AUDIT_ROOT/initial_nlp.json" "$AUDIT_ROOT/nlp_128.json" \
    "$AUDIT_ROOT/nlp_256.json" "$AUDIT_ROOT/nlp_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("selected_step",d.get("accepted_count",""))))' "$path"
    else echo "$path missing"; fi
  done
  if [[ -f data/distill/p0a32_nlp_teacher_trace.jsonl ]]; then
    wc -l data/distill/p0a32_nlp_teacher_trace.jsonl
  fi
}

case "${1:-help}" in
  requests-build) requests_build ;;
  teacher-generate) teacher_generate ;;
  guarded-teacher-generate) guarded_teacher_generate ;;
  train-data-build) train_data_build ;;
  preflight) preflight ;;
  train) train ;;
  validation) validation ;;
  guarded-validation) guarded_validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a32.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a32_mmlu_requests.py \
      model_compression/build_p0a32_train.py model_compression/train_p0a6_student.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a32_nlp.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a32.sh <requests-build|guarded-teacher-generate|train-data-build|preflight|train|guarded-validation|status|structural-check>"
    ;;
esac
