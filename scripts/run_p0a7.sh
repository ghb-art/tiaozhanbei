#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A7_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A7_SERVE_GPU:-0}"
PORT="${P0A7_PORT:-18461}"
ENDPOINT="http://127.0.0.1:$PORT"
OUTPUT_DIR="models/checkpoints/p0a7/nlp-mmlu-aux-specialist"
AUDIT_ROOT="reports/audit/p0a7"

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
    raise SystemExit(f"P0-A7 requires four distinct GPU ids: {values}")
print("P0-A7 GPU group:", ','.join(values))
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

source_verify() {
  require_file data/datasets/mmlu/data.tar
  local actual expected
  actual="$(sha256sum data/datasets/mmlu/data.tar | awk '{print $1}')"
  expected="bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b"
  [[ "$actual" == "$expected" ]] || { echo "MMLU archive hash mismatch" >&2; return 1; }
  require_dir data/datasets/mmlu/data/auxiliary_train
  local count
  count="$(find data/datasets/mmlu/data/auxiliary_train -maxdepth 1 -type f -name '*.csv' | wc -l)"
  [[ "$count" == 8 ]] || { echo "Expected 8 MMLU auxiliary domains, found $count" >&2; return 1; }
  echo "P0-A7 source guard passed: SHA-256=$actual auxiliary_domains=$count"
}

teacher_prepare() {
  source_verify
  "$PYTHON_BIN" model_compression/generate_p0a7_mmlu_chinese.py prepare \
    --train-target 3000 --validation-target 256
}

teacher_generate() {
  validate_gpu_group
  source_verify
  require_file data/distill/p0a7_nlp_teacher_requests.jsonl
  require_status reports/audit/gate_p0a5_train_teacher.json passed
  require_dir models/pretrained/Qwen--Qwen2.5-14B-Instruct
  require_dir models/checkpoints/p0a5/teacher
  mkdir -p logs runtime
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port 8000 \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct \
    --quantization none --tensor-parallel-size 4 \
    --served-model-name p0a5-teacher-base \
    --lora-module "p0a5-teacher=$ROOT/models/checkpoints/p0a5/teacher" \
    --max-model-len 4096 --gpu-memory-utilization 0.85 \
    >logs/p0a7_mmlu_teacher_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a7_mmlu_teacher_server.pid
  cleanup_teacher() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_teacher EXIT INT TERM
  local ready=0 attempt
  for attempt in $(seq 1 240); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Teacher service exited; see logs/p0a7_mmlu_teacher_server.log" >&2
      return 1
    fi
    if "$PYTHON_BIN" - >/dev/null 2>&1 <<'PY'
import json
from urllib.request import urlopen
with urlopen('http://127.0.0.1:8000/v1/models', timeout=2) as response:
    ids={str(x.get('id')) for x in json.load(response).get('data', [])}
raise SystemExit(0 if 'p0a5-teacher' in ids else 1)
PY
    then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" == 1 ]] || { echo "Teacher readiness timeout" >&2; return 1; }
  "$PYTHON_BIN" model_compression/generate_p0a7_mmlu_chinese.py generate \
    --endpoint http://127.0.0.1:8000 --model-id p0a5-teacher \
    --fallback-model-id auto --train-target 3000 --validation-target 256 \
    --minimum-domain-equal-quota-ratio 0.8 \
    --workers "${P0A7_TEACHER_WORKERS:-16}" --retries 3 --timeout-sec 120
  require_status reports/audit/gate_p0a7_nlp_data.json passed
  cleanup_teacher
  trap - EXIT INT TERM
}

data_build() {
  require_status reports/audit/gate_p0a7_nlp_data.json passed
  "$PYTHON_BIN" model_compression/build_p0a7_mmlu_specialist_data.py
  require_status reports/audit/gate_p0a7_mmlu_specialist_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a7_mmlu_specialist_data.json passed
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a7/nlp_mmlu_aux_train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a7_nlp_preflight.json \
    --max-steps 188 --checkpoint-steps 94 \
    --focus-domain nlp_mmlu_aux --learning-rate 0.00003 \
    --mcq-answer-token-weight-multiplier 1 --dry-run
  require_status reports/audit/gate_p0a7_nlp_preflight.json dry_run_passed
}

train() {
  validate_gpu_group
  preflight
  if [[ -f reports/audit/gate_p0a7_train_nlp.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a7_train_nlp.json")).get("status", ""))')"
    if [[ "$status" == passed ]]; then
      require_checkpoint 94; require_checkpoint 188
      echo "P0-A7 NLP training is already complete."
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
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a7/nlp_mmlu_aux_train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a7_train_nlp.json \
    --max-steps 188 --checkpoint-steps 94 \
    --focus-domain nlp_mmlu_aux --learning-rate 0.00003 \
    --mcq-answer-token-weight-multiplier 1 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume[@]}"
  require_status reports/audit/gate_p0a7_train_nlp.json passed
  require_checkpoint 94; require_checkpoint 188
}

validation() {
  require_checkpoint 94; require_checkpoint 188
  require_file data/p0a7/nlp_mmlu_aux_validation.jsonl
  require_file models/checkpoints/p0a6/student-pilot/checkpoint-200/adapter_config.json
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a7-base \
    --lora-module "p0a6-step-200=$ROOT/models/checkpoints/p0a6/student-pilot/checkpoint-200" \
    --lora-module "p0a7-nlp-94=$ROOT/$OUTPUT_DIR/checkpoint-94" \
    --lora-module "p0a7-nlp-188=$ROOT/$OUTPUT_DIR/checkpoint-188" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a7_validation_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a7_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_validation EXIT INT TERM
  wait_endpoint "$server_pid" "p0a7-base,p0a6-step-200,p0a7-nlp-94,p0a7-nlp-188" \
    logs/p0a7_validation_server.log
  local model name
  for model in p0a7-base p0a7-nlp-94 p0a7-nlp-188; do
    name="${model//-/_}"
    if [[ -f "$AUDIT_ROOT/${name}.json" ]]; then
      require_status "$AUDIT_ROOT/${name}.json" passed
    else
      "$PYTHON_BIN" scripts/evaluate_p0a7_nlp.py \
        --manifest data/p0a7/nlp_mmlu_aux_validation.jsonl \
        --endpoint "$ENDPOINT" --model-id "$model" --candidate-name "$model" \
        --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" \
        --audit "$AUDIT_ROOT/${name}.json"
    fi
  done
  "$PYTHON_BIN" scripts/select_p0a7_nlp.py \
    --base-audit "$AUDIT_ROOT/p0a7_base.json" \
    --candidate "94=$AUDIT_ROOT/p0a7_nlp_94.json" \
    --candidate "188=$AUDIT_ROOT/p0a7_nlp_188.json" \
    --minimum-gain 0.03 --output "$AUDIT_ROOT/checkpoint_selection.json"
  local step
  step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a7/checkpoint_selection.json"))["selected_step"])')"
  require_status reports/audit/p0a6/base_quick.json passed
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/quick_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id p0a7-base --model-id-math p0a7-base \
    --model-id-code p0a6-step-200 --model-id-nlp "p0a7-nlp-$step" \
    --candidate-name "p0a7-router-step-$step-quick" \
    --output-trace reports/audit/p0a6/p0a7_router_quick_trace.jsonl \
    --audit reports/audit/p0a6/p0a7_router_quick.json
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit reports/audit/p0a6/base_quick.json \
    --candidate "$step=reports/audit/p0a6/p0a7_router_quick.json" \
    --output reports/audit/p0a6/p0a7_quick_selection.json
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id p0a7-base --candidate-name p0a7-base-full \
    --output-trace reports/audit/p0a6/p0a7_base_full_trace.jsonl \
    --audit reports/audit/p0a6/p0a7_base_full.json
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id p0a7-base --model-id-math p0a7-base \
    --model-id-code p0a6-step-200 --model-id-nlp "p0a7-nlp-$step" \
    --candidate-name "p0a7-router-step-$step-full" \
    --output-trace reports/audit/p0a6/p0a7_router_full_trace.jsonl \
    --audit reports/audit/p0a6/p0a7_router_full.json
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --validation-manifest data/p0a6/full_validation.jsonl \
    --base-audit reports/audit/p0a6/p0a7_base_full.json \
    --candidate "$step=reports/audit/p0a6/p0a7_router_full.json" \
    --output reports/audit/p0a6/p0a7_full_selection.json
  cleanup_validation
  trap - EXIT INT TERM
}

status() {
  local path
  for path in reports/audit/gate_p0a7_nlp_data.json \
    reports/audit/gate_p0a7_mmlu_specialist_data.json \
    reports/audit/gate_p0a7_nlp_preflight.json \
    reports/audit/gate_p0a7_train_nlp.json \
    reports/audit/p0a7/checkpoint_selection.json \
    reports/audit/p0a6/p0a7_quick_selection.json \
    reports/audit/p0a6/p0a7_full_selection.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"))' "$path"
    else
      echo "$path missing"
    fi
  done
}

case "${1:-help}" in
  source-verify) source_verify ;;
  teacher-prepare) teacher_prepare ;;
  teacher-generate) teacher_generate ;;
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  validation) validation ;;
  auto) source_verify; teacher_prepare; teacher_generate; data_build; preflight; train; validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a7.sh
    "$PYTHON_BIN" -m py_compile \
      model_compression/generate_p0a7_mmlu_chinese.py \
      model_compression/build_p0a7_mmlu_specialist_data.py \
      scripts/evaluate_p0a7_nlp.py scripts/select_p0a7_nlp.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a7.sh <command>"
    echo "Commands: source-verify | teacher-prepare | teacher-generate | data-build"
    echo "          preflight | train | validation | auto | status | structural-check"
    ;;
esac
