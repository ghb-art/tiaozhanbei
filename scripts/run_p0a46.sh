#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A46_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A46_SERVE_GPU:-0}"
PORT="${P0A46_PORT:-18560}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE="models/checkpoints/p0a4/student-shared-merged"
OUTPUT="models/checkpoints/p0a46/nlp"
TRAIN_AUDIT="reports/audit/gate_p0a46_train_nlp.json"

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not p.is_file(): raise SystemExit(f"Missing audit: {p}")
status=json.loads(p.read_text(encoding='utf-8')).get('status')
if status not in allowed: raise SystemExit(f"Audit rejected: {p} status={status} allowed={sorted(allowed)}")
print(f"Audit guard passed: {p} status={status}")
PY
}

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[x.strip() for x in sys.argv[1].split(',') if x.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(x.isdigit() for x in values):
 raise SystemExit(f"P0-A46 requires four distinct GPU ids: {values}")
print("P0-A46 GPU group:", ",".join(values))
PY
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a46_nlp_data.py
  require_status reports/audit/gate_p0a46_data.json passed
}

preflight() {
  data_build
  validate_gpus
  [[ -d "$BASE" ]] || { echo "Missing base model: $BASE" >&2; return 1; }
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE" --train-data data/p0a46/nlp_train.jsonl \
    --output-dir "$OUTPUT" --audit reports/audit/gate_p0a46_train_preflight.json \
    --max-steps 318 --checkpoint-steps 159 --focus-domain nlp \
    --learning-rate 4e-7 --lora-rank 16 --lora-alpha 32 --dry-run
  require_status reports/audit/gate_p0a46_train_preflight.json dry_run_passed
}

checkpoint_ok() {
  local step="$1" path="$OUTPUT/checkpoint-$1"
  [[ -s "$path/adapter_config.json" && -s "$path/trainer_state.json" ]] && \
    [[ -s "$path/adapter_model.safetensors" || -s "$path/adapter_model.bin" ]]
}

train() {
  local resume=()
  preflight
  if [[ -f "$TRAIN_AUDIT" ]] && \
     [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$TRAIN_AUDIT")" == passed ]]; then
    checkpoint_ok 159; checkpoint_ok 318
    echo "P0-A46 NLP training already complete."
    return 0
  fi
  if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0a6_student.py \
        --model-dir "$BASE" --train-data data/p0a46/nlp_train.jsonl \
        --output-dir "$OUTPUT" --audit "$TRAIN_AUDIT" \
        --max-steps 318 --checkpoint-steps 159 --focus-domain nlp \
        --learning-rate 4e-7 --lora-rank 16 --lora-alpha 32 \
        --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
        "${resume[@]}" \
    2>&1 | tee logs/p0a46_nlp_train.log
  require_status "$TRAIN_AUDIT" passed
  checkpoint_ok 159; checkpoint_ok 318
}

wait_endpoint() {
  local pid="$1" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
need={'p0a46-base','p0a46-nlp-159','p0a46-nlp-318'}
raise SystemExit(0 if need.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

eval_one() {
  local dataset="$1" manifest="$2" model="$3" label="$4"
  local trace="reports/audit/p0a44/p0a46_${label}_trace.jsonl"
  local audit="reports/audit/p0a44/p0a46_${label}.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a44_aligned.py \
    --dataset "$dataset" --manifest "$manifest" --endpoint "$ENDPOINT" \
    --model-id "$model" --candidate-name "$model" --workers 8 \
    --timeout-sec 180 --max-tokens 16 --output-trace "$trace" --audit "$audit"
  require_status "$audit" passed
}

validate_hf() {
  require_status "$TRAIN_AUDIT" passed
  checkpoint_ok 159; checkpoint_ok 318
  mkdir -p logs reports/audit/p0a44
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a46-base \
    --lora-module "p0a46-nlp-159=$ROOT/$OUTPUT/checkpoint-159" \
    --lora-module "p0a46-nlp-318=$ROOT/$OUTPUT/checkpoint-318" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a46_hf_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() { if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  local dataset manifest model step
  for step in base 159 318; do
    model="p0a46-nlp-$step"; [[ "$step" == base ]] && model=p0a46-base
    for dataset in ceval cmmlu; do
      manifest="data/p0a44/nlp_${dataset}_dev.jsonl"
      eval_one "$dataset" "$manifest" "$model" "hf_${step}_${dataset}"
    done
  done
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/select_p0a46_nlp.py
}

status() {
  local path
  for path in reports/audit/gate_p0a46_data.json \
    reports/audit/gate_p0a46_train_preflight.json "$TRAIN_AUDIT" \
    reports/audit/gate_p0a46_hf_selection.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[1],d.get("status"),d.get("selected",d.get("global_step","")))' "$path"
    else echo "$path missing"; fi
  done
}

structural_check() {
  bash -n scripts/run_p0a46.sh
  "$PYTHON_BIN" -m py_compile model_compression/build_p0a46_nlp_data.py \
    scripts/select_p0a46_nlp.py
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  validate-hf) validate_hf ;;
  all) train; validate_hf ;;
  status) status ;;
  structural-check) structural_check ;;
  *) echo "Usage: bash scripts/run_p0a46.sh <data-build|preflight|train|validate-hf|all|status|structural-check>" ;;
esac
