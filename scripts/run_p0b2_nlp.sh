#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0B2_NLP_GPUS:-0,1,2,3}"
BASE="models/checkpoints/p0a4/student-shared-merged"
STAGE1_CONFIG="configs/p0b2_nlp_stage1.json"
STAGE2_CONFIG="configs/p0b2_nlp_stage2.json"
STAGE1_OUT="models/checkpoints/p0b2/nlp-stage1"
STAGE2_OUT="models/checkpoints/p0b2/nlp-stage2"
DATA_AUDIT="reports/audit/gate_p0b2_nlp_data.json"
STAGE1_AUDIT="reports/audit/gate_p0b2_nlp_stage1_train.json"
STAGE2_AUDIT="reports/audit/gate_p0b2_nlp_stage2_train.json"
MERGE_AUDIT="reports/audit/gate_p0b2_nlp_merge.json"

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
 raise SystemExit(f'P0-B2 NLP requires four distinct GPU ids: {values}')
print('P0-B2 NLP GPU group:', ','.join(values))
PY
}

latest_checkpoint() {
  local output="$1" checkpoint
  [[ -d "$output" ]] || return 0
  while IFS= read -r checkpoint; do
    if [[ -s "$checkpoint/trainer_state.json" \
       && -s "$checkpoint/adapter_config.json" \
       && -s "$checkpoint/adapter_model.safetensors" \
       && -s "$checkpoint/optimizer.pt" ]]; then
      printf '%s\n' "$checkpoint"
    fi
  done < <(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' | sort -V)
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0b2_nlp_data.py
  require_status "$DATA_AUDIT" passed
}

stage1_train() {
  local checkpoint="" resume=()
  data_build
  validate_gpus
  [[ -d "$BASE" ]] || { echo "Missing base model: $BASE" >&2; return 1; }
  if [[ -f "$STAGE1_AUDIT" ]] && \
     [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$STAGE1_AUDIT")" == passed ]]; then
    echo "P0-B2 NLP stage 1 already complete."
    return 0
  fi
  checkpoint="$(latest_checkpoint "$STAGE1_OUT" | tail -n 1)"
  if [[ -n "$checkpoint" ]]; then
    resume=(--resume-from-checkpoint "$checkpoint")
    echo "Resuming stage 1 from $checkpoint"
  fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0a5_lora.py \
      --config "$STAGE1_CONFIG" --role student --candidate-index 1 \
      --model-dir "$BASE" \
      --train-data data/p0b2/nlp_stage1.jsonl \
      --validation-data data/p0b1/internal_validation.jsonl \
      --output-dir "$STAGE1_OUT" --audit "$STAGE1_AUDIT" \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
      "${resume[@]}" \
    2>&1 | tee -a logs/p0b2_nlp_stage1_train.log
  require_status "$STAGE1_AUDIT" passed
}

stage2_train() {
  local checkpoint="" resume=()
  require_status "$STAGE1_AUDIT" passed
  [[ -d "$BASE" ]] || { echo "Missing base model: $BASE" >&2; return 1; }
  if [[ -f "$STAGE2_AUDIT" ]] && \
     [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$STAGE2_AUDIT")" == passed ]]; then
    echo "P0-B2 NLP stage 2 already complete."
    return 0
  fi
  checkpoint="$(latest_checkpoint "$STAGE1_OUT" | tail -n 1)"
  if [[ -z "$checkpoint" ]]; then
    echo "No stage 1 checkpoint found." >&2
    return 2
  fi
  resume=(--resume-from-checkpoint "$checkpoint")
  echo "Starting stage 2 (math replay) from $checkpoint"
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0a5_lora.py \
      --config "$STAGE2_CONFIG" --role student --candidate-index 1 \
      --model-dir "$BASE" \
      --train-data data/p0b2/math_stage2.jsonl \
      --validation-data data/p0b1/internal_validation.jsonl \
      --output-dir "$STAGE2_OUT" --audit "$STAGE2_AUDIT" \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
      "${resume[@]}" \
    2>&1 | tee -a logs/p0b2_nlp_stage2_train.log
  require_status "$STAGE2_AUDIT" passed
}

merge() {
  local checkpoint
  require_status "$STAGE2_AUDIT" passed
  if [[ -f "$MERGE_AUDIT" && -d models/checkpoints/p0b2/nlp-staged-merged ]]; then
    require_status "$MERGE_AUDIT" passed
    return 0
  fi
  checkpoint="$(latest_checkpoint "$STAGE2_OUT" | tail -n 1)"
  [[ -n "$checkpoint" ]] || { echo "No stage 2 checkpoint found." >&2; return 2; }
  "$PYTHON_BIN" model_compression/merge_lora_adapter.py \
    --base-model "$BASE" --adapter "$checkpoint" \
    --output models/checkpoints/p0b2/nlp-staged-merged --audit "$MERGE_AUDIT"
  require_status "$MERGE_AUDIT" passed
}

status() {
  echo "P0-B2 NLP staged status:"
  for audit in "$DATA_AUDIT" "$STAGE1_AUDIT" "$STAGE2_AUDIT" "$MERGE_AUDIT"; do
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
  stage1) stage1_train ;;
  stage2) stage2_train ;;
  merge) merge ;;
  status) status ;;
  *)
    echo "Usage: $0 {data-build|stage1|stage2|merge|status}" >&2
    return 1
    ;;
esac
