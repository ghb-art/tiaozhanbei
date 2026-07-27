#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A4R4_GPUS:-0,1,2,3}"
EVAL_GPU="${P0A4R4_EVAL_GPU:-0}"
TEACHER_ENDPOINT="${P0A4R4_TEACHER_ENDPOINT:-http://127.0.0.1:8000}"
TEACHER_MODEL_ID="${P0A4R4_TEACHER_MODEL_ID:-distill-teacher-v1}"
TEACHER_FALLBACK_MODEL_ID="${P0A4R4_TEACHER_FALLBACK_MODEL_ID:-auto}"
NLP_WORKERS="${P0A4R4_NLP_WORKERS:-16}"
CONFIG="configs/p0a4r4_long_code_distillation.json"
BASE_MODEL="models/checkpoints/p0a4/student-shared-merged"
TRAIN_DATA="data/distill/p0a4r4_shared_train.jsonl"
VALIDATION_DATA="data/distill/p0a4r4_train_only_validation.jsonl"
DATA_AUDIT="reports/audit/gate_p0a4r4_shared_data.json"
SELECTION_AUDIT="reports/audit/gate_p0a4r4_candidate_selection.json"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; return 1; }
}

require_status() {
  "$PYTHON_BIN" -c '
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"Missing audit: {p}")
d=json.loads(p.read_text(encoding="utf-8"))
if d.get("status") not in sys.argv[2].split(","):
    raise SystemExit("Audit status rejected: {} status={}".format(p, d.get("status")))
print("Audit guard passed: {} status={}".format(p, d.get("status")))
' "$1" "$2"
}

candidate_output() {
  echo "models/checkpoints/p0a4r4/candidate-$1"
}

candidate_merged() {
  echo "models/checkpoints/p0a4r4/candidate-$1-merged"
}

candidate_train_audit() {
  echo "reports/audit/gate_p0a4r4_train_candidate_$1.json"
}

candidate_merge_audit() {
  echo "reports/audit/gate_p0a4r4_merge_candidate_$1.json"
}

candidate_eval_audit() {
  echo "reports/audit/gate_p0a4r4_candidate_$1_train_only.json"
}

candidate_eval_trace() {
  echo "data/eval/p0a4r4_candidate_$1_train_only.jsonl"
}

apps_fresh_validation() {
  "$PYTHON_BIN" model_compression/rebuild_p0a4r_apps_data.py build \
    --output data/distill/p0a4r4_apps_validation.jsonl \
    --audit reports/audit/gate_p0a4r4_apps_validation.json \
    --gate-name P0-A4R4-APPS-FRESH-VALIDATION \
    --cache runtime/p0a4r_apps_validation.jsonl \
    --tokenizer-dir "$BASE_MODEL" \
    --exclude-jsonl data/distill/p0a4r3_apps_verified_train.jsonl \
    --target-unique 128 --minimum-unique 128 \
    --max-sequence-tokens 1536 --workers 12 --seed 20260728
}

code_fresh_validation() {
  "$PYTHON_BIN" model_compression/build_p0a4r3_code_data.py build \
    --train-output runtime/p0a4r4_empty_code_train.jsonl \
    --validation-output data/distill/p0a4r4_code_contests_validation.jsonl \
    --cache runtime/p0a4r4_code_contests_compact_cache.jsonl \
    --audit reports/audit/gate_p0a4r4_code_contests_validation.json \
    --gate-name P0-A4R4-CODE-CONTESTS-FRESH-VALIDATION \
    --exclude-jsonl data/distill/p0a4r3_code_contests_train.jsonl \
    --exclude-jsonl data/distill/p0a4r3_code_contests_validation.jsonl \
    --train-target 0 --validation-target 128 \
    --max-solutions 8 --solution-selection shortest_tokens \
    --max-answer-tokens 768 --max-sequence-tokens 1536 \
    --workers 12 --seed 20260728
}

code_compact_train() {
  require_file data/distill/p0a4r4_code_contests_validation.jsonl
  "$PYTHON_BIN" model_compression/build_p0a4r3_code_data.py build \
    --train-output data/distill/p0a4r4_code_contests_compact_train.jsonl \
    --validation-output runtime/p0a4r4_empty_code_validation.jsonl \
    --cache runtime/p0a4r4_code_contests_compact_cache.jsonl \
    --audit reports/audit/gate_p0a4r4_code_contests_compact_train.json \
    --gate-name P0-A4R4-CODE-CONTESTS-COMPACT-TRAIN \
    --exclude-jsonl data/distill/p0a4r3_code_contests_validation.jsonl \
    --exclude-jsonl data/distill/p0a4r4_code_contests_validation.jsonl \
    --train-target 1000 --validation-target 0 \
    --max-solutions 8 --solution-selection shortest_tokens \
    --max-answer-tokens 768 --max-sequence-tokens 1536 \
    --workers 12 --seed 20260729
}

nlp_prepare() {
  "$PYTHON_BIN" model_compression/generate_p0a4r3_nlp_data.py prepare \
    --requests data/distill/p0a4r4_nlp_teacher_requests.jsonl \
    --trace data/distill/p0a4r4_nlp_teacher_trace.jsonl \
    --train-output runtime/p0a4r4_empty_nlp_train.jsonl \
    --validation-output data/distill/p0a4r4_nlp_verified_validation.jsonl \
    --audit reports/audit/gate_p0a4r4_nlp_validation.json \
    --exclude-jsonl data/distill/p0a4r3_nlp_verified_train.jsonl \
    --exclude-jsonl data/distill/p0a4r3_nlp_verified_validation.jsonl \
    --route-prefix p0a4r4 --train-target 0 --validation-target 256 \
    --oversample-factor 2.0 --seed 20260728
}

nlp_generate() {
  "$PYTHON_BIN" model_compression/generate_p0a4r3_nlp_data.py generate \
    --requests data/distill/p0a4r4_nlp_teacher_requests.jsonl \
    --trace data/distill/p0a4r4_nlp_teacher_trace.jsonl \
    --train-output runtime/p0a4r4_empty_nlp_train.jsonl \
    --validation-output data/distill/p0a4r4_nlp_verified_validation.jsonl \
    --audit reports/audit/gate_p0a4r4_nlp_validation.json \
    --route-prefix p0a4r4 \
    --endpoint "$TEACHER_ENDPOINT" --model-id "$TEACHER_MODEL_ID" \
    --fallback-model-id "$TEACHER_FALLBACK_MODEL_ID" \
    --train-target 0 --validation-target 256 \
    --minimum-domain-equal-quota-ratio 0.8 \
    --workers "$NLP_WORKERS" --retries 3 --seed 20260728
}

assemble() {
  "$PYTHON_BIN" model_compression/assemble_p0a4r4_shared_data.py \
    --config "$CONFIG" --output "$TRAIN_DATA" \
    --validation-output "$VALIDATION_DATA" --audit "$DATA_AUDIT"
}

preflight() {
  require_status "$DATA_AUDIT" passed
  require_dir "$BASE_MODEL"
  require_file "$TRAIN_DATA"
  require_file "$VALIDATION_DATA"
  local candidate
  for candidate in 1 2; do
    "$PYTHON_BIN" model_compression/train_p0a4_lora.py \
      --config "$CONFIG" --role student_shared \
      --candidate-index "$candidate" --model-dir "$BASE_MODEL" \
      --train-data "$TRAIN_DATA" --validation-data "$VALIDATION_DATA" \
      --output-dir "$(candidate_output "$candidate")" \
      --audit "$(candidate_train_audit "$candidate")" \
      --sample-weight-key training_weight --dry-run
  done
}

train_candidate() {
  local candidate="${1:?candidate index 1 or 2 is required}"
  [[ "$candidate" == "1" || "$candidate" == "2" ]] || {
    echo "Candidate index must be 1 or 2" >&2
    return 2
  }
  require_status "$DATA_AUDIT" passed
  local output
  output="$(candidate_output "$candidate")"
  [[ ! -e "$output" ]] || {
    echo "Refusing to overwrite candidate output: $output" >&2
    return 2
  }
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a4_lora.py \
    --config "$CONFIG" --role student_shared \
    --candidate-index "$candidate" --model-dir "$BASE_MODEL" \
    --train-data "$TRAIN_DATA" --validation-data "$VALIDATION_DATA" \
    --output-dir "$output" --audit "$(candidate_train_audit "$candidate")" \
    --sample-weight-key training_weight \
    --gradient-accumulation-steps 8
}

merge_candidate() {
  local candidate="${1:?candidate index 1 or 2 is required}"
  local output merged
  output="$(candidate_output "$candidate")"
  merged="$(candidate_merged "$candidate")"
  require_status "$(candidate_train_audit "$candidate")" passed
  [[ ! -e "$merged" ]] || {
    echo "Refusing to overwrite merged candidate: $merged" >&2
    return 2
  }
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" \
    model_compression/merge_p0a4_adapter.py \
    --base-model "$BASE_MODEL" --adapter "$output" --output "$merged" \
    --audit "$(candidate_merge_audit "$candidate")"
}

evaluate_model() {
  local name="$1" model="$2" trace="$3" audit="$4"
  require_dir "$model"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON_BIN" \
    scripts/evaluate_edge_candidate_dev.py \
    --local-model-dir "$model" --candidate-name "$name" \
    --validation-data "$VALIDATION_DATA" \
    --output-trace "$trace" --audit "$audit" \
    --device cuda:0 --dtype bfloat16 --disable-thinking \
    --max-input-length 1536 \
    --max-new-tokens-map gsm8k=512 \
    --max-new-tokens-map humaneval=768 \
    --max-new-tokens-map cmmlu=256 \
    --code-timeout-sec 4
}

evaluate_base() {
  require_status "$DATA_AUDIT" passed
  evaluate_model p0a4-v1-p0a4r4-train-only "$BASE_MODEL" \
    data/eval/p0a4r4_v1_train_only.jsonl \
    reports/audit/gate_p0a4r4_v1_train_only.json
}

evaluate_candidate() {
  local candidate="${1:?candidate index 1 or 2 is required}"
  evaluate_model "p0a4r4-candidate-$candidate" \
    "$(candidate_merged "$candidate")" \
    "$(candidate_eval_trace "$candidate")" \
    "$(candidate_eval_audit "$candidate")"
}

select_candidate() {
  local arguments=()
  local candidate
  for candidate in 1 2; do
    if [[ -f "$(candidate_eval_audit "$candidate")" && -d "$(candidate_merged "$candidate")" ]]; then
      arguments+=(--candidate "$candidate:$(candidate_eval_audit "$candidate"):$(candidate_merged "$candidate")")
    fi
  done
  [[ "${#arguments[@]}" -gt 0 ]] || {
    echo "No evaluated preregistered candidates are available" >&2
    return 1
  }
  "$PYTHON_BIN" scripts/select_p0a4r3_candidate.py \
    --config "$CONFIG" \
    --baseline-audit reports/audit/gate_p0a4r4_v1_train_only.json \
    "${arguments[@]}" --output "$SELECTION_AUDIT"
}

status() {
  local path
  for path in \
    reports/audit/gate_p0a4r4_apps_validation.json \
    reports/audit/gate_p0a4r4_code_contests_validation.json \
    reports/audit/gate_p0a4r4_code_contests_compact_train.json \
    reports/audit/gate_p0a4r4_nlp_validation.json \
    "$DATA_AUDIT" "$SELECTION_AUDIT"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"))' \
        "$path"
    else
      echo "$path missing"
    fi
  done
}

case "${1:-help}" in
  apps-fresh-validation) apps_fresh_validation ;;
  code-fresh-validation) code_fresh_validation ;;
  code-compact-train) code_compact_train ;;
  nlp-prepare) nlp_prepare ;;
  nlp-generate) nlp_generate ;;
  assemble) assemble ;;
  preflight) preflight ;;
  evaluate-base) evaluate_base ;;
  train) train_candidate "${2:-}" ;;
  merge) merge_candidate "${2:-}" ;;
  evaluate) evaluate_candidate "${2:-}" ;;
  select) select_candidate ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a4r4.sh
    "$PYTHON_BIN" -m py_compile \
      model_compression/build_p0a4r3_code_data.py \
      model_compression/rebuild_p0a4r_apps_data.py \
      model_compression/generate_p0a4r3_nlp_data.py \
      model_compression/assemble_p0a4r4_shared_data.py \
      model_compression/train_p0a4_lora.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a4r4.sh <command>"
    echo "Commands:"
    echo "  apps-fresh-validation | code-fresh-validation | code-compact-train"
    echo "  nlp-prepare | nlp-generate | assemble | preflight"
    echo "  evaluate-base | train 1|2 | merge 1|2 | evaluate 1|2 | select"
    echo "  status | structural-check"
    ;;
esac
