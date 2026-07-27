#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A4R3_GPUS:-0,1,2,3}"
EVAL_GPU="${P0A4R3_EVAL_GPU:-0}"
TEACHER_ENDPOINT="${P0A4R3_TEACHER_ENDPOINT:-http://127.0.0.1:8000}"
TEACHER_MODEL_ID="${P0A4R3_TEACHER_MODEL_ID:-distill-teacher-v1}"
TEACHER_FALLBACK_MODEL_ID="${P0A4R3_TEACHER_FALLBACK_MODEL_ID:-auto}"
NLP_WORKERS="${P0A4R3_NLP_WORKERS:-16}"
CONFIG="configs/p0a4r3_shared_distillation.json"
BASE_MODEL="models/checkpoints/p0a4/student-shared-merged"
TRAIN_DATA="data/distill/p0a4r3_shared_train.jsonl"
VALIDATION_DATA="data/distill/p0a4r3_train_only_validation.jsonl"
DATA_AUDIT="reports/audit/gate_p0a4r3_shared_data.json"
PROTOCOL_AUDIT="reports/audit/gate_p0a4r3_protocol.json"
SELECTION_AUDIT="reports/audit/gate_p0a4r3_candidate_selection.json"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-external/llama.cpp}"

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
  echo "models/checkpoints/p0a4r3/candidate-$1"
}

candidate_merged() {
  echo "models/checkpoints/p0a4r3/candidate-$1-merged"
}

candidate_train_audit() {
  echo "reports/audit/gate_p0a4r3_train_candidate_$1.json"
}

candidate_merge_audit() {
  echo "reports/audit/gate_p0a4r3_merge_candidate_$1.json"
}

candidate_eval_audit() {
  echo "reports/audit/gate_p0a4r3_candidate_$1_train_only.json"
}

candidate_eval_trace() {
  echo "data/eval/p0a4r3_candidate_$1_train_only.jsonl"
}

protocol() {
  "$PYTHON_BIN" scripts/p0a4r3_protocol.py --config "$CONFIG" --output "$PROTOCOL_AUDIT"
}

apps_expand() {
  "$PYTHON_BIN" model_compression/rebuild_p0a4r_apps_data.py build \
    --output data/distill/p0a4r3_apps_verified_train.jsonl \
    --audit reports/audit/gate_p0a4r3_apps_source.json \
    --cache runtime/p0a4r_apps_validation.jsonl \
    --tokenizer-dir "$BASE_MODEL" \
    --target-unique 2500 --minimum-unique 2400 \
    --max-sequence-tokens 1536 --workers 12
}

code_download() {
  "$PYTHON_BIN" model_compression/build_p0a4r3_code_data.py download
}

code_build() {
  "$PYTHON_BIN" model_compression/build_p0a4r3_code_data.py build \
    --train-target 1000 --validation-target 256 --workers 12
}

nlp_prepare() {
  "$PYTHON_BIN" model_compression/generate_p0a4r3_nlp_data.py prepare \
    --train-target 3000 --validation-target 256
}

nlp_generate() {
  "$PYTHON_BIN" model_compression/generate_p0a4r3_nlp_data.py generate \
    --endpoint "$TEACHER_ENDPOINT" --model-id "$TEACHER_MODEL_ID" \
    --fallback-model-id "$TEACHER_FALLBACK_MODEL_ID" \
    --train-target 3000 --validation-target 256 \
    --minimum-domain-equal-quota-ratio 0.8 \
    --workers "$NLP_WORKERS" --retries 3
}

assemble() {
  "$PYTHON_BIN" model_compression/assemble_p0a4r3_shared_data.py \
    --config "$CONFIG" --output "$TRAIN_DATA" \
    --validation-output "$VALIDATION_DATA" --audit "$DATA_AUDIT"
}

preflight() {
  protocol
  require_status "$PROTOCOL_AUDIT" passed
  require_status "$DATA_AUDIT" passed
  require_dir "$BASE_MODEL"
  require_file "$TRAIN_DATA"
  require_file "$VALIDATION_DATA"
  "$PYTHON_BIN" model_compression/train_p0a4_lora.py \
    --config "$CONFIG" --role student_shared --candidate-index 1 \
    --model-dir "$BASE_MODEL" --train-data "$TRAIN_DATA" \
    --validation-data "$VALIDATION_DATA" \
    --output-dir "$(candidate_output 1)" --audit "$(candidate_train_audit 1)" \
    --dry-run
  "$PYTHON_BIN" model_compression/train_p0a4_lora.py \
    --config "$CONFIG" --role student_shared --candidate-index 2 \
    --model-dir "$BASE_MODEL" --train-data "$TRAIN_DATA" \
    --validation-data "$VALIDATION_DATA" \
    --output-dir "$(candidate_output 2)" --audit "$(candidate_train_audit 2)" \
    --dry-run
}

train_candidate() {
  local candidate="${1:?candidate index 1 or 2 is required}"
  [[ "$candidate" == "1" || "$candidate" == "2" ]] || {
    echo "Candidate index must be 1 or 2" >&2
    return 2
  }
  require_status "$PROTOCOL_AUDIT" passed
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
    --config "$CONFIG" --role student_shared --candidate-index "$candidate" \
    --model-dir "$BASE_MODEL" --train-data "$TRAIN_DATA" \
    --validation-data "$VALIDATION_DATA" \
    --output-dir "$output" --audit "$(candidate_train_audit "$candidate")" \
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
    --max-new-tokens-map humaneval=1024 \
    --max-new-tokens-map cmmlu=256 \
    --code-timeout-sec 4
}

evaluate_base() {
  require_status "$DATA_AUDIT" passed
  evaluate_model p0a4-v1-train-only "$BASE_MODEL" \
    data/eval/p0a4r3_v1_train_only.jsonl \
    reports/audit/gate_p0a4r3_v1_train_only.json
}

evaluate_candidate() {
  local candidate="${1:?candidate index 1 or 2 is required}"
  evaluate_model "p0a4r3-candidate-$candidate" "$(candidate_merged "$candidate")" \
    "$(candidate_eval_trace "$candidate")" "$(candidate_eval_audit "$candidate")"
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
    --baseline-audit reports/audit/gate_p0a4r3_v1_train_only.json \
    "${arguments[@]}" --output "$SELECTION_AUDIT"
}

selected_model() {
  "$PYTHON_BIN" -c '
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding="utf-8"))
if d.get("status") != "passed" or not d.get("selected_model"):
    raise SystemExit("No selected P0-A4R3 model")
print(d["selected_model"])
' "$SELECTION_AUDIT"
}

quantize_selected() {
  require_status "$SELECTION_AUDIT" passed
  require_dir "$LLAMA_CPP_DIR"
  local model
  model="$(selected_model)"
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source "$TRAIN_DATA" \
    --output data/distill/p0a4r3_imatrix_calibration.txt \
    --audit reports/audit/gate_p0a4r3_imatrix_calibration.json \
    --stratify-key dataset_key \
    --stratum gsm8k --stratum humaneval --stratum cmmlu \
    --rows-per-stratum 128 --seed 20260727
  "$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$model" \
    --outfile models/quantized/p0a4r3-selected-f16.gguf --outtype f16
  "$LLAMA_CPP_DIR/build/bin/llama-imatrix" \
    --model models/quantized/p0a4r3-selected-f16.gguf \
    --file data/distill/p0a4r3_imatrix_calibration.txt \
    --output-frequency 10 \
    --output-file models/quantized/p0a4r3-selected.imatrix \
    --ctx-size 512 --threads 8
  "$PYTHON_BIN" scripts/prepare_edge_gguf.py \
    --merged-hf-dir "$model" --llama-cpp-dir "$LLAMA_CPP_DIR" \
    --f16-gguf models/quantized/p0a4r3-selected-f16.gguf \
    --quantized-gguf models/quantized/p0a4r3-selected-q4_k_m.gguf \
    --quant-type Q4_K_M \
    --imatrix models/quantized/p0a4r3-selected.imatrix \
    --max-quantized-bytes 1150000000 --skip-f16-if-exists \
    --audit reports/audit/gate_p0a4r3_selected_q4_prepare.json
}

status() {
  "$PYTHON_BIN" model_compression/build_p0a4r3_code_data.py status
  "$PYTHON_BIN" model_compression/generate_p0a4r3_nlp_data.py status
  for path in "$PROTOCOL_AUDIT" "$DATA_AUDIT" "$SELECTION_AUDIT"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"))' "$path"
    else
      echo "$path missing"
    fi
  done
}

case "${1:-help}" in
  protocol) protocol ;;
  apps-expand) apps_expand ;;
  code-download) code_download ;;
  code-build) code_build ;;
  code-all) code_download; code_build ;;
  nlp-prepare) nlp_prepare ;;
  nlp-generate) nlp_generate ;;
  assemble) assemble ;;
  preflight) preflight ;;
  train) train_candidate "${2:-}" ;;
  merge) merge_candidate "${2:-}" ;;
  evaluate-base) evaluate_base ;;
  evaluate) evaluate_candidate "${2:-}" ;;
  select) select_candidate ;;
  quantize) quantize_selected ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a4r3.sh
    "$PYTHON_BIN" -m py_compile \
      model_compression/code_contests_utils.py \
      model_compression/build_p0a4r3_code_data.py \
      model_compression/generate_p0a4r3_nlp_data.py \
      model_compression/assemble_p0a4r3_shared_data.py \
      scripts/p0a4r3_protocol.py scripts/select_p0a4r3_candidate.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a4r3.sh <command>"
    echo "Commands:"
    echo "  protocol | apps-expand | code-download | code-build | code-all"
    echo "  nlp-prepare | nlp-generate | assemble | preflight"
    echo "  train 1|2 | merge 1|2 | evaluate-base | evaluate 1|2"
    echo "  select | quantize | status | structural-check"
    ;;
esac
