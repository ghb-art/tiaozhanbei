#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG="configs/p0a6_accuracy.json"
GPUS="${P0A6_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A6_SERVE_GPU:-0}"
ENDPOINT="${P0A6_ENDPOINT:-http://127.0.0.1:18460}"
PORT="${P0A6_PORT:-18460}"
BASE_MODEL_ID="p0a6-base"
OUTPUT_DIR="models/checkpoints/p0a6/student-pilot"
NLP_OUTPUT_DIR="models/checkpoints/p0a6/nlp-specialist"
NLP_MCQ_OUTPUT_DIR="models/checkpoints/p0a6/nlp-mcq-specialist"
NLP_RATIONALE_OUTPUT_DIR="models/checkpoints/p0a6/nlp-rationale-specialist"
NLP_ANSWER_FIRST_OUTPUT_DIR="models/checkpoints/p0a6/nlp-answer-first-specialist"
AUDIT_ROOT="reports/audit/p0a6"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; return 1; }
}

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
allowed = set(sys.argv[2].split(","))
if not path.is_file():
    raise SystemExit(f"Missing audit: {path}")
status = json.loads(path.read_text(encoding="utf-8")).get("status")
if status not in allowed:
    raise SystemExit(f"Audit rejected: {path} status={status} allowed={sorted(allowed)}")
print(f"Audit guard passed: {path} status={status}")
PY
}

require_checkpoint() {
  local checkpoint="$OUTPUT_DIR/checkpoint-$1"
  require_file "$checkpoint/trainer_state.json"
  require_file "$checkpoint/adapter_config.json"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" && ! -s "$checkpoint/adapter_model.bin" ]]; then
    echo "Missing adapter weights: $checkpoint" >&2
    return 1
  fi
}

require_nlp_checkpoint() {
  local checkpoint="$NLP_OUTPUT_DIR/checkpoint-$1"
  require_file "$checkpoint/trainer_state.json"
  require_file "$checkpoint/adapter_config.json"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" && ! -s "$checkpoint/adapter_model.bin" ]]; then
    echo "Missing NLP specialist adapter weights: $checkpoint" >&2
    return 1
  fi
}

require_nlp_mcq_checkpoint() {
  local checkpoint="$NLP_MCQ_OUTPUT_DIR/checkpoint-$1"
  require_file "$checkpoint/trainer_state.json"
  require_file "$checkpoint/adapter_config.json"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" && ! -s "$checkpoint/adapter_model.bin" ]]; then
    echo "Missing NLP MCQ adapter weights: $checkpoint" >&2
    return 1
  fi
}

require_nlp_rationale_checkpoint() {
  local checkpoint="$NLP_RATIONALE_OUTPUT_DIR/checkpoint-$1"
  require_file "$checkpoint/trainer_state.json"
  require_file "$checkpoint/adapter_config.json"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" && ! -s "$checkpoint/adapter_model.bin" ]]; then
    echo "Missing NLP rationale adapter weights: $checkpoint" >&2
    return 1
  fi
}

require_nlp_answer_first_checkpoint() {
  local checkpoint="$NLP_ANSWER_FIRST_OUTPUT_DIR/checkpoint-$1"
  require_file "$checkpoint/trainer_state.json"
  require_file "$checkpoint/adapter_config.json"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" && ! -s "$checkpoint/adapter_model.bin" ]]; then
    echo "Missing NLP answer-first adapter weights: $checkpoint" >&2
    return 1
  fi
}

validate_gpu_group() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
if len(values) != 4 or len(set(values)) != 4 or not all(item.isdigit() for item in values):
    raise SystemExit(f"P0-A6 pilot requires four distinct GPU ids, got: {values}")
print(f"P0-A6 GPU group: {','.join(values)}")
PY
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a6_data.py
}

protocol_check() {
  require_file "$CONFIG"
  require_file data/p0a6/manifest.json
  require_file reports/audit/gate_p0a6_data.json
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import math
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = json.loads(Path(config["artifacts"]["data_audit"]).read_text(encoding="utf-8"))
manifest = json.loads(Path(config["data"]["manifest"]).read_text(encoding="utf-8"))
if audit.get("status") != "passed":
    raise SystemExit(f"P0-A6 data audit is not passed: {audit.get('status')}")
if audit["outputs"]["train"]["sha256"] != manifest["train"]["sha256"]:
    raise SystemExit("P0-A6 train hash mismatch between audit and manifest")
if audit["counts"]["train_by_task"] != config["data"]["expected_train_counts"]:
    raise SystemExit("P0-A6 train counts differ from preregistered config")
for domain, expected in config["data"]["effective_task_mass"].items():
    actual = float(audit["effective_task_mass"][domain])
    if not math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-9):
        raise SystemExit(f"Task mass mismatch for {domain}: {actual} != {expected}")
if audit["policy"]["kl_weight"] != config["base_preservation"]["kl_weight"]:
    raise SystemExit("P0-A6 KL weights differ from preregistered config")
if audit["policy"].get("formal_test_loaded") is not False:
    raise SystemExit("Formal test data was read by the P0-A6 builder")
print("P0-A6 protocol check passed: data counts, hashes, task mass and KL weights match.")
PY
}

preflight() {
  protocol_check
  require_dir models/checkpoints/p0a4/student-shared-merged
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_student_preflight.json \
    --max-steps 200 --dry-run
  require_status reports/audit/gate_p0a6_student_preflight.json dry_run_passed
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py --help >/dev/null
  echo "P0-A6 CPU preflight complete. No GPU process was started."
}

student_train_pilot() {
  validate_gpu_group
  protocol_check
  require_status reports/audit/gate_p0a6_student_preflight.json dry_run_passed
  if [[ -f reports/audit/gate_p0a6_student_pilot.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a6_student_pilot.json")).get("status", ""))')"
    if [[ "$status" == "passed" ]]; then
      require_checkpoint 100
      require_checkpoint 200
      echo "P0-A6 Student pilot is already complete."
      return 0
    fi
  fi

  local resume_args=()
  if [[ -d "$OUTPUT_DIR" ]]; then
    if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
      resume_args=(--resume-from-checkpoint auto)
      echo "A checkpoint exists; the trainer will resume from the latest complete one."
    fi
  elif [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing non-directory output path: $OUTPUT_DIR" >&2
    return 2
  fi

  mkdir -p logs "$AUDIT_ROOT"
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_student_pilot.json \
    --max-steps 200 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    "${resume_args[@]}"
  require_status reports/audit/gate_p0a6_student_pilot.json passed
  require_checkpoint 100
  require_checkpoint 200
}

student_train_extension() {
  validate_gpu_group
  protocol_check
  require_status reports/audit/gate_p0a6_student_pilot.json passed
  require_status "$AUDIT_ROOT/checkpoint_selection.json" failed
  require_checkpoint 200
  if [[ -f reports/audit/gate_p0a6_student_extension.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a6_student_extension.json")).get("status", ""))')"
    if [[ "$status" == "passed" ]]; then
      require_checkpoint 300
      echo "P0-A6 Student extension is already complete."
      return 0
    fi
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_student_extension.json \
    --max-steps 300 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --resume-from-checkpoint "$OUTPUT_DIR/checkpoint-200"
  require_status reports/audit/gate_p0a6_student_extension.json passed
  require_checkpoint 300
}

validation_serve() {
  require_checkpoint 100
  require_checkpoint 200
  exec "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 \
    --served-model-name "$BASE_MODEL_ID" \
    --lora-module "p0a6-step-100=$ROOT/$OUTPUT_DIR/checkpoint-100" \
    --lora-module "p0a6-step-200=$ROOT/$OUTPUT_DIR/checkpoint-200" \
    --max-model-len 1536 --gpu-memory-utilization 0.80
}

validation_plan() {
  require_checkpoint 100
  require_checkpoint 200
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 \
    --served-model-name "$BASE_MODEL_ID" \
    --lora-module "p0a6-step-100=$ROOT/$OUTPUT_DIR/checkpoint-100" \
    --lora-module "p0a6-step-200=$ROOT/$OUTPUT_DIR/checkpoint-200" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 --dry-run
}

quick_eval() {
  local target="${1:-}"
  local model_id candidate slug
  case "$target" in
    base)
      model_id="$BASE_MODEL_ID"; candidate="p0a6-base"; slug="base" ;;
    100|200|300)
      model_id="p0a6-step-$target"; candidate="p0a6-student-step-$target"; slug="step_$target" ;;
    *)
      echo "Quick evaluation target must be base, 100, 200 or 300" >&2
      return 2 ;;
  esac
  mkdir -p "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/quick_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" \
    --candidate-name "$candidate" \
    --output-trace "$AUDIT_ROOT/${slug}_quick_trace.jsonl" \
    --audit "$AUDIT_ROOT/${slug}_quick.json"
}

select_checkpoint() {
  require_status "$AUDIT_ROOT/base_quick.json" passed
  require_status "$AUDIT_ROOT/step_100_quick.json" passed
  require_status "$AUDIT_ROOT/step_200_quick.json" passed
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit "$AUDIT_ROOT/base_quick.json" \
    --candidate "100=$AUDIT_ROOT/step_100_quick.json" \
    --candidate "200=$AUDIT_ROOT/step_200_quick.json" \
    --output "$AUDIT_ROOT/checkpoint_selection.json"
}

select_extension_checkpoint() {
  require_status "$AUDIT_ROOT/base_quick.json" passed
  require_status "$AUDIT_ROOT/step_200_quick.json" passed
  require_status "$AUDIT_ROOT/step_300_quick.json" passed
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit "$AUDIT_ROOT/base_quick.json" \
    --candidate "200=$AUDIT_ROOT/step_200_quick.json" \
    --candidate "300=$AUDIT_ROOT/step_300_quick.json" \
    --output "$AUDIT_ROOT/checkpoint_selection_extension.json"
}

full_eval_selected() {
  local selection_audit="${1:-$AUDIT_ROOT/checkpoint_selection.json}"
  require_status "$selection_audit" passed
  local step
  step="$($PYTHON_BIN - "$selection_audit" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["selected_step"])
PY
)"
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "p0a6-step-$step" \
    --candidate-name "p0a6-student-step-$step" \
    --output-trace "$AUDIT_ROOT/step_${step}_full_trace.jsonl" \
    --audit "$AUDIT_ROOT/step_${step}_full.json"
}

wait_for_endpoint() {
  local server_pid="$1"
  local required_ids="${2:-p0a6-base,p0a6-step-100,p0a6-step-200}"
  local attempt
  for attempt in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Validation service exited before becoming ready; see logs/p0a6_validation_server.log" >&2
      return 1
    fi
    if "$PYTHON_BIN" - "$ENDPOINT" "$required_ids" >/dev/null 2>&1 <<'PY'
import json
import sys
from urllib.request import urlopen
url = sys.argv[1].rstrip("/") + "/v1/models"
with urlopen(url, timeout=2) as response:
    payload = json.load(response)
ids = {item.get("id") for item in payload.get("data", [])}
required = {item for item in sys.argv[2].split(",") if item}
if not required.issubset(ids):
    raise SystemExit(1)
PY
    then
      echo "P0-A6 validation service is ready."
      return 0
    fi
    sleep 2
  done
  echo "Validation service did not become ready within 360 seconds." >&2
  return 1
}

validation_auto() {
  require_file scripts/select_p0a6_checkpoint.py
  require_checkpoint 100
  require_checkpoint 200
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 \
    --served-model-name "$BASE_MODEL_ID" \
    --lora-module "p0a6-step-100=$ROOT/$OUTPUT_DIR/checkpoint-100" \
    --lora-module "p0a6-step-200=$ROOT/$OUTPUT_DIR/checkpoint-200" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a6_validation_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a6_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f runtime/p0a6_validation_server.pid
  }
  trap cleanup_validation EXIT INT TERM
  wait_for_endpoint "$server_pid"
  quick_eval base
  quick_eval 100
  quick_eval 200
  select_checkpoint
  full_eval_selected
  cleanup_validation
  trap - EXIT INT TERM
}

extension_validation_auto() {
  require_file scripts/select_p0a6_checkpoint.py
  require_checkpoint 300
  require_status "$AUDIT_ROOT/base_quick.json" passed
  require_status "$AUDIT_ROOT/step_200_quick.json" passed
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 \
    --served-model-name "$BASE_MODEL_ID" \
    --lora-module "p0a6-step-300=$ROOT/$OUTPUT_DIR/checkpoint-300" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a6_validation_extension_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a6_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f runtime/p0a6_validation_server.pid
  }
  trap cleanup_validation EXIT INT TERM
  wait_for_endpoint "$server_pid" "p0a6-base,p0a6-step-300"
  quick_eval 300
  select_extension_checkpoint
  full_eval_selected "$AUDIT_ROOT/checkpoint_selection_extension.json"
  cleanup_validation
  trap - EXIT INT TERM
}

extension_auto() {
  student_train_extension
  extension_validation_auto
}

nlp_specialist_preflight() {
  protocol_check
  require_status reports/audit/gate_p0a6_merge_step_200.json passed
  require_dir models/checkpoints/p0a6/student-step-200-merged
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a6/student-step-200-merged \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$NLP_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_nlp_specialist_preflight.json \
    --max-steps 200 --focus-domain nlp --learning-rate 0.000005 \
    --mcq-answer-token-weight-multiplier 2 --dry-run
  require_status reports/audit/gate_p0a6_nlp_specialist_preflight.json dry_run_passed
}

nlp_specialist_train() {
  validate_gpu_group
  require_status reports/audit/gate_p0a6_nlp_specialist_preflight.json dry_run_passed
  if [[ -f reports/audit/gate_p0a6_train_nlp_specialist.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a6_train_nlp_specialist.json")).get("status", ""))')"
    if [[ "$status" == "passed" ]]; then
      require_nlp_checkpoint 100
      require_nlp_checkpoint 200
      echo "P0-A6 NLP specialist training is already complete."
      return 0
    fi
  fi
  local resume_args=()
  if [[ -d "$NLP_OUTPUT_DIR" ]] && find "$NLP_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume_args=(--resume-from-checkpoint auto)
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a6/student-step-200-merged \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$NLP_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_train_nlp_specialist.json \
    --max-steps 200 --focus-domain nlp --learning-rate 0.000005 \
    --mcq-answer-token-weight-multiplier 2 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume_args[@]}"
  require_status reports/audit/gate_p0a6_train_nlp_specialist.json passed
  require_nlp_checkpoint 100
  require_nlp_checkpoint 200
}

router_quick_eval() {
  local step="${1:-}"
  [[ "$step" == "100" || "$step" == "200" ]] || {
    echo "NLP specialist step must be 100 or 200" >&2
    return 2
  }
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/quick_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "p0a6-shared-step-200" \
    --model-id-math "p0a6-shared-step-200" \
    --model-id-code "p0a6-shared-step-200" \
    --model-id-nlp "p0a6-nlp-specialist-$step" \
    --candidate-name "p0a6-router-nlp-step-$step" \
    --output-trace "$AUDIT_ROOT/router_nlp_${step}_quick_trace.jsonl" \
    --audit "$AUDIT_ROOT/router_nlp_${step}_quick.json"
}

select_router_checkpoint() {
  require_status "$AUDIT_ROOT/base_quick.json" passed
  require_status "$AUDIT_ROOT/router_nlp_100_quick.json" passed
  require_status "$AUDIT_ROOT/router_nlp_200_quick.json" passed
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit "$AUDIT_ROOT/base_quick.json" \
    --candidate "100=$AUDIT_ROOT/router_nlp_100_quick.json" \
    --candidate "200=$AUDIT_ROOT/router_nlp_200_quick.json" \
    --output "$AUDIT_ROOT/router_checkpoint_selection.json"
}

router_full_eval_selected() {
  require_status "$AUDIT_ROOT/router_checkpoint_selection.json" passed
  local step
  step="$($PYTHON_BIN - <<'PY'
import json
print(json.load(open("reports/audit/p0a6/router_checkpoint_selection.json"))["selected_step"])
PY
)"
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "p0a6-shared-step-200" \
    --model-id-math "p0a6-shared-step-200" \
    --model-id-code "p0a6-shared-step-200" \
    --model-id-nlp "p0a6-nlp-specialist-$step" \
    --candidate-name "p0a6-router-nlp-step-$step" \
    --output-trace "$AUDIT_ROOT/router_nlp_${step}_full_trace.jsonl" \
    --audit "$AUDIT_ROOT/router_nlp_${step}_full.json"
}

router_validation_auto() {
  require_nlp_checkpoint 100
  require_nlp_checkpoint 200
  require_status "$AUDIT_ROOT/base_quick.json" passed
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a6/student-step-200-merged \
    --quantization none --tensor-parallel-size 1 \
    --served-model-name "p0a6-shared-step-200" \
    --lora-module "p0a6-nlp-specialist-100=$ROOT/$NLP_OUTPUT_DIR/checkpoint-100" \
    --lora-module "p0a6-nlp-specialist-200=$ROOT/$NLP_OUTPUT_DIR/checkpoint-200" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a6_router_validation_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a6_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f runtime/p0a6_validation_server.pid
  }
  trap cleanup_validation EXIT INT TERM
  wait_for_endpoint "$server_pid" "p0a6-shared-step-200,p0a6-nlp-specialist-100,p0a6-nlp-specialist-200"
  router_quick_eval 100
  router_quick_eval 200
  select_router_checkpoint
  router_full_eval_selected
  cleanup_validation
  trap - EXIT INT TERM
}

nlp_specialist_auto() {
  nlp_specialist_preflight
  nlp_specialist_train
  router_validation_auto
}

nlp_mcq_preflight() {
  protocol_check
  require_dir models/checkpoints/p0a4/student-shared-merged
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$NLP_MCQ_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_nlp_mcq_preflight.json \
    --max-steps 100 --checkpoint-steps 50 \
    --focus-domain nlp_mcq --learning-rate 0.000002 \
    --mcq-answer-token-weight-multiplier 4 --dry-run
  require_status reports/audit/gate_p0a6_nlp_mcq_preflight.json dry_run_passed
}

nlp_mcq_train() {
  validate_gpu_group
  require_status reports/audit/gate_p0a6_nlp_mcq_preflight.json dry_run_passed
  if [[ -f reports/audit/gate_p0a6_train_nlp_mcq.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a6_train_nlp_mcq.json")).get("status", ""))')"
    if [[ "$status" == "passed" ]]; then
      require_nlp_mcq_checkpoint 50
      require_nlp_mcq_checkpoint 100
      echo "P0-A6 NLP MCQ specialist training is already complete."
      return 0
    fi
  fi
  local resume_args=()
  if [[ -d "$NLP_MCQ_OUTPUT_DIR" ]] && find "$NLP_MCQ_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume_args=(--resume-from-checkpoint auto)
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/train.jsonl \
    --output-dir "$NLP_MCQ_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_train_nlp_mcq.json \
    --max-steps 100 --checkpoint-steps 50 \
    --focus-domain nlp_mcq --learning-rate 0.000002 \
    --mcq-answer-token-weight-multiplier 4 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume_args[@]}"
  require_status reports/audit/gate_p0a6_train_nlp_mcq.json passed
  require_nlp_mcq_checkpoint 50
  require_nlp_mcq_checkpoint 100
}

mcq_router_quick_eval() {
  local step="${1:-}"
  [[ "$step" == "50" || "$step" == "100" ]] || {
    echo "NLP MCQ specialist step must be 50 or 100" >&2
    return 2
  }
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/quick_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$BASE_MODEL_ID" \
    --model-id-math "$BASE_MODEL_ID" \
    --model-id-code "p0a6-step-200" \
    --model-id-nlp "p0a6-nlp-mcq-$step" \
    --candidate-name "p0a6-router-mcq-step-$step" \
    --output-trace "$AUDIT_ROOT/router_mcq_${step}_quick_trace.jsonl" \
    --audit "$AUDIT_ROOT/router_mcq_${step}_quick.json"
}

verify_mcq_router_identity() {
  local audit="$1" step="$2"
  "$PYTHON_BIN" - "$audit" "$step" "$BASE_MODEL_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
step = sys.argv[2]
base = sys.argv[3]
report = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "math": base,
    "code": "p0a6-step-200",
    "nlp": f"p0a6-nlp-mcq-{step}",
}
actual = report.get("served_model_id_by_domain")
if actual != expected:
    raise SystemExit(f"Router identity mismatch: {actual} != {expected}")
print(f"Router identity guard passed: {path}")
PY
}

select_mcq_router_checkpoint() {
  require_status "$AUDIT_ROOT/base_quick.json" passed
  require_status "$AUDIT_ROOT/router_mcq_50_quick.json" passed
  require_status "$AUDIT_ROOT/router_mcq_100_quick.json" passed
  verify_mcq_router_identity "$AUDIT_ROOT/router_mcq_50_quick.json" 50
  verify_mcq_router_identity "$AUDIT_ROOT/router_mcq_100_quick.json" 100
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit "$AUDIT_ROOT/base_quick.json" \
    --candidate "50=$AUDIT_ROOT/router_mcq_50_quick.json" \
    --candidate "100=$AUDIT_ROOT/router_mcq_100_quick.json" \
    --output "$AUDIT_ROOT/router_mcq_checkpoint_selection.json"
}

mcq_router_full_eval() {
  require_status "$AUDIT_ROOT/router_mcq_checkpoint_selection.json" passed
  local step
  step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a6/router_mcq_checkpoint_selection.json"))["selected_step"])')"
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$BASE_MODEL_ID" \
    --candidate-name "p0a6-base-full" \
    --output-trace "$AUDIT_ROOT/base_full_trace.jsonl" \
    --audit "$AUDIT_ROOT/base_full.json"
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$BASE_MODEL_ID" \
    --model-id-math "$BASE_MODEL_ID" \
    --model-id-code "p0a6-step-200" \
    --model-id-nlp "p0a6-nlp-mcq-$step" \
    --candidate-name "p0a6-router-mcq-step-$step-full" \
    --output-trace "$AUDIT_ROOT/router_mcq_${step}_full_trace.jsonl" \
    --audit "$AUDIT_ROOT/router_mcq_${step}_full.json"
  verify_mcq_router_identity "$AUDIT_ROOT/router_mcq_${step}_full.json" "$step"
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --validation-manifest data/p0a6/full_validation.jsonl \
    --base-audit "$AUDIT_ROOT/base_full.json" \
    --candidate "$step=$AUDIT_ROOT/router_mcq_${step}_full.json" \
    --output "$AUDIT_ROOT/router_mcq_full_selection.json"
}

nlp_mcq_validation_auto() {
  require_checkpoint 200
  require_nlp_mcq_checkpoint 50
  require_nlp_mcq_checkpoint 100
  require_status "$AUDIT_ROOT/base_quick.json" passed
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 \
    --served-model-name "$BASE_MODEL_ID" \
    --lora-module "p0a6-step-200=$ROOT/$OUTPUT_DIR/checkpoint-200" \
    --lora-module "p0a6-nlp-mcq-50=$ROOT/$NLP_MCQ_OUTPUT_DIR/checkpoint-50" \
    --lora-module "p0a6-nlp-mcq-100=$ROOT/$NLP_MCQ_OUTPUT_DIR/checkpoint-100" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a6_nlp_mcq_validation_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a6_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f runtime/p0a6_validation_server.pid
  }
  trap cleanup_validation EXIT INT TERM
  wait_for_endpoint "$server_pid" "$BASE_MODEL_ID,p0a6-step-200,p0a6-nlp-mcq-50,p0a6-nlp-mcq-100"
  mcq_router_quick_eval 50
  mcq_router_quick_eval 100
  select_mcq_router_checkpoint
  mcq_router_full_eval
  cleanup_validation
  trap - EXIT INT TERM
}

nlp_mcq_auto() {
  nlp_mcq_preflight
  nlp_mcq_train
  nlp_mcq_validation_auto
}

wait_for_named_endpoint() {
  local server_pid="$1" endpoint="$2" required_ids="$3" log_path="$4"
  local attempt
  for attempt in $(seq 1 240); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Model service exited before becoming ready; see $log_path" >&2
      return 1
    fi
    if "$PYTHON_BIN" - "$endpoint" "$required_ids" >/dev/null 2>&1 <<'PY'
import json
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1].rstrip("/") + "/v1/models", timeout=2) as response:
    payload = json.load(response)
actual = {str(item.get("id")) for item in payload.get("data", [])}
required = {item for item in sys.argv[2].split(",") if item}
if not required.issubset(actual):
    raise SystemExit(1)
PY
    then
      echo "Model service is ready: $required_ids"
      return 0
    fi
    sleep 2
  done
  echo "Model service did not become ready within 480 seconds; see $log_path" >&2
  return 1
}

nlp_rationale_data_preflight() {
  require_status reports/audit/gate_p0a5_train_teacher.json passed
  "$PYTHON_BIN" model_compression/generate_p0a6_mcq_rationales.py \
    --source data/p0a6/train.jsonl \
    --output data/p0a6/nlp_mcq_rationale_train.jsonl \
    --trace data/p0a6/nlp_mcq_rationale_trace.jsonl \
    --audit reports/audit/gate_p0a6_mcq_rationales.json \
    --expected-rows 1335 --dry-run
  require_status reports/audit/gate_p0a6_mcq_rationales.json passed
}

nlp_rationale_generate() {
  validate_gpu_group
  require_status reports/audit/gate_p0a5_train_teacher.json passed
  # A failed generation audit is a valid resume point: the generator reuses
  # only hash-validated accepted rows and retries the rejected remainder.
  require_status reports/audit/gate_p0a6_mcq_rationales.json passed,failed
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
    >logs/p0a6_rationale_teacher_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a6_rationale_teacher_server.pid
  cleanup_teacher() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f runtime/p0a6_rationale_teacher_server.pid
  }
  trap cleanup_teacher EXIT INT TERM
  wait_for_named_endpoint "$server_pid" "http://127.0.0.1:8000" \
    "p0a5-teacher" "logs/p0a6_rationale_teacher_server.log"
  "$PYTHON_BIN" model_compression/generate_p0a6_mcq_rationales.py \
    --source data/p0a6/train.jsonl \
    --output data/p0a6/nlp_mcq_rationale_train.jsonl \
    --trace data/p0a6/nlp_mcq_rationale_trace.jsonl \
    --audit reports/audit/gate_p0a6_mcq_rationales.json \
    --endpoint http://127.0.0.1:8000 --model-id p0a5-teacher \
    --workers "${P0A6_RATIONALE_WORKERS:-16}" \
    --max-attempts 3 --max-tokens 384 --temperature 0 \
    --timeout-sec 120 --expected-rows 1335 --resume
  require_status reports/audit/gate_p0a6_mcq_rationales.json passed
  cleanup_teacher
  trap - EXIT INT TERM
}

nlp_rationale_student_preflight() {
  require_status reports/audit/gate_p0a6_mcq_rationales.json passed
  require_file data/p0a6/nlp_mcq_rationale_train.jsonl
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/nlp_mcq_rationale_train.jsonl \
    --output-dir "$NLP_RATIONALE_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_nlp_rationale_preflight.json \
    --max-steps 100 --checkpoint-steps 50 \
    --focus-domain nlp_rationale --learning-rate 0.000002 \
    --mcq-answer-token-weight-multiplier 4 --dry-run
  require_status reports/audit/gate_p0a6_nlp_rationale_preflight.json dry_run_passed
}

nlp_rationale_train() {
  validate_gpu_group
  nlp_rationale_student_preflight
  require_status reports/audit/gate_p0a6_nlp_rationale_preflight.json dry_run_passed
  if [[ -f reports/audit/gate_p0a6_train_nlp_rationale.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a6_train_nlp_rationale.json")).get("status", ""))')"
    if [[ "$status" == "passed" ]]; then
      require_nlp_rationale_checkpoint 50
      require_nlp_rationale_checkpoint 100
      echo "P0-A6 NLP rationale specialist training is already complete."
      return 0
    fi
  fi
  local resume_args=()
  if [[ -d "$NLP_RATIONALE_OUTPUT_DIR" ]] && find "$NLP_RATIONALE_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume_args=(--resume-from-checkpoint auto)
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/nlp_mcq_rationale_train.jsonl \
    --output-dir "$NLP_RATIONALE_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_train_nlp_rationale.json \
    --max-steps 100 --checkpoint-steps 50 \
    --focus-domain nlp_rationale --learning-rate 0.000002 \
    --mcq-answer-token-weight-multiplier 4 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume_args[@]}"
  require_status reports/audit/gate_p0a6_train_nlp_rationale.json passed
  require_nlp_rationale_checkpoint 50
  require_nlp_rationale_checkpoint 100
}

rationale_router_eval() {
  local step="$1" scope="$2" manifest output_prefix
  [[ "$step" == "50" || "$step" == "100" ]] || return 2
  if [[ "$scope" == "quick" ]]; then
    manifest="data/p0a6/quick_validation.jsonl"
    output_prefix="$AUDIT_ROOT/router_rationale_${step}_quick"
  elif [[ "$scope" == "full" ]]; then
    manifest="data/p0a6/full_validation.jsonl"
    output_prefix="$AUDIT_ROOT/router_rationale_${step}_full"
  else
    echo "Rationale evaluation scope must be quick or full" >&2
    return 2
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest "$manifest" --endpoint "$ENDPOINT" --model-id "$BASE_MODEL_ID" \
    --model-id-math "$BASE_MODEL_ID" --model-id-code p0a6-step-200 \
    --model-id-nlp "p0a6-nlp-rationale-$step" \
    --candidate-name "p0a6-router-rationale-step-$step-$scope" \
    --output-trace "${output_prefix}_trace.jsonl" --audit "${output_prefix}.json"
}

verify_rationale_router_identity() {
  local audit="$1" step="$2"
  "$PYTHON_BIN" - "$audit" "$step" "$BASE_MODEL_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "math": sys.argv[3],
    "code": "p0a6-step-200",
    "nlp": f"p0a6-nlp-rationale-{sys.argv[2]}",
}
actual = json.loads(path.read_text(encoding="utf-8")).get("served_model_id_by_domain")
if actual != expected:
    raise SystemExit(f"Rationale router identity mismatch: {actual} != {expected}")
print(f"Rationale router identity guard passed: {path}")
PY
}

nlp_rationale_validation_auto() {
  require_checkpoint 200
  require_nlp_rationale_checkpoint 50
  require_nlp_rationale_checkpoint 100
  require_status "$AUDIT_ROOT/base_quick.json" passed
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 --served-model-name "$BASE_MODEL_ID" \
    --lora-module "p0a6-step-200=$ROOT/$OUTPUT_DIR/checkpoint-200" \
    --lora-module "p0a6-nlp-rationale-50=$ROOT/$NLP_RATIONALE_OUTPUT_DIR/checkpoint-50" \
    --lora-module "p0a6-nlp-rationale-100=$ROOT/$NLP_RATIONALE_OUTPUT_DIR/checkpoint-100" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a6_nlp_rationale_validation_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a6_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f runtime/p0a6_validation_server.pid
  }
  trap cleanup_validation EXIT INT TERM
  wait_for_endpoint "$server_pid" "$BASE_MODEL_ID,p0a6-step-200,p0a6-nlp-rationale-50,p0a6-nlp-rationale-100"
  rationale_router_eval 50 quick
  rationale_router_eval 100 quick
  verify_rationale_router_identity "$AUDIT_ROOT/router_rationale_50_quick.json" 50
  verify_rationale_router_identity "$AUDIT_ROOT/router_rationale_100_quick.json" 100
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit "$AUDIT_ROOT/base_quick.json" \
    --candidate "50=$AUDIT_ROOT/router_rationale_50_quick.json" \
    --candidate "100=$AUDIT_ROOT/router_rationale_100_quick.json" \
    --output "$AUDIT_ROOT/router_rationale_checkpoint_selection.json"
  local step
  step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a6/router_rationale_checkpoint_selection.json"))["selected_step"])')"
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id "$BASE_MODEL_ID" --candidate-name p0a6-base-full \
    --output-trace "$AUDIT_ROOT/base_full_trace.jsonl" --audit "$AUDIT_ROOT/base_full.json"
  rationale_router_eval "$step" full
  verify_rationale_router_identity "$AUDIT_ROOT/router_rationale_${step}_full.json" "$step"
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --validation-manifest data/p0a6/full_validation.jsonl \
    --base-audit "$AUDIT_ROOT/base_full.json" \
    --candidate "$step=$AUDIT_ROOT/router_rationale_${step}_full.json" \
    --output "$AUDIT_ROOT/router_rationale_full_selection.json"
  cleanup_validation
  trap - EXIT INT TERM
}

nlp_rationale_auto() {
  nlp_rationale_data_preflight
  nlp_rationale_generate
  nlp_rationale_student_preflight
  nlp_rationale_train
  nlp_rationale_validation_auto
}

nlp_answer_first_data() {
  require_status reports/audit/gate_p0a6_mcq_rationales.json passed
  "$PYTHON_BIN" model_compression/build_p0a6_answer_first_data.py
  require_status reports/audit/gate_p0a6_answer_first_data.json passed
}

nlp_answer_first_preflight() {
  require_status reports/audit/gate_p0a6_answer_first_data.json passed
  require_file data/p0a6/nlp_answer_first_train.jsonl
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/nlp_answer_first_train.jsonl \
    --output-dir "$NLP_ANSWER_FIRST_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_nlp_answer_first_preflight.json \
    --max-steps 50 --checkpoint-steps 25 \
    --focus-domain nlp_answer_first --learning-rate 0.000002 \
    --mcq-answer-token-weight-multiplier 4 --dry-run
  require_status reports/audit/gate_p0a6_nlp_answer_first_preflight.json dry_run_passed
}

nlp_answer_first_train() {
  validate_gpu_group
  nlp_answer_first_preflight
  if [[ -f reports/audit/gate_p0a6_train_nlp_answer_first.json ]]; then
    local status
    status="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/gate_p0a6_train_nlp_answer_first.json")).get("status", ""))')"
    if [[ "$status" == "passed" ]]; then
      require_nlp_answer_first_checkpoint 25
      require_nlp_answer_first_checkpoint 50
      echo "P0-A6 NLP answer-first training is already complete."
      return 0
    fi
  fi
  local resume_args=()
  if [[ -d "$NLP_ANSWER_FIRST_OUTPUT_DIR" ]] && find "$NLP_ANSWER_FIRST_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume_args=(--resume-from-checkpoint auto)
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --train-data data/p0a6/nlp_answer_first_train.jsonl \
    --output-dir "$NLP_ANSWER_FIRST_OUTPUT_DIR" \
    --audit reports/audit/gate_p0a6_train_nlp_answer_first.json \
    --max-steps 50 --checkpoint-steps 25 \
    --focus-domain nlp_answer_first --learning-rate 0.000002 \
    --mcq-answer-token-weight-multiplier 4 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume_args[@]}"
  require_status reports/audit/gate_p0a6_train_nlp_answer_first.json passed
  require_nlp_answer_first_checkpoint 25
  require_nlp_answer_first_checkpoint 50
}

answer_first_router_eval() {
  local step="$1" scope="$2" manifest output_prefix
  [[ "$step" == "25" || "$step" == "50" ]] || return 2
  if [[ "$scope" == "quick" ]]; then
    manifest="data/p0a6/quick_validation.jsonl"
    output_prefix="$AUDIT_ROOT/router_answer_first_${step}_quick"
  elif [[ "$scope" == "full" ]]; then
    manifest="data/p0a6/full_validation.jsonl"
    output_prefix="$AUDIT_ROOT/router_answer_first_${step}_full"
  else
    echo "Answer-first evaluation scope must be quick or full" >&2
    return 2
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest "$manifest" --endpoint "$ENDPOINT" --model-id "$BASE_MODEL_ID" \
    --model-id-math "$BASE_MODEL_ID" --model-id-code p0a6-step-200 \
    --model-id-nlp "p0a6-nlp-answer-first-$step" \
    --candidate-name "p0a6-router-answer-first-step-$step-$scope" \
    --output-trace "${output_prefix}_trace.jsonl" --audit "${output_prefix}.json"
}

verify_answer_first_router_identity() {
  local audit="$1" step="$2"
  "$PYTHON_BIN" - "$audit" "$step" "$BASE_MODEL_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "math": sys.argv[3],
    "code": "p0a6-step-200",
    "nlp": f"p0a6-nlp-answer-first-{sys.argv[2]}",
}
actual = json.loads(path.read_text(encoding="utf-8")).get("served_model_id_by_domain")
if actual != expected:
    raise SystemExit(f"Answer-first router identity mismatch: {actual} != {expected}")
print(f"Answer-first router identity guard passed: {path}")
PY
}

nlp_answer_first_validation_auto() {
  require_checkpoint 200
  require_nlp_answer_first_checkpoint 25
  require_nlp_answer_first_checkpoint 50
  require_status "$AUDIT_ROOT/base_quick.json" passed
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 --served-model-name "$BASE_MODEL_ID" \
    --lora-module "p0a6-step-200=$ROOT/$OUTPUT_DIR/checkpoint-200" \
    --lora-module "p0a6-nlp-answer-first-25=$ROOT/$NLP_ANSWER_FIRST_OUTPUT_DIR/checkpoint-25" \
    --lora-module "p0a6-nlp-answer-first-50=$ROOT/$NLP_ANSWER_FIRST_OUTPUT_DIR/checkpoint-50" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a6_nlp_answer_first_validation_server.log 2>&1 &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a6_validation_server.pid
  cleanup_validation() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f runtime/p0a6_validation_server.pid
  }
  trap cleanup_validation EXIT INT TERM
  wait_for_endpoint "$server_pid" "$BASE_MODEL_ID,p0a6-step-200,p0a6-nlp-answer-first-25,p0a6-nlp-answer-first-50"
  answer_first_router_eval 25 quick
  answer_first_router_eval 50 quick
  verify_answer_first_router_identity "$AUDIT_ROOT/router_answer_first_25_quick.json" 25
  verify_answer_first_router_identity "$AUDIT_ROOT/router_answer_first_50_quick.json" 50
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --base-audit "$AUDIT_ROOT/base_quick.json" \
    --candidate "25=$AUDIT_ROOT/router_answer_first_25_quick.json" \
    --candidate "50=$AUDIT_ROOT/router_answer_first_50_quick.json" \
    --output "$AUDIT_ROOT/router_answer_first_checkpoint_selection.json"
  local step
  step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a6/router_answer_first_checkpoint_selection.json"))["selected_step"])')"
  "$PYTHON_BIN" scripts/evaluate_p0a6_internal.py \
    --manifest data/p0a6/full_validation.jsonl --endpoint "$ENDPOINT" \
    --model-id "$BASE_MODEL_ID" --candidate-name p0a6-base-full \
    --output-trace "$AUDIT_ROOT/base_full_trace.jsonl" --audit "$AUDIT_ROOT/base_full.json"
  answer_first_router_eval "$step" full
  verify_answer_first_router_identity "$AUDIT_ROOT/router_answer_first_${step}_full.json" "$step"
  "$PYTHON_BIN" scripts/select_p0a6_checkpoint.py \
    --validation-manifest data/p0a6/full_validation.jsonl \
    --base-audit "$AUDIT_ROOT/base_full.json" \
    --candidate "$step=$AUDIT_ROOT/router_answer_first_${step}_full.json" \
    --output "$AUDIT_ROOT/router_answer_first_full_selection.json"
  cleanup_validation
  trap - EXIT INT TERM
}

nlp_answer_first_auto() {
  nlp_answer_first_data
  nlp_answer_first_train
  nlp_answer_first_validation_auto
}

pilot_auto() {
  data_build
  preflight
  student_train_pilot
  validation_auto
}

status() {
  local path
  for path in \
    reports/audit/gate_p0a6_data.json \
    reports/audit/gate_p0a6_student_preflight.json \
    reports/audit/gate_p0a6_student_pilot.json \
    reports/audit/gate_p0a6_student_extension.json \
    reports/audit/gate_p0a6_merge_step_200.json \
    reports/audit/gate_p0a6_nlp_specialist_preflight.json \
    reports/audit/gate_p0a6_train_nlp_specialist.json \
    "$AUDIT_ROOT/base_quick.json" \
    "$AUDIT_ROOT/step_100_quick.json" \
    "$AUDIT_ROOT/step_200_quick.json" \
    "$AUDIT_ROOT/checkpoint_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"), d.get("selected_step", ""))' "$path"
    else
      echo "$path missing"
    fi
  done
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V || true
  for path in "$AUDIT_ROOT/step_300_quick.json" "$AUDIT_ROOT/checkpoint_selection_extension.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"), d.get("selected_step", ""))' "$path"
    else
      echo "$path missing"
    fi
  done
  for path in "$AUDIT_ROOT/router_nlp_100_quick.json" "$AUDIT_ROOT/router_nlp_200_quick.json" "$AUDIT_ROOT/router_checkpoint_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"), d.get("selected_step", ""))' "$path"
    else
      echo "$path missing"
    fi
  done
  if [[ -f runtime/p0a6_pipeline.pid ]]; then
    local pid
    pid="$(cat runtime/p0a6_pipeline.pid)"
    ps -p "$pid" -o pid,etime,%cpu,%mem,cmd --no-headers || echo "Recorded pipeline pid $pid is no longer running."
  fi
}

help_text() {
  cat <<'EOF'
P0-A6 accuracy-first Student pipeline

Preparation and training:
  data-build                 Rebuild the train-only P0-A6 data and manifests.
  preflight                  Validate hashes/config and scan all rows without a GPU.
  student-train-pilot        Run/resume four-GPU LoRA training to steps 100 and 200.
  student-train-extension    Resume step 200 once and stop at step 300.
  nlp-specialist-preflight   Validate the NLP-only routed Adapter plan on CPU.
  nlp-specialist-train       Train/resume two NLP-specialist checkpoints on four GPUs.
  nlp-mcq-preflight          Validate the MCQ-only NLP Adapter plan on CPU.
  nlp-mcq-train              Train MCQ-only checkpoints 50 and 100 on four GPUs.
  nlp-rationale-preflight    Validate label-locked Teacher rationale input on CPU.
  nlp-rationale-generate     Generate 1,335 verified rationales with the 4-GPU Teacher.
  nlp-rationale-train        Train rationale checkpoints 50 and 100 on four GPUs.
  nlp-answer-first-data      Build C-Eval + CMMLU-dev answer-first training rows.
  nlp-answer-first-preflight Validate answer-first rows and token budget on CPU.
  nlp-answer-first-train     Train answer-first checkpoints 25 and 50 on four GPUs.

Internal checkpoint selection:
  validation-plan            Print one-GPU base + two-LoRA vLLM launch plan.
  validation-serve           Serve base, step-100 and step-200 in the foreground.
  quick-eval base|100|200    Evaluate one served model on 100 items per domain.
  select-checkpoint          Apply Math drift and Code/NLP gain constraints.
  full-eval-selected         Evaluate the selected checkpoint on train-only full validation.
  validation-auto            Start service, evaluate all, select, full-evaluate and stop it.
  extension-validation-auto  Evaluate only step 300, reselect, then full-evaluate if passed.
  router-validation-auto     Evaluate routed Math/Code shared base + NLP specialist.
  nlp-mcq-validation-auto    Evaluate base Math, step-200 Code and MCQ NLP routing.
  nlp-rationale-validation-auto Evaluate rationale-routed checkpoints and full gate.
  nlp-answer-first-validation-auto Evaluate answer-first checkpoints and full gate.

Automation:
  pilot-auto                 Build, preflight, train/resume and validate in order.
  extension-auto             Run the single authorized 100-step extension and validate it.
  nlp-specialist-auto        Preflight, train and validate the routed NLP specialist.
  nlp-mcq-auto               Preflight, train and validate the MCQ-only NLP route.
  nlp-rationale-auto         Generate rationales, train and validate in order.
  nlp-answer-first-auto      Build, train and validate the answer-first route.
  status                     Show audits, checkpoints and a detached pipeline process.

No formal GSM8K/HumanEval/CMMLU test split is read by these commands.
EOF
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  student-train-pilot) student_train_pilot ;;
  student-train-extension) student_train_extension ;;
  nlp-specialist-preflight) nlp_specialist_preflight ;;
  nlp-specialist-train) nlp_specialist_train ;;
  nlp-mcq-preflight) nlp_mcq_preflight ;;
  nlp-mcq-train) nlp_mcq_train ;;
  nlp-rationale-preflight) nlp_rationale_data_preflight ;;
  nlp-rationale-generate) nlp_rationale_generate ;;
  nlp-rationale-train) nlp_rationale_train ;;
  nlp-answer-first-data) nlp_answer_first_data ;;
  nlp-answer-first-preflight) nlp_answer_first_preflight ;;
  nlp-answer-first-train) nlp_answer_first_train ;;
  validation-plan) validation_plan ;;
  validation-serve) validation_serve ;;
  quick-eval) quick_eval "${2:-}" ;;
  select-checkpoint) select_checkpoint ;;
  full-eval-selected) full_eval_selected ;;
  validation-auto) validation_auto ;;
  extension-validation-auto) extension_validation_auto ;;
  router-validation-auto) router_validation_auto ;;
  nlp-mcq-validation-auto) nlp_mcq_validation_auto ;;
  nlp-rationale-validation-auto) nlp_rationale_validation_auto ;;
  nlp-answer-first-validation-auto) nlp_answer_first_validation_auto ;;
  pilot-auto) pilot_auto ;;
  extension-auto) extension_auto ;;
  nlp-specialist-auto) nlp_specialist_auto ;;
  nlp-mcq-auto) nlp_mcq_auto ;;
  nlp-rationale-auto) nlp_rationale_auto ;;
  nlp-answer-first-auto) nlp_answer_first_auto ;;
  status) status ;;
  help|-h|--help) help_text ;;
  *)
    echo "Unknown command: $1" >&2
    help_text >&2
    exit 2 ;;
esac
