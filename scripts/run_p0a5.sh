#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG="configs/p0a5_capability.json"
GPUS="${P0A5_GPUS:-0,1,2,3}"
TEACHER_ENDPOINT="${P0A5_TEACHER_ENDPOINT:-http://127.0.0.1:8000}"
BASELINE_ENDPOINT="${P0A5_BASELINE_ENDPOINT:-http://127.0.0.1:8001}"
STUDENT_ENDPOINT="${P0A5_STUDENT_ENDPOINT:-http://127.0.0.1:18450}"
TEACHER_MODEL_ID="${P0A5_TEACHER_MODEL_ID:-p0a5-teacher}"
BASELINE_MODEL_ID="${P0A5_BASELINE_MODEL_ID:-baseline-14b-awq}"
STUDENT_MODEL_ID="${P0A5_STUDENT_MODEL_ID:-p0a5-edge-q4}"

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

require_recommended_gate() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("status") != "passed" or data.get("recommended_full") is not True:
    raise SystemExit(
        f"Gate is not eligible for full evaluation: {path} "
        f"status={data.get('status')} decision={data.get('decision')}"
    )
print(f"Recommended gate passed: {path}")
PY
}

require_second_candidate() {
  "$PYTHON_BIN" - reports/audit/gate_p0a5_student_candidate_1_gate300.json <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Candidate 2 is locked; missing Candidate 1 gate: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
if (
    data.get("status") != "passed"
    or data.get("recommended_full") is not False
    or data.get("decision") != "eligible_for_second_preregistered_candidate"
):
    raise SystemExit(
        "Candidate 2 is not authorized: "
        f"status={data.get('status')} decision={data.get('decision')}"
    )
print("Candidate 2 preregistration guard passed.")
PY
}

data_download() {
  "$PYTHON_BIN" model_compression/build_p0a5_data.py download --config "$CONFIG"
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a5_data.py build \
    --config "$CONFIG" --workers "${P0A5_DATA_WORKERS:-12}"
}

preflight() {
  require_file data/capability_v2/source_train.jsonl
  require_file data/capability_v2/internal_validation.jsonl
  require_file data/capability_v2/gate300.jsonl
  "$PYTHON_BIN" scripts/p0a5_protocol.py --config "$CONFIG"
  require_status reports/audit/gate_p0a5_protocol.json passed
  "$PYTHON_BIN" model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role teacher --candidate-index 1 \
    --train-data data/capability_v2/source_train.jsonl \
    --validation-data data/capability_v2/internal_validation.jsonl \
    --output-dir models/checkpoints/p0a5/teacher \
    --audit reports/audit/gate_p0a5_teacher_preflight.json \
    --deepspeed configs/deepspeed_zero3.json --dry-run
  "$PYTHON_BIN" model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role student --candidate-index 1 \
    --train-data data/capability_v2/source_train.jsonl \
    --validation-data data/capability_v2/internal_validation.jsonl \
    --output-dir models/checkpoints/p0a5/student-candidate-1 \
    --audit reports/audit/gate_p0a5_student_preflight.json --dry-run
  "$PYTHON_BIN" model_compression/generate_p0a5_distill.py \
    --config "$CONFIG" --dry-run
  echo "CPU preflight complete. No GPU command was started."
}

teacher_train() {
  local output="models/checkpoints/p0a5/teacher"
  local latest_checkpoint=""
  local resume_args=()
  require_status reports/audit/gate_p0a5_protocol.json passed
  require_dir models/pretrained/Qwen--Qwen2.5-14B-Instruct
  "$PYTHON_BIN" model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role teacher --candidate-index 1 \
    --train-data data/capability_v2/source_train.jsonl \
    --validation-data data/capability_v2/internal_validation.jsonl \
    --output-dir "$output" \
    --audit reports/audit/gate_p0a5_teacher_preflight.json \
    --deepspeed configs/deepspeed_zero3.json --dry-run
  require_status reports/audit/gate_p0a5_teacher_preflight.json dry_run_passed
  if [[ -d "$output" ]]; then
    while IFS= read -r candidate; do
      if [[
        -s "$output/$candidate/trainer_state.json"
        && -s "$output/$candidate/adapter_model.safetensors"
        && -s "$output/$candidate/latest"
      ]]; then
        latest_checkpoint="$candidate"
      fi
    done < <(
      find "$output" -mindepth 1 -maxdepth 1 -type d \
        -name 'checkpoint-[0-9]*' -printf '%f\n' \
        | sort -V
    )
    [[ -n "$latest_checkpoint" ]] || {
      echo "No complete Teacher checkpoint in: $output" >&2
      return 2
    }
    latest_checkpoint="$output/$latest_checkpoint"
    resume_args=(--resume-from-checkpoint "$latest_checkpoint")
    echo "Resuming Teacher from $latest_checkpoint"
  elif [[ -e "$output" ]]; then
    echo "Refusing non-directory Teacher output: $output" >&2
    return 2
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role teacher --candidate-index 1 \
    --train-data data/capability_v2/source_train.jsonl \
    --validation-data data/capability_v2/internal_validation.jsonl \
    --output-dir "$output" \
    --audit reports/audit/gate_p0a5_train_teacher.json \
    --deepspeed configs/deepspeed_zero3.json \
    --gradient-accumulation-steps 8 \
    "${resume_args[@]}"
}

teacher_eval() {
  local step="${1:-}"
  local checkpoint="models/checkpoints/p0a5/teacher/checkpoint-$step"
  [[ "$step" == "600" || "$step" == "800" ]] || {
    echo "Teacher evaluation step must be 600 or 800" >&2
    return 2
  }
  require_status reports/audit/gate_p0a5_protocol.json passed
  require_file "$checkpoint/trainer_state.json"
  require_file "$checkpoint/adapter_model.safetensors"
  require_file "$checkpoint/latest"
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role teacher --candidate-index 1 \
    --train-data data/capability_v2/source_train.jsonl \
    --validation-data data/capability_v2/internal_validation.jsonl \
    --output-dir models/checkpoints/p0a5/teacher \
    --audit "reports/audit/p0a5_teacher_checkpoint_${step}_eval.json" \
    --deepspeed configs/deepspeed_zero3.json \
    --resume-from-checkpoint "$checkpoint" \
    --evaluate-only
}

teacher_select() {
  require_status reports/audit/p0a5_teacher_checkpoint_600_eval.json passed
  require_status reports/audit/p0a5_teacher_checkpoint_800_eval.json passed
  "$PYTHON_BIN" scripts/select_p0a5_teacher.py \
    --config "$CONFIG" \
    --evaluation reports/audit/p0a5_teacher_checkpoint_600_eval.json \
    --evaluation reports/audit/p0a5_teacher_checkpoint_800_eval.json \
    --output-dir models/checkpoints/p0a5/teacher \
    --audit reports/audit/gate_p0a5_train_teacher.json
}

teacher_plan() {
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port 8000 \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct \
    --quantization none --tensor-parallel-size 4 \
    --lora-module "$TEACHER_MODEL_ID=$ROOT/models/checkpoints/p0a5/teacher" \
    --dry-run
}

teacher_serve() {
  require_status reports/audit/gate_p0a5_train_teacher.json passed
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port 8000 \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct \
    --quantization none --tensor-parallel-size 4 \
    --lora-module "$TEACHER_MODEL_ID=$ROOT/models/checkpoints/p0a5/teacher"
}

distill_generate() {
  require_status reports/audit/gate_p0a5_train_teacher.json passed
  "$PYTHON_BIN" model_compression/generate_p0a5_distill.py \
    --config "$CONFIG" --endpoint "$TEACHER_ENDPOINT" \
    --model-id "$TEACHER_MODEL_ID" --workers "${P0A5_TEACHER_WORKERS:-16}"
}

student_preflight() {
  require_status reports/audit/gate_p0a5_distill.json passed
  local candidate="${1:-1}"
  [[ "$candidate" == "1" || "$candidate" == "2" ]] || {
    echo "Candidate must be 1 or 2" >&2
    return 2
  }
  if [[ "$candidate" == "2" ]]; then
    require_second_candidate
  fi
  "$PYTHON_BIN" model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role student --candidate-index "$candidate" \
    --train-data data/capability_v2/distill_train.jsonl \
    --validation-data data/capability_v2/internal_validation.jsonl \
    --output-dir "models/checkpoints/p0a5/student-candidate-$candidate" \
    --audit "reports/audit/gate_p0a5_student_candidate_${candidate}_preflight.json" \
    --dry-run
}

student_train() {
  local candidate="${1:-1}"
  local latest_checkpoint=""
  local resume_args=()
  require_status reports/audit/gate_p0a5_distill.json passed
  [[ "$candidate" == "1" || "$candidate" == "2" ]] || {
    echo "Candidate must be 1 or 2" >&2
    return 2
  }
  if [[ "$candidate" == "2" ]]; then
    require_second_candidate
  fi
  local output="models/checkpoints/p0a5/student-candidate-$candidate"
  if [[ -d "$output" ]]; then
    while IFS= read -r checkpoint; do
      if [[
        -s "$output/$checkpoint/trainer_state.json"
        && -s "$output/$checkpoint/adapter_model.safetensors"
        && -s "$output/$checkpoint/optimizer.pt"
        && -s "$output/$checkpoint/scheduler.pt"
      ]]; then
        latest_checkpoint="$checkpoint"
      fi
    done < <(
      find "$output" -mindepth 1 -maxdepth 1 -type d \
        -name 'checkpoint-[0-9]*' -printf '%f\n' \
        | sort -V
    )
    if [[ -n "$latest_checkpoint" ]]; then
      latest_checkpoint="$output/$latest_checkpoint"
      resume_args=(--resume-from-checkpoint "$latest_checkpoint")
      echo "Resuming Student Candidate $candidate from $latest_checkpoint"
    elif find "$output" -mindepth 1 -print -quit | grep -q .; then
      echo "No complete Student checkpoint in non-empty output: $output" >&2
      return 2
    fi
  elif [[ -e "$output" ]]; then
    echo "Refusing non-directory Student output: $output" >&2
    return 2
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role student --candidate-index "$candidate" \
    --train-data data/capability_v2/distill_train.jsonl \
    --validation-data data/capability_v2/internal_validation.jsonl \
    --output-dir "$output" \
    --audit "reports/audit/gate_p0a5_train_student_candidate_$candidate.json" \
    --gradient-accumulation-steps 8 \
    "${resume_args[@]}"
}

student_merge() {
  local candidate="${1:-1}"
  require_status "reports/audit/gate_p0a5_train_student_candidate_$candidate.json" passed
  "$PYTHON_BIN" model_compression/merge_lora_adapter.py \
    --base-model models/checkpoints/p0a4/student-shared-merged \
    --adapter "models/checkpoints/p0a5/student-candidate-$candidate" \
    --output "models/checkpoints/p0a5/student-candidate-$candidate-merged" \
    --audit "reports/audit/gate_p0a5_merge_student_candidate_$candidate.json"
}

baseline_gate() {
  "$PYTHON_BIN" scripts/evaluate_p0a5_gate.py \
    --endpoint "$BASELINE_ENDPOINT" --model-id "$BASELINE_MODEL_ID" \
    --candidate-name baseline-14b-awq \
    --output-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --audit reports/audit/gate_p0a5_baseline14b_gate300_eval.json
}

student_gate() {
  local candidate="${1:-1}"
  "$PYTHON_BIN" scripts/evaluate_p0a5_gate.py \
    --endpoint "$STUDENT_ENDPOINT" --model-id "$STUDENT_MODEL_ID" \
    --candidate-name "p0a5-student-candidate-$candidate-q4" \
    --output-trace "data/eval/p0a5_student_candidate_${candidate}_q4_gate300.jsonl" \
    --audit "reports/audit/gate_p0a5_student_candidate_${candidate}_q4_gate300_eval.json"
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config "$CONFIG" \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "data/eval/p0a5_student_candidate_${candidate}_q4_gate300.jsonl" \
    --candidate-name "p0a5-student-candidate-$candidate-q4" \
    --output "reports/audit/gate_p0a5_student_candidate_${candidate}_gate300.json"
}

imatrix_corpus() {
  require_status reports/audit/gate_p0a5_distill.json passed
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source data/capability_v2/distill_train.jsonl \
    --output data/capability_v2/imatrix_calibration.txt \
    --audit reports/audit/gate_p0a5_imatrix_calibration.json \
    --stratify-key dataset_key \
    --stratum gsm8k --stratum opencodeinstruct --stratum cmmlu \
    --rows-per-stratum 128 --rows-per-source 384 --seed 20260729
}

student_quantize() {
  local candidate="${1:-1}"
  [[ "$candidate" == "1" || "$candidate" == "2" ]] || {
    echo "Candidate must be 1 or 2" >&2
    return 2
  }
  require_status "reports/audit/gate_p0a5_merge_student_candidate_$candidate.json" passed
  require_status reports/audit/gate_p0a5_imatrix_calibration.json passed
  bash scripts/run_with_memory_guard.sh \
    "$PYTHON_BIN" scripts/prepare_p0a5_quantized_student.py \
    --candidate "$candidate" \
    --gpu "${P0A5_IMATRIX_GPU:-0}" \
    --chunks "${P0A5_IMATRIX_CHUNKS:-170}"
}

baseline_plan() {
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port 8001 \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ \
    --quantization awq --tensor-parallel-size 4 \
    --served-model-name "$BASELINE_MODEL_ID" \
    --dry-run
}

baseline_serve() {
  require_status reports/audit/gate_p0a5_protocol.json passed
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port 8001 \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ \
    --quantization awq --tensor-parallel-size 4 \
    --served-model-name "$BASELINE_MODEL_ID"
}

student_plan() {
  local candidate="${1:-1}"
  local model="models/quantized/p0a5-student-candidate-$candidate-q4_k_m.gguf"
  require_status "reports/audit/gate_p0a5_quantize_student_candidate_$candidate.json" passed
  require_file "$model"
  cat <<EOF
CUDA_VISIBLE_DEVICES=${P0A5_STUDENT_GPU:-0} \
$ROOT/external/llama.cpp/build/bin/llama-server \
  --model $ROOT/$model \
  --alias $STUDENT_MODEL_ID \
  --host 127.0.0.1 --port 18450 \
  --ctx-size 1536 --threads 8 --parallel 1 \
  --batch-size 32 --ubatch-size 16 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --n-gpu-layers all --no-repack \
  --cache-ram 0 --no-cache-idle-slots \
  --reasoning off --reasoning-format none
EOF
}

student_serve() {
  local candidate="${1:-1}"
  local model="models/quantized/p0a5-student-candidate-$candidate-q4_k_m.gguf"
  local server="$ROOT/external/llama.cpp/build/bin/llama-server"
  require_status "reports/audit/gate_p0a5_quantize_student_candidate_$candidate.json" passed
  require_file "$model"
  require_file "$server"
  exec env CUDA_VISIBLE_DEVICES="${P0A5_STUDENT_GPU:-0}" \
    "$server" \
    --model "$ROOT/$model" \
    --alias "$STUDENT_MODEL_ID" \
    --host 127.0.0.1 --port 18450 \
    --ctx-size 1536 --threads 8 --parallel 1 \
    --batch-size 32 --ubatch-size 16 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --n-gpu-layers all --no-repack \
    --cache-ram 0 --no-cache-idle-slots \
    --reasoning off --reasoning-format none
}

gate300_pipeline() {
  local candidate="${1:-1}"
  require_status "reports/audit/gate_p0a5_quantize_student_candidate_$candidate.json" passed
  bash scripts/run_p0a5_gate300_pipeline.sh "$candidate"
}

status() {
  local path
  for path in \
    reports/audit/gate_p0a5_data.json \
    reports/audit/gate_p0a5_protocol.json \
    reports/audit/gate_p0a5_teacher_preflight.json \
    reports/audit/gate_p0a5_train_teacher.json \
    reports/audit/gate_p0a5_distill.json \
    reports/audit/gate_p0a5_train_student_candidate_1.json \
    reports/audit/gate_p0a5_merge_student_candidate_1.json \
    reports/audit/gate_p0a5_imatrix_calibration.json \
    reports/audit/gate_p0a5_quantize_student_candidate_1.json \
    reports/audit/gate_p0a5_baseline14b_gate300_eval.json \
    reports/audit/gate_p0a5_student_candidate_1_q4_gate300_eval.json \
    reports/audit/gate_p0a5_student_candidate_1_gate300.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], d.get("status"))' \
        "$path"
    else
      echo "$path missing"
    fi
  done
}

help_text() {
  cat <<'EOF'
P0-A5 single-gate capability pipeline

CPU preparation:
  data-download              Download three OpenCodeInstruct shards and COIG-CQIA.
  data-build                 Build 36,673 train, 2,200 internal validation and gate300.
  preflight                  Run protocol, Teacher, Student and distillation dry-runs.
  status                     Print current audit status.

GPU stages (run manually):
  teacher-train              Four-GPU BF16 14B ZeRO-3 LoRA training.
  teacher-eval [600|800]     Evaluate one frozen Teacher checkpoint on 2,200 internal rows.
  teacher-select             Select the lower-loss 600/800 checkpoint and publish it.
  teacher-plan               Print the four-GPU Teacher serving command.
  teacher-serve              Serve BF16 14B + P0-A5 LoRA on port 8000.
  distill-generate           Generate verified Student targets through the Teacher endpoint.
  student-preflight [1|2]    Validate one preregistered Student plan.
  student-train [1|2]        Four-GPU shared Student LoRA with Math KL preservation.
  student-merge [1|2]        Merge the selected shared LoRA into the v1 base.
  imatrix-corpus             Build train-only Q4_K_M calibration text.
  student-quantize [1|2]     Convert HF, compute GPU imatrix and create Q4_K_M.
  baseline-plan              Print the frozen four-GPU AWQ baseline service.
  baseline-serve             Serve frozen 14B AWQ on port 8001.
  student-plan [1|2]         Print the Q4_K_M + Q8 KV Student service.
  student-serve [1|2]        Serve Q4_K_M + Q8 KV Student on port 18450.
  gate300-pipeline [1|2]     Run baseline and Student services/evaluations in order.
  baseline-gate              Evaluate frozen 14B AWQ on the single 300-item gate.
  student-gate [1|2]         Evaluate Q4 Student and compute 78%/82% decisions.

Memory validation and the formal 13,065-item run are only authorized after a
Q4_K_M + Q8 KV candidate receives recommended_full=true.
EOF
}

case "${1:-help}" in
  data-download) data_download ;;
  data-build) data_build ;;
  preflight) preflight ;;
  teacher-train) teacher_train ;;
  teacher-eval) teacher_eval "${2:-}" ;;
  teacher-select) teacher_select ;;
  teacher-plan) teacher_plan ;;
  teacher-serve) teacher_serve ;;
  distill-generate) distill_generate ;;
  student-preflight) student_preflight "${2:-1}" ;;
  student-train) student_train "${2:-1}" ;;
  student-merge) student_merge "${2:-1}" ;;
  imatrix-corpus) imatrix_corpus ;;
  student-quantize) student_quantize "${2:-1}" ;;
  baseline-plan) baseline_plan ;;
  baseline-serve) baseline_serve ;;
  student-plan) student_plan "${2:-1}" ;;
  student-serve) student_serve "${2:-1}" ;;
  gate300-pipeline) gate300_pipeline "${2:-1}" ;;
  baseline-gate) baseline_gate ;;
  student-gate) student_gate "${2:-1}" ;;
  status) status ;;
  help|-h|--help) help_text ;;
  *)
    echo "Unknown command: $1" >&2
    help_text >&2
    exit 2
    ;;
esac
