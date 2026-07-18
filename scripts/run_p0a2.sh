#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
P0A2_GPUS="${P0A2_GPUS:-0,1,2,3}"
P0A2_CONFIG="${P0A2_CONFIG:-configs/p0a2_recovery.json}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-external/llama.cpp}"

BASE_MODEL="models/pretrained/deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_DATA="data/distill/p0a2_recovery_train.jsonl"
VALIDATION_DATA="data/distill/p0a2_recovery_validation.jsonl"
ADAPTER_DIR="models/adapters/p0a2_deepseek_recovery"
MERGED_DIR="models/checkpoints/p0a2-deepseek-recovery-merged"
F16_GGUF="models/quantized/p0a2-deepseek-recovery-f16.gguf"
IMATRIX="models/quantized/p0a2-deepseek-recovery-imatrix.gguf"
Q2_GGUF="models/quantized/p0a2-deepseek-recovery-q2_k_s.gguf"

require_runtime() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python runtime: $PYTHON_BIN" >&2; exit 1; }
  [[ -d "$BASE_MODEL" ]] || { echo "Missing recovery base: $BASE_MODEL" >&2; exit 1; }
}

gpu_count() {
  "$PYTHON_BIN" - "$P0A2_GPUS" <<'PY'
import sys
values = [item.strip() for item in sys.argv[1].replace(" ", ",").split(",") if item.strip()]
if not values:
    raise SystemExit("P0A2_GPUS selects no devices")
print(len(values))
PY
}

build_data() {
  "$PYTHON_BIN" model_compression/build_p0a2_recovery_data.py --config "$P0A2_CONFIG"
}

preflight() {
  require_runtime
  build_data
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
g0 = json.loads((root / "reports/audit/gate_g0_capmem.json").read_text(encoding="utf-8"))
memory = json.loads((root / "reports/audit/g0_memory_deepseek_r1_1p5b_q2_k_s.json").read_text(encoding="utf-8"))
data = json.loads((root / "reports/audit/gate_p0a2_recovery_data.json").read_text(encoding="utf-8"))
errors = []
if g0.get("status") != "failed" or int(g0.get("feasible_candidate_count", -1)) != 0:
    errors.append("P0-A2 requires the frozen initial G0 failure")
if memory.get("status") != "passed" or float(memory.get("peak_total_memory_mb_decimal", 1e9)) > 1500:
    errors.append("DeepSeek Q2 memory-safe premise is missing")
if data.get("status") != "passed":
    errors.append("P0-A2 recovery data gate did not pass")
if data.get("formal_test_reference_count") or data.get("formal_humaneval_prompt_overlap_count"):
    errors.append("Formal-test leakage was detected")
if data.get("train_validation_group_overlap_count"):
    errors.append("Training and validation groups overlap")
if errors:
    raise SystemExit("; ".join(errors))
print(
    "P0-A2 preflight passed: "
    f"train={data['outputs']['train']['rows']} "
    f"validation={data['outputs']['validation']['rows']} "
    f"baseline_peak_mb={memory['peak_total_memory_mb_decimal']}"
)
PY
}

upper_bound() {
  require_runtime
  [[ -f "$VALIDATION_DATA" ]] || build_data
  CUDA_VISIBLE_DEVICES="${P0A2_UPPER_BOUND_GPU:-${P0A2_GPUS%%,*}}" \
    "$PYTHON_BIN" scripts/evaluate_p0a2_recovery.py \
      --local-model-dir "$BASE_MODEL" \
      --validation-data "$VALIDATION_DATA" \
      --output-trace data/eval/p0a2_deepseek_upper_bound.jsonl \
      --audit reports/audit/gate_p0a2_deepseek_upper_bound.json \
      --max-new-tokens-map cmmlu=256,gsm8k=512,humaneval=512 \
      --close-reasoning-prefix \
      --device cuda \
      --dtype bfloat16
}

upper_bound_smoke() {
  require_runtime
  [[ -f "$VALIDATION_DATA" ]] || build_data
  CUDA_VISIBLE_DEVICES="${P0A2_UPPER_BOUND_GPU:-${P0A2_GPUS%%,*}}" \
    "$PYTHON_BIN" scripts/evaluate_p0a2_recovery.py \
      --local-model-dir "$BASE_MODEL" \
      --validation-data "$VALIDATION_DATA" \
      --output-trace reports/audit/p0a2_deepseek_upper_bound_smoke.jsonl \
      --audit reports/audit/gate_p0a2_deepseek_upper_bound_smoke.json \
      --sample-limit-per-dataset 1 \
      --max-new-tokens-map cmmlu=256,gsm8k=512,humaneval=512 \
      --close-reasoning-prefix \
      --device cuda \
      --dtype bfloat16
}

train_recovery() {
  require_runtime
  [[ -x "$TORCHRUN_BIN" ]] || { echo "Missing torchrun: $TORCHRUN_BIN" >&2; exit 1; }
  [[ -f "$TRAIN_DATA" && -f "$VALIDATION_DATA" ]] || build_data
  local processes
  processes="$(gpu_count)"
  local targets=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)
  local target_args=()
  local target
  for target in "${targets[@]}"; do
    target_args+=(--target-module "$target")
  done
  CUDA_VISIBLE_DEVICES="$P0A2_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node "$processes" \
    model_compression/train_cedd_repair.py \
      --student-init "$BASE_MODEL" \
      --distill-data "$TRAIN_DATA" \
      --distill-repeat 0 \
      --repair-repeat 0 \
      --capability-repeat 0 \
      --capability-rehearsal-jsonl "$TRAIN_DATA" \
      --capability-sample-limit 0 \
      --capability-jsonl-repeat 1 \
      --generation-validation-jsonl "$VALIDATION_DATA" \
      --generation-validation-sample-limit 170 \
      --generation-validation-examples-per-group 1 \
      --generation-validation-max-new-tokens 256 \
      --generation-validation-code-timeout-sec 5 \
      --validation-fraction 0.05 \
      --validation-seed 202606 \
      --validation-selection-metric generation \
      --restore-best-validation \
      --min-generation-validation-improvement 0.01 \
      --max-train-examples-per-validation-group 16 \
      --min-validation-group-count 100 \
      --eval-every 16 \
      --early-stopping-patience 3 \
      --min-optimizer-steps 64 \
      --max-steps 128 \
      --epochs 1 \
      --batch-size 1 \
      --grad-accum-steps 8 \
      --learning-rate 2e-5 \
      --weight-decay 0.01 \
      --lr-scheduler cosine \
      --warmup-ratio 0.05 \
      --parent-preservation-weight 0.1 \
      --max-length 1024 \
      --lora-rank 16 \
      --lora-alpha 32 \
      --lora-dropout 0.05 \
      "${target_args[@]}" \
      --dtype bfloat16 \
      --close-reasoning-prefix \
      --gradient-checkpointing \
      --stage-name p0a2_deepseek_recovery \
      --output-dir "$ADAPTER_DIR" \
      --audit reports/audit/gate_p0a2_deepseek_recovery_train.json
}

evaluate_adapter() {
  require_runtime
  [[ -d "$ADAPTER_DIR" ]] || { echo "Missing trained adapter: $ADAPTER_DIR" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${P0A2_EVAL_GPU:-${P0A2_GPUS%%,*}}" \
    "$PYTHON_BIN" scripts/evaluate_p0a2_recovery.py \
      --local-model-dir "$BASE_MODEL" \
      --adapter-path "$ADAPTER_DIR" \
      --validation-data "$VALIDATION_DATA" \
      --output-trace data/eval/p0a2_deepseek_recovered_dev.jsonl \
      --audit reports/audit/gate_p0a2_deepseek_recovered_dev.json \
      --close-reasoning-prefix \
      --device cuda \
      --dtype bfloat16
}

export_merged() {
  require_runtime
  [[ -d "$ADAPTER_DIR" ]] || { echo "Missing trained adapter: $ADAPTER_DIR" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${P0A2_EXPORT_GPU:-${P0A2_GPUS%%,*}}" \
    "$PYTHON_BIN" model_compression/export_merged_hf.py \
      --local-model-dir "$BASE_MODEL" \
      --adapter-path "$ADAPTER_DIR" \
      --output-dir "$MERGED_DIR" \
      --audit reports/audit/gate_p0a2_deepseek_merged_hf.json \
      --dtype bfloat16
}

build_imatrix() {
  [[ -d "$MERGED_DIR" ]] || { echo "Missing merged model: $MERGED_DIR" >&2; exit 1; }
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source "$TRAIN_DATA" \
    --output data/distill/p0a2_imatrix_calibration.txt \
    --audit reports/audit/gate_p0a2_imatrix_calibration.json \
    --rows-per-source 768 \
    --seed 202606
  if [[ ! -f "$F16_GGUF" ]]; then
    "$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
      --outfile "$F16_GGUF" --outtype f16
  fi
  local partial="${IMATRIX}.partial"
  rm -f "$partial"
  "$LLAMA_CPP_DIR/build/bin/llama-imatrix" \
    --model "$F16_GGUF" \
    --file data/distill/p0a2_imatrix_calibration.txt \
    --output "$partial" \
    --chunks 64 \
    --ctx-size 512 \
    --threads 16 \
    --no-ppl \
    --output-frequency 16
  mv "$partial" "$IMATRIX"
}

quantize() {
  [[ -d "$MERGED_DIR" ]] || { echo "Missing merged model: $MERGED_DIR" >&2; exit 1; }
  [[ -f "$IMATRIX" ]] || build_imatrix
  "$PYTHON_BIN" scripts/prepare_edge_gguf.py \
    --merged-hf-dir "$MERGED_DIR" \
    --llama-cpp-dir "$LLAMA_CPP_DIR" \
    --f16-gguf "$F16_GGUF" \
    --quantized-gguf "$Q2_GGUF" \
    --quant-type Q2_K_S \
    --imatrix "$IMATRIX" \
    --max-quantized-bytes 900000000 \
    --audit reports/audit/gate_p0a2_deepseek_q2_prepare.json
}

g0_reentry() {
  [[ -f "$Q2_GGUF" ]] || { echo "Missing recovered Q2 model: $Q2_GGUF" >&2; exit 1; }
  "$PYTHON_BIN" scripts/run_g0_capmem.py \
    --config configs/g0_capmem_candidates.json \
    --candidate p0a2-deepseek-recovery-q2-k-s \
    --execute-memory \
    --execute-capability-smoke \
    --output reports/audit/gate_g0_capmem_p0a2.json \
    --require-feasible
}

checks() {
  "$PYTHON_BIN" scripts/validate_project_structure.py
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'
  preflight
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p0a2.sh <command>

Commands:
  build-data         Freeze train-only and selection-only recovery JSONL files.
  preflight          Validate the failed-G0 premise, memory premise and data isolation.
  upper-bound-smoke  Evaluate one Dev sample per task on the unquantized DeepSeek base.
  upper-bound        Evaluate the full independent Dev set on the unquantized base.
  train              Train one guarded DeepSeek recovery LoRA on four GPUs by default.
  evaluate-adapter   Evaluate the selected LoRA on the same frozen Dev protocol.
  export             Merge the selected LoRA into a Hugging Face checkpoint.
  build-imatrix      Build train-only importance calibration for the merged model.
  quantize           Build the deployment Q2_K_S GGUF (artifact pre-gate <=900MB).
  g0-reentry         Rerun matched capability and 20+100-request peak-memory gates.
  checks             Run structure checks, unit tests and P0-A2 preflight.
EOF
}

case "${1:-}" in
  build-data) build_data ;;
  preflight) preflight ;;
  upper-bound-smoke) upper_bound_smoke ;;
  upper-bound) upper_bound ;;
  train) train_recovery ;;
  evaluate-adapter) evaluate_adapter ;;
  export) export_merged ;;
  build-imatrix) build_imatrix ;;
  quantize) quantize ;;
  g0-reentry) g0_reentry ;;
  checks) checks ;;
  *) usage; exit 2 ;;
esac
