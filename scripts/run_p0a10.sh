#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A10_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A10_SERVE_GPU:-0}"
PORT="${P0A10_PORT:-18464}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
AUDIT_ROOT="reports/audit/p0a10"

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; return 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; return 1; }; }
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

domain_value() {
  local domain="$1" key="$2"
  case "$domain:$key" in
    math:max) echo 224 ;; math:checkpoint) echo 112 ;; math:lr) echo 0.00001 ;; math:focus) echo math ;;
    code:max) echo 256 ;; code:checkpoint) echo 128 ;; code:lr) echo 0.000005 ;; code:focus) echo code ;;
    nlp:max) echo 136 ;; nlp:checkpoint) echo 68 ;; nlp:lr) echo 0.00001 ;; nlp:focus) echo nlp_mixed_mcq ;;
    *) echo "Invalid domain setting: $domain:$key" >&2; return 2 ;;
  esac
}

validate_gpu_group() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
v=[x.strip() for x in sys.argv[1].split(',') if x.strip()]
if len(v)!=4 or len(set(v))!=4 or not all(x.isdigit() for x in v):
 raise SystemExit(f"P0-A10 requires four distinct GPU ids: {v}")
print("P0-A10 GPU group:",','.join(v))
PY
}

output_dir() { echo "models/checkpoints/p0a10/$1-specialist"; }

require_checkpoint() {
  local domain="$1" step="$2" dir
  dir="$(output_dir "$domain")/checkpoint-$step"
  require_file "$dir/trainer_state.json"
  require_file "$dir/adapter_config.json"
  [[ -s "$dir/adapter_model.safetensors" || -s "$dir/adapter_model.bin" ]] || {
    echo "Missing adapter weights: $dir" >&2; return 1;
  }
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a10_data.py
  require_status reports/audit/gate_p0a10_data.json passed
}

preflight_domain() {
  local domain="$1" max checkpoint lr focus dir
  max="$(domain_value "$domain" max)"; checkpoint="$(domain_value "$domain" checkpoint)"
  lr="$(domain_value "$domain" lr)"; focus="$(domain_value "$domain" focus)"
  dir="$(output_dir "$domain")"
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data data/p0a10/train.jsonl \
    --output-dir "$dir" --audit "reports/audit/gate_p0a10_${domain}_preflight.json" \
    --max-steps "$max" --checkpoint-steps "$checkpoint" \
    --focus-domain "$focus" --learning-rate "$lr" --dry-run
  require_status "reports/audit/gate_p0a10_${domain}_preflight.json" dry_run_passed
}

preflight() {
  require_status reports/audit/gate_p0a10_data.json passed
  require_dir "$BASE_DIR"
  preflight_domain math
  preflight_domain code
  preflight_domain nlp
}

train_domain() {
  local domain="$1" max checkpoint lr focus dir audit status
  max="$(domain_value "$domain" max)"; checkpoint="$(domain_value "$domain" checkpoint)"
  lr="$(domain_value "$domain" lr)"; focus="$(domain_value "$domain" focus)"
  dir="$(output_dir "$domain")"; audit="reports/audit/gate_p0a10_train_${domain}.json"
  preflight_domain "$domain"
  if [[ -f "$audit" ]]; then
    status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$audit")"
    if [[ "$status" == passed ]]; then
      require_checkpoint "$domain" "$checkpoint"; require_checkpoint "$domain" "$max"
      echo "P0-A10 $domain training already complete."
      return 0
    fi
  fi
  local resume=()
  if [[ -d "$dir" ]] && find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data data/p0a10/train.jsonl \
    --output-dir "$dir" --audit "$audit" \
    --max-steps "$max" --checkpoint-steps "$checkpoint" \
    --focus-domain "$focus" --learning-rate "$lr" \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume[@]}"
  require_status "$audit" passed
  require_checkpoint "$domain" "$checkpoint"; require_checkpoint "$domain" "$max"
}

train_all() {
  validate_gpu_group
  train_domain math
  train_domain code
  train_domain nlp
}

wait_endpoint() {
  local pid="$1" ids="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "Service exited before readiness" >&2; return 1; }
    if "$PYTHON_BIN" - "$ENDPOINT" "$ids" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as r:
 actual={str(x.get('id')) for x in json.load(r).get('data',[])}
need=set(sys.argv[2].split(','))
raise SystemExit(0 if need.issubset(actual) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

evaluate_once() {
  local domain="$1" model="$2" name
  name="${model//-/_}_${domain}"
  if [[ -f "$AUDIT_ROOT/${name}.json" ]]; then
    require_status "$AUDIT_ROOT/${name}.json" passed
    return 0
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a10_domain.py \
    --domain "$domain" --manifest "data/p0a10/${domain}_validation.jsonl" \
    --endpoint "$ENDPOINT" --model-id "$model" --candidate-name "$model" \
    --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" \
    --audit "$AUDIT_ROOT/${name}.json"
}

select_domain() {
  local domain="$1" first second gain
  case "$domain" in
    math) first=112; second=224; gain=0.02 ;;
    code) first=128; second=256; gain=0.03 ;;
    nlp) first=68; second=136; gain=0.03 ;;
  esac
  "$PYTHON_BIN" scripts/select_p0a10_domain.py \
    --domain "$domain" --steps "$first,$second" \
    --base-audit "$AUDIT_ROOT/p0a10_base_${domain}.json" \
    --candidate "$first=$AUDIT_ROOT/p0a10_${domain}_${first}_${domain}.json" \
    --candidate "$second=$AUDIT_ROOT/p0a10_${domain}_${second}_${domain}.json" \
    --minimum-gain "$gain" --output "$AUDIT_ROOT/${domain}_selection.json"
}

validation() {
  local domain first second selection_rc selection_failed=0
  for domain in math code nlp; do
    first="$(domain_value "$domain" checkpoint)"; second="$(domain_value "$domain" max)"
    require_checkpoint "$domain" "$first"; require_checkpoint "$domain" "$second"
  done
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a10-base \
    --lora-module "p0a10-math-112=$ROOT/$(output_dir math)/checkpoint-112" \
    --lora-module "p0a10-math-224=$ROOT/$(output_dir math)/checkpoint-224" \
    --lora-module "p0a10-code-128=$ROOT/$(output_dir code)/checkpoint-128" \
    --lora-module "p0a10-code-256=$ROOT/$(output_dir code)/checkpoint-256" \
    --lora-module "p0a10-nlp-68=$ROOT/$(output_dir nlp)/checkpoint-68" \
    --lora-module "p0a10-nlp-136=$ROOT/$(output_dir nlp)/checkpoint-136" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a10_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a10-base,p0a10-math-112,p0a10-math-224,p0a10-code-128,p0a10-code-256,p0a10-nlp-68,p0a10-nlp-136"
  for domain in math code nlp; do
    first="$(domain_value "$domain" checkpoint)"; second="$(domain_value "$domain" max)"
    evaluate_once "$domain" p0a10-base
    evaluate_once "$domain" "p0a10-$domain-$first"
    evaluate_once "$domain" "p0a10-$domain-$second"
    set +e
    select_domain "$domain"
    selection_rc=$?
    set -e
    if [[ "$selection_rc" -ne 0 ]]; then
      selection_failed=1
    fi
  done
  if [[ "$selection_failed" -ne 0 ]]; then
    echo "At least one P0-A10 domain selection failed; final gate300 is blocked." >&2
    cleanup; trap - EXIT INT TERM
    return 1
  fi
  local math_step code_step nlp_step
  math_step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a10/math_selection.json"))["selected_step"])')"
  code_step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a10/code_selection.json"))["selected_step"])')"
  nlp_step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a10/nlp_selection.json"))["selected_step"])')"
  "$PYTHON_BIN" scripts/evaluate_p0a9_router_gate.py \
    --endpoint "$ENDPOINT" \
    --model-id-math "p0a10-math-$math_step" \
    --model-id-code "p0a10-code-$code_step" \
    --model-id-nlp "p0a10-nlp-$nlp_step" \
    --candidate-name p0a10-final-router-hf \
    --output-trace data/eval/p0a10_final_router_hf_gate300.jsonl \
    --audit reports/audit/gate_p0a10_final_router_hf_gate300_eval.json
  set +e
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace data/eval/p0a10_final_router_hf_gate300.jsonl \
    --candidate-name p0a10-final-router-hf \
    --output reports/audit/gate_p0a10_final_router_hf_gate300_retention.json
  local gate_rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$gate_rc"
}

status() {
  local path
  for path in reports/audit/gate_p0a10_data.json \
    reports/audit/gate_p0a10_math_preflight.json reports/audit/gate_p0a10_train_math.json \
    reports/audit/gate_p0a10_code_preflight.json reports/audit/gate_p0a10_train_code.json \
    reports/audit/gate_p0a10_nlp_preflight.json reports/audit/gate_p0a10_train_nlp.json \
    reports/audit/p0a10/math_selection.json reports/audit/p0a10/code_selection.json \
    reports/audit/p0a10/nlp_selection.json reports/audit/gate_p0a10_final_router_hf_gate300_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; print(sys.argv[1],json.load(open(sys.argv[1])).get("status"))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train-math) validate_gpu_group; train_domain math ;;
  train-code) validate_gpu_group; train_domain code ;;
  train-nlp) validate_gpu_group; train_domain nlp ;;
  train-all) train_all ;;
  validation) validation ;;
  auto) data_build; preflight; train_all; validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a10.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a10_data.py \
      model_compression/train_p0a6_student.py scripts/evaluate_p0a10_domain.py \
      scripts/select_p0a10_domain.py
    ;;
  *) echo "Usage: bash scripts/run_p0a10.sh <data-build|preflight|train-math|train-code|train-nlp|train-all|validation|auto|status|structural-check>" ;;
esac
