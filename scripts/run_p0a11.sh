#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A11_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A11_SERVE_GPU:-0}"
PORT="${P0A11_PORT:-18465}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
NLP_DIR="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
AUDIT_ROOT="reports/audit/p0a11"

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

validate_gpu_group() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[x.strip() for x in sys.argv[1].split(',') if x.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(x.isdigit() for x in values):
 raise SystemExit(f"P0-A11 requires four distinct GPU ids: {values}")
print("P0-A11 GPU group:",','.join(values))
PY
}

domain_value() {
  local domain="$1" key="$2"
  case "$domain:$key" in
    math:max) echo 128 ;; math:checkpoint) echo 64 ;; math:lr) echo 0.000001 ;;
    math:rank) echo 8 ;; math:alpha) echo 16 ;; math:gain) echo 0.02 ;;
    math:rows) echo 300 ;; math:data) echo data/p0a11/math_train.jsonl ;;
    code:max) echo 500 ;; code:checkpoint) echo 250 ;; code:lr) echo 0.000003 ;;
    code:rank) echo 16 ;; code:alpha) echo 32 ;; code:gain) echo 0.03 ;;
    code:rows) echo 814 ;; code:data) echo data/p0a11/code_train.jsonl ;;
    *) echo "Invalid domain setting: $domain:$key" >&2; return 2 ;;
  esac
}

output_dir() { echo "models/checkpoints/p0a11/$1-specialist"; }

require_checkpoint() {
  local domain="$1" step="$2" dir
  dir="$(output_dir "$domain")/checkpoint-$step"
  require_file "$dir/trainer_state.json"
  require_file "$dir/adapter_config.json"
  [[ -s "$dir/adapter_model.safetensors" || -s "$dir/adapter_model.bin" ]] || {
    echo "Missing adapter weights: $dir" >&2; return 1;
  }
}

wait_endpoint() {
  local pid="$1" ids="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "Service exited before readiness" >&2; return 1; }
    if "$PYTHON_BIN" - "$ENDPOINT" "$ids" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 actual={str(x.get('id')) for x in json.load(response).get('data',[])}
needed=set(sys.argv[2].split(','))
raise SystemExit(0 if needed.issubset(actual) else 1)
PY
    then return 0; fi
    sleep 2
  done
  echo "Timed out waiting for $ENDPOINT" >&2
  return 1
}

prepare() {
  "$PYTHON_BIN" model_compression/build_p0a11_data.py prepare
  require_status reports/audit/gate_p0a11_data.json prepared,passed
}

mine_math() {
  require_status reports/audit/gate_p0a11_data.json prepared,passed
  if [[ -f reports/audit/gate_p0a11_math_mining.json ]]; then
    if require_status reports/audit/gate_p0a11_math_mining.json passed; then
      echo "P0-A11 Math mining already complete."
      return 0
    fi
  fi
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a11-base \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a11_math_mining_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a11-base
  "$PYTHON_BIN" scripts/mine_p0a11_math.py \
    --endpoint "$ENDPOINT" --model-id p0a11-base --workers "${P0A11_MINE_WORKERS:-8}"
  require_status reports/audit/gate_p0a11_math_mining.json passed
  cleanup
  trap - EXIT INT TERM
}

finalize() {
  require_status reports/audit/gate_p0a11_math_mining.json passed
  "$PYTHON_BIN" model_compression/build_p0a11_data.py finalize
  require_status reports/audit/gate_p0a11_data.json passed
}

preflight_domain() {
  local domain="$1" max checkpoint lr rank alpha data dir
  max="$(domain_value "$domain" max)"; checkpoint="$(domain_value "$domain" checkpoint)"
  lr="$(domain_value "$domain" lr)"; rank="$(domain_value "$domain" rank)"
  alpha="$(domain_value "$domain" alpha)"; data="$(domain_value "$domain" data)"
  dir="$(output_dir "$domain")"
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$data" --output-dir "$dir" \
    --audit "reports/audit/gate_p0a11_${domain}_preflight.json" \
    --max-steps "$max" --checkpoint-steps "$checkpoint" --focus-domain "$domain" \
    --learning-rate "$lr" --lora-rank "$rank" --lora-alpha "$alpha" --dry-run
  require_status "reports/audit/gate_p0a11_${domain}_preflight.json" dry_run_passed
}

preflight() {
  require_status reports/audit/gate_p0a11_data.json passed
  require_dir "$BASE_DIR"
  require_dir "$NLP_DIR"
  require_status reports/audit/p0a10/nlp_selection.json passed
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('reports/audit/p0a10/nlp_selection.json').read_text())
if d.get('selected_step') != 136:
 raise SystemExit('Frozen P0-A10 NLP selection is not step 136')
print('Frozen NLP guard passed: step=136')
PY
  preflight_domain math
  preflight_domain code
}

train_domain() {
  local domain="$1" max checkpoint lr rank alpha data dir audit status
  max="$(domain_value "$domain" max)"; checkpoint="$(domain_value "$domain" checkpoint)"
  lr="$(domain_value "$domain" lr)"; rank="$(domain_value "$domain" rank)"
  alpha="$(domain_value "$domain" alpha)"; data="$(domain_value "$domain" data)"
  dir="$(output_dir "$domain")"; audit="reports/audit/gate_p0a11_train_${domain}.json"
  preflight_domain "$domain"
  if [[ -f "$audit" ]]; then
    status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$audit")"
    if [[ "$status" == passed ]]; then
      require_checkpoint "$domain" "$checkpoint"; require_checkpoint "$domain" "$max"
      echo "P0-A11 $domain training already complete."
      return 0
    fi
  fi
  local resume=()
  if [[ -d "$dir" ]] && find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
    --model-dir "$BASE_DIR" --train-data "$data" --output-dir "$dir" --audit "$audit" \
    --max-steps "$max" --checkpoint-steps "$checkpoint" --focus-domain "$domain" \
    --learning-rate "$lr" --lora-rank "$rank" --lora-alpha "$alpha" \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    "${resume[@]}"
  require_status "$audit" passed
  require_checkpoint "$domain" "$checkpoint"; require_checkpoint "$domain" "$max"
}

train_all() {
  validate_gpu_group
  train_domain math
  train_domain code
}

evaluate_once() {
  local domain="$1" model="$2" rows name
  rows="$(domain_value "$domain" rows)"
  name="${model//-/_}_${domain}"
  if [[ -f "$AUDIT_ROOT/${name}.json" ]]; then
    require_status "$AUDIT_ROOT/${name}.json" passed
    return 0
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --domain "$domain" --manifest "data/p0a11/${domain}_validation.jsonl" \
    --endpoint "$ENDPOINT" --model-id "$model" --candidate-name "$model" \
    --expected-rows "$rows" --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" \
    --audit "$AUDIT_ROOT/${name}.json"
}

select_domain() {
  local domain="$1" first second gain
  first="$(domain_value "$domain" checkpoint)"; second="$(domain_value "$domain" max)"
  gain="$(domain_value "$domain" gain)"
  "$PYTHON_BIN" scripts/select_p0a11_domain.py \
    --domain "$domain" --steps "$first,$second" \
    --base-audit "$AUDIT_ROOT/p0a11_base_${domain}.json" \
    --candidate "$first=$AUDIT_ROOT/p0a11_${domain}_${first}_${domain}.json" \
    --candidate "$second=$AUDIT_ROOT/p0a11_${domain}_${second}_${domain}.json" \
    --minimum-gain "$gain" --output "$AUDIT_ROOT/${domain}_selection.json"
}

run_gate300_once() {
  local math_step="$1" code_step="$2"
  local trace="data/eval/p0a11_final_router_hf_gate300.jsonl"
  local eval_audit="reports/audit/gate_p0a11_final_router_hf_gate300_eval.json"
  local retention="reports/audit/gate_p0a11_final_router_hf_gate300_retention.json"
  if [[ -e "$trace" || -e "$eval_audit" || -e "$retention" ]]; then
    echo "P0-A11 gate300 artifacts already exist; refusing a repeated run." >&2
    return 1
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a9_router_gate.py \
    --endpoint "$ENDPOINT" --model-id-math "p0a11-math-$math_step" \
    --model-id-code "p0a11-code-$code_step" --model-id-nlp p0a11-nlp-136 \
    --candidate-name p0a11-final-router-hf --output-trace "$trace" --audit "$eval_audit"
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$trace" --candidate-name p0a11-final-router-hf --output "$retention"
}

validation() {
  local domain first second selection_rc failed=0
  require_status reports/audit/gate_p0a11_train_math.json passed
  require_status reports/audit/gate_p0a11_train_code.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  for domain in math code; do
    first="$(domain_value "$domain" checkpoint)"; second="$(domain_value "$domain" max)"
    require_checkpoint "$domain" "$first"; require_checkpoint "$domain" "$second"
  done
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a11-base \
    --lora-module "p0a11-math-64=$ROOT/$(output_dir math)/checkpoint-64" \
    --lora-module "p0a11-math-128=$ROOT/$(output_dir math)/checkpoint-128" \
    --lora-module "p0a11-code-250=$ROOT/$(output_dir code)/checkpoint-250" \
    --lora-module "p0a11-code-500=$ROOT/$(output_dir code)/checkpoint-500" \
    --lora-module "p0a11-nlp-136=$ROOT/$NLP_DIR" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a11_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a11-base,p0a11-math-64,p0a11-math-128,p0a11-code-250,p0a11-code-500,p0a11-nlp-136"
  for domain in math code; do
    first="$(domain_value "$domain" checkpoint)"; second="$(domain_value "$domain" max)"
    evaluate_once "$domain" p0a11-base
    evaluate_once "$domain" "p0a11-$domain-$first"
    evaluate_once "$domain" "p0a11-$domain-$second"
    set +e
    select_domain "$domain"
    selection_rc=$?
    set -e
    [[ "$selection_rc" -eq 0 ]] || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "P0-A11 train-only selection failed; gate300 remains closed." >&2
    cleanup; trap - EXIT INT TERM
    return 1
  fi
  local math_step code_step
  math_step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a11/math_selection.json"))["selected_step"])')"
  code_step="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a11/code_selection.json"))["selected_step"])')"
  run_gate300_once "$math_step" "$code_step"
  local gate_rc=$?
  cleanup; trap - EXIT INT TERM
  return "$gate_rc"
}

status() {
  local path
  for path in reports/audit/gate_p0a11_data.json \
    reports/audit/gate_p0a11_math_mining.json \
    reports/audit/gate_p0a11_math_preflight.json reports/audit/gate_p0a11_train_math.json \
    reports/audit/gate_p0a11_code_preflight.json reports/audit/gate_p0a11_train_code.json \
    reports/audit/p0a11/math_selection.json reports/audit/p0a11/code_selection.json \
    reports/audit/gate_p0a11_final_router_hf_gate300_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("ratios","")))' "$path"
    else
      echo "$path missing"
    fi
  done
}

case "${1:-help}" in
  prepare) prepare ;;
  mine-math) mine_math ;;
  finalize) finalize ;;
  preflight) preflight ;;
  train-math) validate_gpu_group; train_domain math ;;
  train-code) validate_gpu_group; train_domain code ;;
  train-all) train_all ;;
  validation) validation ;;
  auto) prepare; mine_math; finalize; preflight; train_all; validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a11.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a11_data.py \
      model_compression/train_p0a6_student.py scripts/mine_p0a11_math.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a11_domain.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a11.sh <prepare|mine-math|finalize|preflight|train-math|train-code|train-all|validation|auto|status|structural-check>"
    ;;
esac
