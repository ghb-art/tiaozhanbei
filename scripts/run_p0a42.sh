#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A42_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A42_SERVE_GPU:-0}"
PORT="${P0A42_PORT:-18506}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE="models/pretrained/Qwen--Qwen3-1.7B"
AUDIT_ROOT="reports/audit/p0a42"

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

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[x.strip() for x in sys.argv[1].split(',') if x.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(x.isdigit() for x in values):
 raise SystemExit(f'P0-A42 requires four distinct GPU ids: {values}')
print('P0-A42 GPU group:',','.join(values))
PY
}

domain_value() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
d=json.load(open('configs/p0a42_one_round.json'))['training'][sys.argv[1]]
print(d[sys.argv[2]])
PY
}

checkpoint_ok() {
  local domain="$1" step="$2" dir="models/checkpoints/p0a42/$1/checkpoint-$2"
  [[ -f "$dir/trainer_state.json" && -f "$dir/adapter_config.json" ]] || return 1
  [[ -s "$dir/adapter_model.safetensors" || -s "$dir/adapter_model.bin" ]]
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a42_data.py
  require_status reports/audit/gate_p0a42_data.json passed
}

preflight_domain() {
  local domain="$1" rank alpha lr max checkpoint multiplier
  rank="$(domain_value "$domain" rank)"; alpha="$(domain_value "$domain" alpha)"
  lr="$(domain_value "$domain" learning_rate)"; max="$(domain_value "$domain" max_steps)"
  checkpoint="$(domain_value "$domain" checkpoint_steps)"; multiplier=1
  if [[ "$domain" == nlp ]]; then multiplier="$(domain_value nlp answer_token_weight_multiplier)"; fi
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE" --train-data "data/p0a42/${domain}_train.jsonl" \
    --output-dir "models/checkpoints/p0a42/$domain" \
    --audit "reports/audit/gate_p0a42_${domain}_preflight.json" \
    --max-steps "$max" --checkpoint-steps "$checkpoint" \
    --focus-domain "$domain" --learning-rate "$lr" \
    --lora-rank "$rank" --lora-alpha "$alpha" \
    --mcq-answer-token-weight-multiplier "$multiplier" --dry-run
  require_status "reports/audit/gate_p0a42_${domain}_preflight.json" dry_run_passed
}

preflight() {
  validate_gpus
  require_status reports/audit/gate_p0a42_data.json passed
  [[ -d "$BASE" ]] || { echo "Missing P0-A42 original base" >&2; return 1; }
  preflight_domain math
  preflight_domain code
  preflight_domain nlp
}

train_domain() {
  local domain="$1" rank alpha lr max checkpoint multiplier audit output resume=()
  validate_gpus; require_status reports/audit/gate_p0a42_data.json passed
  rank="$(domain_value "$domain" rank)"; alpha="$(domain_value "$domain" alpha)"
  lr="$(domain_value "$domain" learning_rate)"; max="$(domain_value "$domain" max_steps)"
  checkpoint="$(domain_value "$domain" checkpoint_steps)"; multiplier=1
  if [[ "$domain" == nlp ]]; then multiplier="$(domain_value nlp answer_token_weight_multiplier)"; fi
  audit="reports/audit/gate_p0a42_train_${domain}.json"
  output="models/checkpoints/p0a42/$domain"
  preflight_domain "$domain"
  if [[ -f "$audit" ]] && [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$audit")" == passed ]]; then
    checkpoint_ok "$domain" "$checkpoint" && checkpoint_ok "$domain" "$max"
    echo "P0-A42 $domain training already complete."
    return 0
  fi
  if [[ -d "$output" ]] && find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
      --model-dir "$BASE" --train-data "data/p0a42/${domain}_train.jsonl" \
      --output-dir "$output" --audit "$audit" \
      --max-steps "$max" --checkpoint-steps "$checkpoint" \
      --focus-domain "$domain" --learning-rate "$lr" \
      --lora-rank "$rank" --lora-alpha "$alpha" \
      --mcq-answer-token-weight-multiplier "$multiplier" \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
      "${resume[@]}"
  require_status "$audit" passed
  checkpoint_ok "$domain" "$checkpoint"; checkpoint_ok "$domain" "$max"
}

train_all() {
  train_domain math
  train_domain code
  train_domain nlp
}

wait_endpoint() {
  local pid="$1" required="$2" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$required" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
need={x for x in sys.argv[2].split(',') if x}
raise SystemExit(0 if need.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

evaluate_one() {
  local protocol="$1" domain="$2" manifest="$3" rows="$4" model_id="$5" label="$6" max_tokens="$7"
  local audit="$AUDIT_ROOT/${label}.json" trace="$AUDIT_ROOT/${label}_trace.jsonl"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol "$protocol" --domain "$domain" --manifest "$manifest" \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows "$rows" --workers 8 --thinking off --max-tokens "$max_tokens" \
    --timeout-sec 180 --code-timeout-sec 5 \
    --output-trace "$trace" --audit "$audit"
  require_status "$audit" passed
}

validation() {
  require_status reports/audit/gate_p0a42_train_math.json passed
  require_status reports/audit/gate_p0a42_train_code.json passed
  require_status reports/audit/gate_p0a42_train_nlp.json passed
  checkpoint_ok math 108; checkpoint_ok math 216
  checkpoint_ok code 96; checkpoint_ok code 192
  checkpoint_ok nlp 50; checkpoint_ok nlp 100
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a42-base \
    --lora-module "p0a42-current-code=$ROOT/models/checkpoints/p0a25/code-failure-repair/checkpoint-192" \
    --lora-module "p0a42-current-nlp=$ROOT/models/checkpoints/p0a10/nlp-specialist/checkpoint-136" \
    --lora-module "p0a42-math-108=$ROOT/models/checkpoints/p0a42/math/checkpoint-108" \
    --lora-module "p0a42-math-216=$ROOT/models/checkpoints/p0a42/math/checkpoint-216" \
    --lora-module "p0a42-code-96=$ROOT/models/checkpoints/p0a42/code/checkpoint-96" \
    --lora-module "p0a42-code-192=$ROOT/models/checkpoints/p0a42/code/checkpoint-192" \
    --lora-module "p0a42-nlp-50=$ROOT/models/checkpoints/p0a42/nlp/checkpoint-50" \
    --lora-module "p0a42-nlp-100=$ROOT/models/checkpoints/p0a42/nlp/checkpoint-100" \
    --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a42_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a42-base,p0a42-current-code,p0a42-current-nlp,p0a42-math-108,p0a42-math-216,p0a42-code-96,p0a42-code-192,p0a42-nlp-50,p0a42-nlp-100"
  evaluate_one p0a42_math math data/p0a10/math_validation.jsonl 256 p0a42-base base_math 512
  evaluate_one p0a42_math math data/p0a10/math_validation.jsonl 256 p0a42-math-108 math_108 512
  evaluate_one p0a42_math math data/p0a10/math_validation.jsonl 256 p0a42-math-216 math_216 512
  evaluate_one p0a42_code code data/p0a25/code_validation.jsonl 1000 p0a42-base base_code 768
  evaluate_one p0a42_code code data/p0a25/code_validation.jsonl 1000 p0a42-current-code current_code 768
  evaluate_one p0a42_code code data/p0a25/code_validation.jsonl 1000 p0a42-code-96 code_96 768
  evaluate_one p0a42_code code data/p0a25/code_validation.jsonl 1000 p0a42-code-192 code_192 768
  evaluate_one p0a42_nlp nlp data/p0a34/nlp_validation.jsonl 260 p0a42-base base_nlp 256
  evaluate_one p0a42_nlp nlp data/p0a34/nlp_validation.jsonl 260 p0a42-current-nlp current_nlp 256
  evaluate_one p0a42_nlp nlp data/p0a34/nlp_validation.jsonl 260 p0a42-nlp-50 nlp_50 256
  evaluate_one p0a42_nlp nlp data/p0a34/nlp_validation.jsonl 260 p0a42-nlp-100 nlp_100 256
  "$PYTHON_BIN" scripts/select_p0a42_domains.py
  require_status reports/audit/gate_p0a42_domain_selection.json passed
  cleanup; trap - EXIT INT TERM
}

guarded_validation() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a42.sh validation
}

gate300() {
  require_status reports/audit/gate_p0a42_domain_selection.json passed
  require_status reports/audit/gate_p0a5_baseline14b_gate300_eval.json passed
  local trace="data/eval/p0a42_one_round_gate300.jsonl"
  local eval_audit="reports/audit/gate_p0a42_one_round_gate300_eval.json"
  local retention="reports/audit/gate_p0a42_one_round_gate300_retention.json"
  if [[ -e "$trace" || -e "$eval_audit" || -e "$retention" ]]; then
    echo "P0-A42 single test artifacts already exist; repeated run refused." >&2
    return 1
  fi
  local selected math_path code_path nlp_path math_id code_id nlp_id
  selected="$($PYTHON_BIN - <<'PY'
import json
d=json.load(open('reports/audit/gate_p0a42_domain_selection.json'))['selected']
for domain in ('math','code','nlp'):
 print(domain+'\t'+d[domain]['adapter'])
PY
)"
  math_path="$(awk -F '\t' '$1=="math"{print $2}' <<<"$selected")"
  code_path="$(awk -F '\t' '$1=="code"{print $2}' <<<"$selected")"
  nlp_path="$(awk -F '\t' '$1=="nlp"{print $2}' <<<"$selected")"
  math_id=p0a42-final-math; code_id=p0a42-final-code; nlp_id=p0a42-final-nlp
  local loras=() required="p0a42-base"
  if [[ -n "$math_path" ]]; then loras+=(--lora-module "$math_id=$ROOT/$math_path"); required+=",$math_id"; else math_id=p0a42-base; fi
  if [[ -n "$code_path" ]]; then loras+=(--lora-module "$code_id=$ROOT/$code_path"); required+=",$code_id"; else code_id=p0a42-base; fi
  if [[ -n "$nlp_path" ]]; then loras+=(--lora-module "$nlp_id=$ROOT/$nlp_path"); required+=",$nlp_id"; else nlp_id=p0a42-base; fi
  mkdir -p logs runtime
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a42-base \
    "${loras[@]}" --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a42_gate300_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$required"
  "$PYTHON_BIN" scripts/evaluate_p0a42_gate300.py \
    --endpoint "$ENDPOINT" --model-id-math "$math_id" \
    --model-id-code "$code_id" --model-id-nlp "$nlp_id"
  set +e
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$trace" --candidate-name p0a42-one-round-router-hf \
    --output "$retention"
  local result=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$result"
}

guarded_gate300() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a42.sh gate300
}

status() {
  local p
  for p in reports/audit/gate_p0a42_data.json \
    reports/audit/gate_p0a42_train_math.json \
    reports/audit/gate_p0a42_train_code.json \
    reports/audit/gate_p0a42_train_nlp.json \
    reports/audit/gate_p0a42_domain_selection.json \
    reports/audit/gate_p0a42_one_round_gate300_eval.json \
    reports/audit/gate_p0a42_one_round_gate300_retention.json; do
    if [[ -f "$p" ]]; then
      "$PYTHON_BIN" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[1],d.get("status"),d.get("retention_ratios",d.get("selected","")))' "$p"
    else echo "$p missing"; fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train-math) train_domain math ;;
  train-code) train_domain code ;;
  train-nlp) train_domain nlp ;;
  train-all) train_all ;;
  validation) validation ;;
  guarded-validation) guarded_validation ;;
  gate300) gate300 ;;
  guarded-gate300) guarded_gate300 ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a42.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a42_data.py \
      model_compression/train_p0a6_student.py scripts/evaluate_p0a11_domain.py \
      scripts/select_p0a42_domains.py scripts/evaluate_p0a42_gate300.py
    ;;
  *) echo "Usage: bash scripts/run_p0a42.sh <data-build|preflight|train-math|train-code|train-nlp|train-all|guarded-validation|guarded-gate300|status|structural-check>" ;;
esac
