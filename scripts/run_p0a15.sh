#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A15_SERVE_GPU:-0}"
PORT="${P0A15_PORT:-18469}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
MATH_HALF="models/adapters/p0a15/math-step64-scale-0p5"
MATH_FULL="models/checkpoints/p0a11/math-specialist/checkpoint-64"
CODE_DIR="models/checkpoints/p0a11/code-specialist/checkpoint-250"
NLP_DIR="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
AUDIT_ROOT="reports/audit/p0a15"

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

wait_endpoint() {
  local pid="$1" ids="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "Service exited before readiness" >&2; return 1; }
    if "$PYTHON_BIN" - "$ENDPOINT" "$ids" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 actual={str(x.get('id')) for x in json.load(response).get('data',[])}
raise SystemExit(0 if set(sys.argv[2].split(',')).issubset(actual) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a15_data.py build
  require_status reports/audit/gate_p0a15_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a15_data.json passed
  require_status reports/audit/p0a14/runtime_selection.json failed
  require_status reports/audit/p0a11/code_selection.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  require_dir "$BASE_DIR"; require_dir "$MATH_HALF"; require_dir "$MATH_FULL"
  require_dir "$CODE_DIR"; require_dir "$NLP_DIR"
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
half=json.load(open('models/adapters/p0a15/math-step64-scale-0p5/adapter_config.json'))
full=json.load(open('models/checkpoints/p0a11/math-specialist/checkpoint-64/adapter_config.json'))
if (half.get('r'),half.get('lora_alpha')) != (8,8): raise SystemExit('Half-scale adapter mismatch')
if (full.get('r'),full.get('lora_alpha')) != (8,16): raise SystemExit('Full-scale adapter mismatch')
rows=sum(1 for line in Path('data/p0a15/math_validation.jsonl').open() if line.strip())
if rows != 346: raise SystemExit(f'P0-A15 row count changed: {rows}')
report={'gate':'P0-A15-PREFLIGHT','status':'passed','validation_rows':rows,
 'candidate_scales':[0.5,1.0],'gate300_opened':False,'formal_full_opened':False}
Path('reports/audit/gate_p0a15_preflight.json').write_text(
 json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('P0-A15 preflight passed.')
PY
  require_status reports/audit/gate_p0a15_preflight.json passed
}

evaluate_once() {
  local model="$1" name="$2"
  local audit="$AUDIT_ROOT/${name}.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a15 --domain math --manifest data/p0a15/math_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model" --candidate-name "$name" \
    --expected-rows 346 --max-tokens 768 --thinking on \
    --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" --audit "$audit" \
    --workers "${P0A15_EVAL_WORKERS:-8}"
}

select_math() {
  "$PYTHON_BIN" scripts/select_p0a15_math.py \
    --base-audit "$AUDIT_ROOT/base.json" \
    --candidate "0.5=$AUDIT_ROOT/scale_0p5.json" \
    --candidate "1.0=$AUDIT_ROOT/scale_1p0.json" \
    --output "$AUDIT_ROOT/math_selection.json"
}

run_gate300_once() {
  local model="$1"
  local trace="data/eval/p0a15_router_hf_gate300.jsonl"
  local eval_audit="reports/audit/gate_p0a15_router_hf_gate300_eval.json"
  local retention="reports/audit/gate_p0a15_router_hf_gate300_retention.json"
  if [[ -e "$trace" || -e "$eval_audit" || -e "$retention" ]]; then
    echo "P0-A15 gate artifacts already exist; repeated run refused." >&2; return 1
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a15_router_gate.py \
    --endpoint "$ENDPOINT" --model-id-math "$model" \
    --model-id-code p0a15-code-250 --model-id-nlp p0a15-nlp-136 \
    --candidate-name p0a15-router-hf --output-trace "$trace" --audit "$eval_audit"
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$trace" --candidate-name p0a15-router-hf --output "$retention"
}

validation() {
  preflight
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a15-base \
    --lora-module "p0a15-math-0p5=$ROOT/$MATH_HALF" \
    --lora-module "p0a15-math-1p0=$ROOT/$MATH_FULL" \
    --lora-module "p0a15-code-250=$ROOT/$CODE_DIR" \
    --lora-module "p0a15-nlp-136=$ROOT/$NLP_DIR" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a15_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a15-base,p0a15-math-0p5,p0a15-math-1p0,p0a15-code-250,p0a15-nlp-136"
  evaluate_once p0a15-base base
  evaluate_once p0a15-math-0p5 scale_0p5
  evaluate_once p0a15-math-1p0 scale_1p0
  if ! select_math; then
    echo "P0-A15 Math selection failed; gate300 remains closed." >&2
    cleanup; trap - EXIT INT TERM; return 1
  fi
  local model
  model="$($PYTHON_BIN -c 'import json; print(json.load(open("reports/audit/p0a15/math_selection.json"))["selected_model_id"])')"
  run_gate300_once "$model"
  local result=$?
  cleanup; trap - EXIT INT TERM
  return "$result"
}

status() {
  for path in reports/audit/gate_p0a15_data.json reports/audit/gate_p0a15_preflight.json \
    reports/audit/p0a15/math_selection.json \
    reports/audit/gate_p0a15_router_hf_gate300_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("selected_scale",d.get("retention_ratios","")))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  validation) validation ;;
  auto) data_build; preflight; validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a15.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a15_data.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a15_math.py \
      scripts/evaluate_p0a15_router_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a15.sh <data-build|preflight|validation|auto|status|structural-check>" ;;
esac
