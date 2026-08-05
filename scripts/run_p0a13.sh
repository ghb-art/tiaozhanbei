#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A13_SERVE_GPU:-0}"
PORT="${P0A13_PORT:-18467}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
CODE_DIR="models/checkpoints/p0a11/code-specialist/checkpoint-250"
NLP_DIR="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
AUDIT_ROOT="reports/audit/p0a13"

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

asset_guards() {
  require_status reports/audit/p0a12/math_selection.json failed
  require_status reports/audit/p0a11/code_selection.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  require_dir "$BASE_DIR"; require_dir "$CODE_DIR"; require_dir "$NLP_DIR"
  "$PYTHON_BIN" - <<'PY'
import json
code=json.load(open('reports/audit/p0a11/code_selection.json'))
nlp=json.load(open('reports/audit/p0a10/nlp_selection.json'))
math=json.load(open('reports/audit/p0a12/math_selection.json'))
if code.get('selected_step') != 250: raise SystemExit('Frozen Code is not step 250')
if nlp.get('selected_step') != 136: raise SystemExit('Frozen NLP is not step 136')
if math.get('selected_step') is not None: raise SystemExit('P0-A12 unexpectedly selected Math')
print('P0-A13 assets passed: math=base code=250 nlp=136')
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
needed=set(sys.argv[2].split(','))
raise SystemExit(0 if needed.issubset(actual) else 1)
PY
    then return 0; fi
    sleep 2
  done
  echo "Timed out waiting for $ENDPOINT" >&2
  return 1
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a13_data.py build
  require_status reports/audit/gate_p0a13_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a13_data.json passed
  asset_guards
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
cfg=json.load(open('configs/p0a13_math_runtime.json'))
if cfg['math'] != {
 'adapter': None, 'candidate': 'qwen3_native_thinking',
 'baseline_thinking': False, 'candidate_thinking': True,
 'max_tokens': 768, 'temperature': 0, 'maximum_runtime_candidates': 1,
}: raise SystemExit('P0-A13 runtime profile changed')
rows=sum(1 for line in Path('data/p0a13/math_validation.jsonl').open() if line.strip())
if rows != 2237: raise SystemExit(f'P0-A13 row count changed: {rows}')
report={
 'gate':'P0-A13-PREFLIGHT','status':'passed','runtime_candidates':1,
 'math_model':'base','math_thinking_candidate':True,'math_max_tokens':768,
 'code_adapter_step':250,'nlp_adapter_step':136,'validation_rows':rows,
 'gate300_opened':False,'formal_full_opened':False,
}
Path('reports/audit/gate_p0a13_preflight.json').write_text(
 json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('P0-A13 preflight passed.')
PY
  require_status reports/audit/gate_p0a13_preflight.json passed
}

evaluate_runtime() {
  local thinking="$1" name="$2"
  local audit="$AUDIT_ROOT/${name}.json"
  if [[ -f "$audit" ]]; then
    require_status "$audit" passed
    return 0
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a13 --domain math --manifest data/p0a13/math_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id p0a13-base --candidate-name "$name" \
    --expected-rows 2237 --max-tokens 768 --thinking "$thinking" \
    --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" \
    --audit "$audit" --workers "${P0A13_EVAL_WORKERS:-8}"
}

select_runtime() {
  "$PYTHON_BIN" scripts/select_p0a13_runtime.py \
    --base-audit "$AUDIT_ROOT/base_thinking_off.json" \
    --candidate-audit "$AUDIT_ROOT/base_thinking_on.json" \
    --minimum-gain 0.02 --minimum-canonical-format-rate 0.95 \
    --output "$AUDIT_ROOT/runtime_selection.json"
}

run_gate300_once() {
  local trace="data/eval/p0a13_router_hf_gate300.jsonl"
  local eval_audit="reports/audit/gate_p0a13_router_hf_gate300_eval.json"
  local retention="reports/audit/gate_p0a13_router_hf_gate300_retention.json"
  if [[ -e "$trace" || -e "$eval_audit" || -e "$retention" ]]; then
    echo "P0-A13 gate300 artifacts already exist; repeated run refused." >&2
    return 1
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a13_router_gate.py \
    --endpoint "$ENDPOINT" --model-id-math p0a13-base \
    --model-id-code p0a13-code-250 --model-id-nlp p0a13-nlp-136 \
    --candidate-name p0a13-router-hf --output-trace "$trace" --audit "$eval_audit"
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$trace" --candidate-name p0a13-router-hf --output "$retention"
}

validation() {
  preflight
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a13-base \
    --lora-module "p0a13-code-250=$ROOT/$CODE_DIR" \
    --lora-module "p0a13-nlp-136=$ROOT/$NLP_DIR" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a13_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a13-base,p0a13-code-250,p0a13-nlp-136"
  evaluate_runtime off base_thinking_off
  evaluate_runtime on base_thinking_on
  if ! select_runtime; then
    echo "P0-A13 runtime selection failed; gate300 remains closed." >&2
    cleanup; trap - EXIT INT TERM
    return 1
  fi
  run_gate300_once
  local result=$?
  cleanup; trap - EXIT INT TERM
  return "$result"
}

status() {
  local path
  for path in reports/audit/gate_p0a13_data.json \
    reports/audit/gate_p0a13_preflight.json \
    reports/audit/p0a13/runtime_selection.json \
    reports/audit/gate_p0a13_router_hf_gate300_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("retention_ratios",d.get("gain",""))))' "$path"
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
    bash -n scripts/run_p0a13.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a13_data.py \
      scripts/evaluate_p0a11_domain.py scripts/evaluate_p0a13_router_gate.py \
      scripts/select_p0a13_runtime.py scripts/evaluate_p0a5_gate.py \
      scripts/evaluate_p0a6_internal.py
    ;;
  *) echo "Usage: bash scripts/run_p0a13.sh <data-build|preflight|validation|auto|status|structural-check>" ;;
esac
