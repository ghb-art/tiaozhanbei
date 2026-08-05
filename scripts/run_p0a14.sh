#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A14_SERVE_GPU:-0}"
PORT="${P0A14_PORT:-18468}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
CODE_DIR="models/checkpoints/p0a11/code-specialist/checkpoint-250"
NLP_DIR="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
AUDIT_ROOT="reports/audit/p0a14"

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
  require_status reports/audit/p0a13/runtime_selection.json failed
  require_status reports/audit/p0a11/code_selection.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  require_dir "$BASE_DIR"; require_dir "$CODE_DIR"; require_dir "$NLP_DIR"
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
  echo "Timed out waiting for $ENDPOINT" >&2
  return 1
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a14_data.py build
  require_status reports/audit/gate_p0a14_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a14_data.json passed
  asset_guards
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
cfg=json.load(open('configs/p0a14_math_self_consistency.json'))
math=cfg['math']
expected={'thinking':True,'baseline_samples':1,'candidate_samples':3,
 'baseline_temperature':0.0,'candidate_temperature':0.6,'candidate_top_p':0.95,
 'max_tokens':768,'vote':'numeric_majority_then_first','maximum_runtime_candidates':1}
if math != expected: raise SystemExit('P0-A14 runtime profile changed')
rows=sum(1 for line in Path('data/p0a14/math_validation.jsonl').open() if line.strip())
if rows != 727: raise SystemExit(f'P0-A14 row count changed: {rows}')
report={'gate':'P0-A14-PREFLIGHT','status':'passed','validation_rows':rows,
 'runtime_candidates':1,'math_runtime':'thinking_vote3','gate300_opened':False,
 'formal_full_opened':False}
Path('reports/audit/gate_p0a14_preflight.json').write_text(
 json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('P0-A14 preflight passed.')
PY
  require_status reports/audit/gate_p0a14_preflight.json passed
}

evaluate_mode() {
  local mode="$1"
  local audit="$AUDIT_ROOT/${mode}.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a14_math.py \
    --endpoint "$ENDPOINT" --model-id p0a14-base --mode "$mode" \
    --output-trace "$AUDIT_ROOT/${mode}_trace.jsonl" --audit "$audit" \
    --workers "${P0A14_EVAL_WORKERS:-8}"
}

select_runtime() {
  "$PYTHON_BIN" scripts/select_p0a14_runtime.py \
    --base-audit "$AUDIT_ROOT/single.json" \
    --candidate-audit "$AUDIT_ROOT/vote3.json" \
    --output "$AUDIT_ROOT/runtime_selection.json"
}

run_gate300_once() {
  local trace="data/eval/p0a14_router_hf_gate300.jsonl"
  local eval_audit="reports/audit/gate_p0a14_router_hf_gate300_eval.json"
  local retention="reports/audit/gate_p0a14_router_hf_gate300_retention.json"
  if [[ -e "$trace" || -e "$eval_audit" || -e "$retention" ]]; then
    echo "P0-A14 gate300 artifacts already exist; repeated run refused." >&2
    return 1
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a14_router_gate.py \
    --endpoint "$ENDPOINT" --model-id-math p0a14-base \
    --model-id-code p0a14-code-250 --model-id-nlp p0a14-nlp-136 \
    --candidate-name p0a14-router-hf --output-trace "$trace" --audit "$eval_audit"
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$trace" --candidate-name p0a14-router-hf --output "$retention"
}

validation() {
  preflight
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a14-base \
    --lora-module "p0a14-code-250=$ROOT/$CODE_DIR" \
    --lora-module "p0a14-nlp-136=$ROOT/$NLP_DIR" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a14_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a14-base,p0a14-code-250,p0a14-nlp-136"
  evaluate_mode single
  evaluate_mode vote3
  if ! select_runtime; then
    echo "P0-A14 self-consistency selection failed; gate300 remains closed." >&2
    cleanup; trap - EXIT INT TERM
    return 1
  fi
  run_gate300_once
  local result=$?
  cleanup; trap - EXIT INT TERM
  return "$result"
}

status() {
  for path in reports/audit/gate_p0a14_data.json reports/audit/gate_p0a14_preflight.json \
    reports/audit/p0a14/runtime_selection.json \
    reports/audit/gate_p0a14_router_hf_gate300_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("gain",d.get("retention_ratios","")))' "$path"
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
    bash -n scripts/run_p0a14.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a14_data.py \
      scripts/evaluate_p0a14_math.py scripts/select_p0a14_runtime.py \
      scripts/evaluate_p0a14_router_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a14.sh <data-build|preflight|validation|auto|status|structural-check>" ;;
esac
