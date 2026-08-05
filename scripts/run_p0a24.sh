#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A24_SERVE_GPU:-0}"
PORT="${P0A24_PORT:-18478}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
SELECTION="reports/audit/p0a23/code_selection.json"

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

selected_step() {
  "$PYTHON_BIN" - "$SELECTION" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
step=d.get('selected_step')
if d.get('status')!='passed' or step not in (96,192):
 raise SystemExit('P0-A23 selection is not eligible')
print(step)
PY
}

preflight() {
  require_status "$SELECTION" passed
  require_status reports/audit/gate_p0a17_code_nlp_retention.json failed
  require_status reports/audit/gate_p0a4_official_full_retention.json failed
  local step adapter
  step="$(selected_step)"
  adapter="models/checkpoints/p0a23/code-continuation/checkpoint-$step"
  [[ -d "$BASE_DIR" && -s "$adapter/adapter_model.safetensors" ]] || {
    echo "Missing selected P0-A23 model asset: $adapter" >&2; return 1;
  }
  "$PYTHON_BIN" - "$step" "$adapter" <<'PY'
import json,sys
from pathlib import Path
step=int(sys.argv[1]); adapter=Path(sys.argv[2])
cfg=json.load(open('configs/p0a24_code_gate.json',encoding='utf-8'))
math=json.load(open('reports/audit/gate_p0a4_official_full_retention.json',encoding='utf-8'))
nlp=json.load(open('reports/audit/gate_p0a17_code_nlp_retention.json',encoding='utf-8'))
if float(math['ratios']['math_ratio']) < 0.80: raise SystemExit('Frozen full Math retention is below 80%')
if float(nlp['retention_ratios']['nlp']) < 0.78: raise SystemExit('Frozen NLP retention is below 78%')
if cfg['counts'] != {'code':100} or cfg['maximum_runs'] != 1: raise SystemExit('P0-A24 protocol changed')
report={
 'gate':'P0-A24-PREFLIGHT','status':'passed','selected_step':step,
 'selected_adapter':adapter.as_posix(),
 'math_full_retention':math['ratios']['math_ratio'],
 'nlp_frozen_retention':nlp['retention_ratios']['nlp'],
 'code_rows':100,'maximum_runs':1,'formal_full_opened':False,
}
p=Path('reports/audit/gate_p0a24_preflight.json'); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('P0-A24 preflight passed.')
PY
  require_status reports/audit/gate_p0a24_preflight.json passed
}

wait_endpoint() {
  local pid="$1" model_id="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$model_id" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

gate() {
  preflight
  if [[ -e data/eval/p0a24_code_gate100.jsonl || \
        -e reports/audit/gate_p0a24_code_gate100_eval.json || \
        -e reports/audit/gate_p0a24_code_retention.json ]]; then
    echo "P0-A24 gate artifacts already exist; repeated run refused." >&2
    return 1
  fi
  local step adapter model_id
  step="$(selected_step)"
  adapter="$ROOT/models/checkpoints/p0a23/code-continuation/checkpoint-$step"
  model_id="p0a24-code-$step"
  mkdir -p logs runtime
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a24-base \
    --lora-module "$model_id=$adapter" \
    --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a24_code_gate_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$model_id"
  "$PYTHON_BIN" scripts/evaluate_p0a24_code_gate.py \
    --endpoint "$ENDPOINT" --model-id "$model_id" --selected-step "$step"
  set +e
  "$PYTHON_BIN" scripts/p0a24_retention_gate.py
  local result=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$result"
}

guarded_gate() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a24.sh gate
}

status() {
  local path
  for path in reports/audit/gate_p0a24_preflight.json \
    reports/audit/gate_p0a24_code_gate100_eval.json \
    reports/audit/gate_p0a24_code_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("code_retention",d.get("accuracy","")))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  preflight) preflight ;;
  gate) gate ;;
  guarded-gate) guarded_gate ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a24.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a24_code_gate.py scripts/p0a24_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a24.sh <preflight|gate|guarded-gate|status|structural-check>" ;;
esac
