#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPUS="${P0A26_GPUS:-0,1,2,3}"
STUDENT_GPU="${P0A26_STUDENT_GPU:-0}"
PORT="${P0A26_PORT:-18481}"
ENDPOINT="http://127.0.0.1:$PORT"
BASELINE_DIR="models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ"
STUDENT_BASE="models/checkpoints/p0a4/student-shared-merged"
CANDIDATE_ADAPTER="models/checkpoints/p0a25/code-failure-repair/checkpoint-192"
AUDIT_ROOT="reports/audit/p0a26"

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
 raise SystemExit(f"P0-A26 baseline requires four distinct GPU ids: {values}")
print('P0-A26 baseline GPU group:',','.join(values))
PY
}

preflight() {
  validate_gpu_group
  require_status reports/audit/gate_p0a25_data.json passed
  require_status reports/audit/p0a25/code_selection.json passed
  [[ -d "$BASELINE_DIR" && -d "$STUDENT_BASE" && -s "$CANDIDATE_ADAPTER/adapter_model.safetensors" ]] || {
    echo "Missing P0-A26 model asset" >&2; return 1;
  }
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
cfg=json.load(open('configs/p0a26_code_gate.json',encoding='utf-8'))
selection=json.load(open('reports/audit/p0a25/code_selection.json',encoding='utf-8'))
rows=[json.loads(x) for x in Path('data/p0a25/code_gate100.jsonl').read_text(encoding='utf-8').splitlines() if x]
if cfg.get('protocol')!='P0-A26-FRESH-CODE100-GATE': raise SystemExit('P0-A26 config changed')
if selection.get('selected_step')!=192: raise SystemExit('P0-A25 selected step changed')
if len(rows)!=100 or len({x['sample_id'] for x in rows})!=100: raise SystemExit('P0-A26 gate is not 100 unique rows')
if any(x.get('split_role')!='p0a25_frozen_gate' for x in rows): raise SystemExit('P0-A26 split role changed')
report={'gate':'P0-A26-PREFLIGHT','status':'passed','gate_rows':100,
 'baseline_runs':1,'candidate_runs':1,'selected_step':192,
 'p0a24_gate_trace_loaded':False,'formal_full_opened':False}
p=Path('reports/audit/gate_p0a26_preflight.json'); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('P0-A26 preflight passed.')
PY
  require_status reports/audit/gate_p0a26_preflight.json passed
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
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

baseline() {
  preflight
  if [[ -f "$AUDIT_ROOT/baseline.json" ]]; then
    require_status "$AUDIT_ROOT/baseline.json" passed
    echo "P0-A26 14B baseline already frozen."
    return 0
  fi
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port "$PORT" --model-dir "$BASELINE_DIR" \
    --quantization awq --tensor-parallel-size 4 \
    --served-model-name p0a26-baseline14b --max-model-len 2048 \
    --gpu-memory-utilization 0.80 >logs/p0a26_baseline_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a26-baseline14b
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a26_gate --domain code --manifest data/p0a25/code_gate100.jsonl \
    --endpoint "$ENDPOINT" --model-id p0a26-baseline14b \
    --candidate-name baseline-14b-awq --expected-rows 100 --workers 8 \
    --thinking off --max-tokens 768 --timeout-sec 180 --code-timeout-sec 5 \
    --output-trace "$AUDIT_ROOT/baseline_trace.jsonl" \
    --audit "$AUDIT_ROOT/baseline.json"
  require_status "$AUDIT_ROOT/baseline.json" passed
  cleanup; trap - EXIT INT TERM
}

candidate() {
  require_status "$AUDIT_ROOT/baseline.json" passed
  if [[ -e "$AUDIT_ROOT/candidate.json" || -e "$AUDIT_ROOT/candidate_trace.jsonl" || \
        -e reports/audit/gate_p0a26_code_retention.json ]]; then
    echo "P0-A26 candidate artifacts already exist; repeated run refused." >&2
    return 1
  fi
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$STUDENT_GPU" --port "$PORT" --model-dir "$STUDENT_BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a26-student-base \
    --lora-module "p0a26-candidate=$ROOT/$CANDIDATE_ADAPTER" \
    --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a26_candidate_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a26-candidate
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a26_gate --domain code --manifest data/p0a25/code_gate100.jsonl \
    --endpoint "$ENDPOINT" --model-id p0a26-candidate \
    --candidate-name p0a25-code-192 --expected-rows 100 --workers 8 \
    --thinking off --max-tokens 768 --timeout-sec 120 --code-timeout-sec 5 \
    --output-trace "$AUDIT_ROOT/candidate_trace.jsonl" \
    --audit "$AUDIT_ROOT/candidate.json"
  require_status "$AUDIT_ROOT/candidate.json" passed
  set +e
  "$PYTHON_BIN" scripts/p0a26_retention_gate.py
  local result=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$result"
}

all() {
  baseline
  candidate
}

guarded_all() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a26.sh all
}

status() {
  local path
  for path in reports/audit/gate_p0a26_preflight.json \
    "$AUDIT_ROOT/baseline.json" "$AUDIT_ROOT/candidate.json" \
    reports/audit/gate_p0a26_code_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("code_retention","")))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  preflight) preflight ;;
  baseline) baseline ;;
  candidate) candidate ;;
  all) all ;;
  guarded-all) guarded_all ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a26.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a11_domain.py \
      scripts/p0a26_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a26.sh <preflight|baseline|candidate|guarded-all|status|structural-check>" ;;
esac
