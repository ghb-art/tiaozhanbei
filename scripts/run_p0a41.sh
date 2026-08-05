#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A41_GPU:-0}"
PORT="${P0A41_PORT:-18505}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
CODE_DIR="models/checkpoints/p0a25/code-failure-repair/checkpoint-192"
NLP_DIR="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
TRACE="data/eval/p0a41_best_router_hf_gate300.jsonl"
EVAL_AUDIT="reports/audit/gate_p0a41_best_router_hf_gate300_eval.json"
RETENTION="reports/audit/gate_p0a41_best_router_hf_gate300_retention.json"

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

preflight() {
  require_status reports/audit/gate_p0a5_baseline14b_gate300_eval.json passed
  require_status reports/audit/p0a25/code_selection.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  [[ -d "$BASE_DIR" && -s "$CODE_DIR/adapter_model.safetensors" && \
     -s "$NLP_DIR/adapter_model.safetensors" ]] || {
    echo "Missing P0-A41 model asset" >&2; return 1;
  }
  "$PYTHON_BIN" - <<'PY'
import hashlib,json
from collections import Counter
from pathlib import Path
cfg=json.loads(Path('configs/p0a41_best_router_gate300.json').read_text(encoding='utf-8'))
code=json.loads(Path('reports/audit/p0a25/code_selection.json').read_text(encoding='utf-8'))
nlp=json.loads(Path('reports/audit/p0a10/nlp_selection.json').read_text(encoding='utf-8'))
base=json.loads(Path('reports/audit/gate_p0a5_baseline14b_gate300_eval.json').read_text(encoding='utf-8'))
rows=[json.loads(x) for x in Path(cfg['gate_manifest']).read_text(encoding='utf-8').splitlines() if x]
counts=Counter(str(x.get('domain','')) for x in rows)
if cfg.get('maximum_runs') != 1: raise SystemExit('P0-A41 maximum_runs changed')
if counts != Counter(cfg['expected_counts']): raise SystemExit(f'P0-A41 counts changed: {counts}')
if code.get('selected_step') != 192: raise SystemExit('Best Code checkpoint changed')
if nlp.get('selected_step') != 136: raise SystemExit('P0-A10 NLP checkpoint changed')
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(cfg['gate_manifest']) != base.get('manifest_hash'):
 raise SystemExit('P0-A41 manifest differs from the frozen 14B baseline')
report={
 'gate':'P0-A41-PREFLIGHT','status':'passed','counts':dict(counts),
 'math_route':cfg['math_route'],'code_step':192,'nlp_step':136,
 'manifest_hash':sha(cfg['gate_manifest']),'maximum_runs':1,
 'formal_full_opened':False,
}
p=Path('reports/audit/gate_p0a41_preflight.json'); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('P0-A41 preflight passed.')
PY
  require_status reports/audit/gate_p0a41_preflight.json passed
}

wait_endpoint() {
  local pid="$1" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
need={'p0a41-math-base','p0a41-code-192','p0a41-nlp-136'}
raise SystemExit(0 if need.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

gate300() {
  preflight
  if [[ -e "$TRACE" || -e "$EVAL_AUDIT" || -e "$RETENTION" ]]; then
    echo "P0-A41 gate artifacts already exist; repeated run refused." >&2
    return 1
  fi
  mkdir -p logs runtime
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a41-math-base \
    --lora-module "p0a41-code-192=$ROOT/$CODE_DIR" \
    --lora-module "p0a41-nlp-136=$ROOT/$NLP_DIR" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a41_gate300_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  "$PYTHON_BIN" scripts/evaluate_p0a41_best_router_gate.py \
    --endpoint "$ENDPOINT" \
    --model-id-math p0a41-math-base \
    --model-id-code p0a41-code-192 \
    --model-id-nlp p0a41-nlp-136
  set +e
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$TRACE" \
    --candidate-name p0a41-best-router-hf \
    --output "$RETENTION"
  local result=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$result"
}

guarded_gate300() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a41.sh gate300
}

status() {
  for p in reports/audit/gate_p0a41_preflight.json "$EVAL_AUDIT" "$RETENTION"; do
    if [[ -f "$p" ]]; then
      "$PYTHON_BIN" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[1],d.get("status"),d.get("accuracy_by_domain",d.get("retention_ratios","")))' "$p"
    else echo "$p missing"; fi
  done
}

case "${1:-help}" in
  preflight) preflight ;;
  gate300) gate300 ;;
  guarded-gate300) guarded_gate300 ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a41.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a41_best_router_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a41.sh <preflight|gate300|guarded-gate300|status|structural-check>" ;;
esac
