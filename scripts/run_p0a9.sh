#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A9_SERVE_GPU:-0}"
PORT="${P0A9_PORT:-18463}"
ENDPOINT="http://127.0.0.1:$PORT"

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
  local pid="$1" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "P0-A9 service exited" >&2; return 1; }
    if "$PYTHON_BIN" - "$ENDPOINT" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as r:
 ids={str(x.get('id')) for x in json.load(r).get('data',[])}
need={'p0a9-base','p0a8-code-128','p0a7-nlp-188'}
raise SystemExit(0 if need.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

gate300() {
  require_status reports/audit/p0a6/p0a8_full_selection.json passed
  require_status reports/audit/gate_p0a5_baseline14b_gate300_eval.json passed
  mkdir -p logs runtime
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a9-base \
    --lora-module "p0a8-code-128=$ROOT/models/checkpoints/p0a8/code-specialist/checkpoint-128" \
    --lora-module "p0a7-nlp-188=$ROOT/models/checkpoints/p0a7/nlp-mmlu-aux-specialist/checkpoint-188" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a9_gate300_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  "$PYTHON_BIN" scripts/evaluate_p0a9_router_gate.py \
    --endpoint "$ENDPOINT" \
    --model-id-math p0a9-base \
    --model-id-code p0a8-code-128 \
    --model-id-nlp p0a7-nlp-188 \
    --candidate-name p0a9-router-hf \
    --output-trace data/eval/p0a9_router_hf_gate300.jsonl \
    --audit reports/audit/gate_p0a9_router_hf_gate300_eval.json
  set +e
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config configs/p0a5_capability.json \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace data/eval/p0a9_router_hf_gate300.jsonl \
    --candidate-name p0a9-router-hf \
    --output reports/audit/gate_p0a9_router_hf_gate300_retention.json
  local gate_rc=$?
  set -e
  cleanup
  trap - EXIT INT TERM
  return "$gate_rc"
}

case "${1:-help}" in
  gate300) gate300 ;;
  structural-check)
    bash -n scripts/run_p0a9.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a9_router_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a9.sh <gate300|structural-check>" ;;
esac
