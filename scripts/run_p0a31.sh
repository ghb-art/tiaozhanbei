#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A31_SERVE_GPU:-0}"
PORT="${P0A31_PORT:-18491}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE="models/checkpoints/p0a4/student-shared-merged"
ADAPTER="models/adapters/p0a30/nlp-step136-scale-0p75"
MODEL_ID="p0a31-nlp-0p75"
AUDIT_ROOT="reports/audit/p0a31"

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

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a31_nlp_gate.py
  require_status reports/audit/gate_p0a31_data.json passed
}

wait_endpoint() {
  local pid="$1" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$MODEL_ID" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as r:
 ids={str(x.get('id')) for x in json.load(r).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

evaluate() {
  require_status reports/audit/gate_p0a31_data.json passed
  require_status reports/audit/p0a30/nlp_scale_selection.json passed
  [[ -d "$BASE" && -d "$ADAPTER" ]] || { echo "Missing P0-A31 model asset" >&2; return 1; }
  if [[ -e "$AUDIT_ROOT/candidate.json" || -e "$AUDIT_ROOT/candidate_trace.jsonl" || \
        -e reports/audit/gate_p0a31_nlp_retention.json ]]; then
    echo "P0-A31 artifacts already exist; repeated run refused." >&2; return 1
  fi
  mkdir -p logs "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a31-base \
    --lora-module "$MODEL_ID=$ROOT/$ADAPTER" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a31_nlp_gate_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a31_nlp_gate --domain nlp --manifest data/p0a31/nlp_gate100.jsonl \
    --endpoint "$ENDPOINT" --model-id "$MODEL_ID" --candidate-name p0a30-nlp-scale-0p75 \
    --expected-rows 100 --workers 8 --max-tokens 256 --thinking off \
    --output-trace "$AUDIT_ROOT/candidate_trace.jsonl" --audit "$AUDIT_ROOT/candidate.json"
  require_status "$AUDIT_ROOT/candidate.json" passed
  set +e
  "$PYTHON_BIN" scripts/p0a31_retention_gate.py
  local result=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$result"
}

case "${1:-help}" in
  data-build) data_build ;;
  evaluate) evaluate ;;
  guarded-evaluate)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh bash scripts/run_p0a31.sh evaluate
    ;;
  structural-check)
    bash -n scripts/run_p0a31.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a31_nlp_gate.py \
      scripts/evaluate_p0a11_domain.py scripts/p0a31_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a31.sh <data-build|evaluate|guarded-evaluate|structural-check>" ;;
esac
