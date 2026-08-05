#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A33_SERVE_GPU:-0}"
PORT="${P0A33_PORT:-18494}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE="models/checkpoints/p0a4/student-shared-merged"
AUDIT_ROOT="reports/audit/p0a33"

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
  "$PYTHON_BIN" - <<'PY'
import json
d=json.load(open('reports/audit/p0a32/nlp_selection.json'))
if d.get('status')!='passed' or d.get('selected_step') not in (128,256):
 raise SystemExit('P0-A32 has no selected checkpoint')
print(d['selected_step'])
PY
}

wait_endpoint() {
  local pid="$1" model_id="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$model_id" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(row.get('id')) for row in json.load(response).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

evaluate() {
  require_status reports/audit/p0a32/nlp_selection.json passed
  local step adapter model_id
  step="$(selected_step)"
  adapter="models/checkpoints/p0a32/nlp-continuation/checkpoint-$step"
  model_id="p0a33-nlp-$step"
  [[ -d "$BASE" && -s "$adapter/adapter_model.safetensors" ]] || {
    echo "Missing P0-A33 model asset" >&2; return 1;
  }
  if [[ -e "$AUDIT_ROOT/candidate.json" || -e "$AUDIT_ROOT/candidate_trace.jsonl" || \
        -e reports/audit/gate_p0a33_nlp_retention.json ]]; then
    echo "P0-A33 artifacts already exist; repeated frozen gate run refused." >&2
    return 1
  fi
  mkdir -p logs "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a33-base \
    --lora-module "$model_id=$ROOT/$adapter" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a33_nlp_gate_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$model_id"
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a33_nlp_gate --domain nlp \
    --manifest data/p0a31/nlp_gate100.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 100 --workers 8 --max-tokens 256 --thinking off \
    --output-trace "$AUDIT_ROOT/candidate_trace.jsonl" \
    --audit "$AUDIT_ROOT/candidate.json"
  require_status "$AUDIT_ROOT/candidate.json" passed
  set +e
  "$PYTHON_BIN" scripts/p0a33_retention_gate.py
  local result=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$result"
}

case "${1:-help}" in
  evaluate) evaluate ;;
  guarded-evaluate)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
      bash scripts/run_p0a33.sh evaluate
    ;;
  status)
    for path in reports/audit/p0a32/nlp_selection.json \
      "$AUDIT_ROOT/candidate.json" reports/audit/gate_p0a33_nlp_retention.json; do
      if [[ -f "$path" ]]; then
        "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("nlp_retention","")))' "$path"
      else echo "$path missing"; fi
    done
    ;;
  structural-check)
    bash -n scripts/run_p0a33.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a11_domain.py \
      scripts/p0a33_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a33.sh <guarded-evaluate|status|structural-check>" ;;
esac
