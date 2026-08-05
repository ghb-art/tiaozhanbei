#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A35_SERVE_GPU:-0}"
PORT="${P0A35_PORT:-18497}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE="models/checkpoints/p0a4/student-shared-merged"
ADAPTER="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
AUDIT_ROOT="reports/audit/p0a35"

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
  local pid="$1" model_id="$2" attempt
  for attempt in $(seq 1 300); do
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

evaluate_once() {
  local model_id="$1" label="$2" thinking="$3" audit="$AUDIT_ROOT/$2.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a35 --domain nlp --manifest data/p0a34/nlp_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$model_id-$thinking" \
    --expected-rows 260 --workers "${P0A35_EVAL_WORKERS:-8}" \
    --thinking "$thinking" --max-tokens 768 --timeout-sec 180 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$audit"
  require_status "$audit" passed
}

evaluate() {
  [[ -d "$BASE" && -s "$ADAPTER/adapter_model.safetensors" ]] || {
    echo "Missing P0-A35 base or NLP adapter" >&2; return 1;
  }
  mkdir -p logs "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a35-base \
    --lora-module "p0a35-nlp=$ROOT/$ADAPTER" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a35_runtime_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a35-nlp
  evaluate_once p0a35-nlp thinking_off off
  evaluate_once p0a35-nlp thinking_on on
  set +e
  "$PYTHON_BIN" scripts/select_p0a35_runtime.py \
    --off-audit "$AUDIT_ROOT/thinking_off.json" \
    --on-audit "$AUDIT_ROOT/thinking_on.json" \
    --output "$AUDIT_ROOT/runtime_selection.json"
  local rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$rc"
}

case "${1:-help}" in
  evaluate) evaluate ;;
  guarded-evaluate)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
      bash scripts/run_p0a35.sh evaluate
    ;;
  status)
    for path in "$AUDIT_ROOT/thinking_off.json" "$AUDIT_ROOT/thinking_on.json" \
      "$AUDIT_ROOT/runtime_selection.json"; do
      if [[ -f "$path" ]]; then
        "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("gain_questions","")))' "$path"
      else echo "$path missing"; fi
    done
    ;;
  structural-check)
    bash -n scripts/run_p0a35.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a11_domain.py \
      scripts/select_p0a35_runtime.py
    ;;
  *) echo "Usage: bash scripts/run_p0a35.sh <guarded-evaluate|status|structural-check>" ;;
esac
