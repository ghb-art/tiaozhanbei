#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A37_SERVE_GPU:-0}"
PORT="${P0A37_PORT:-18500}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE="models/checkpoints/p0a4/student-shared-merged"
INITIAL="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
P0A36="models/checkpoints/p0a36/nlp-balanced-mcq"
AUDIT_ROOT="reports/audit/p0a37"

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
  local model_id="$1" label="$2" audit="$AUDIT_ROOT/$2.json"
  if [[ -f "$audit" ]]; then return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a37 --domain nlp --manifest data/p0a34/nlp_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 260 --workers "${P0A37_EVAL_WORKERS:-8}" \
    --thinking off --max-tokens 256 --timeout-sec 120 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$audit"
}

evaluate() {
  for path in "$INITIAL/adapter_model.safetensors" \
    "$P0A36/checkpoint-64/adapter_model.safetensors" \
    "$P0A36/checkpoint-128/adapter_model.safetensors"; do
    [[ -s "$path" ]] || { echo "Missing P0-A37 asset: $path" >&2; return 1; }
  done
  mkdir -p logs "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a37-base \
    --lora-module "p0a37-initial=$ROOT/$INITIAL" \
    --lora-module "p0a37-nlp-64=$ROOT/$P0A36/checkpoint-64" \
    --lora-module "p0a37-nlp-128=$ROOT/$P0A36/checkpoint-128" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a37_transfer_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a37-base
  evaluate_once p0a37-initial initial_nlp
  evaluate_once p0a37-nlp-64 nlp_64
  evaluate_once p0a37-nlp-128 nlp_128
  set +e
  "$PYTHON_BIN" scripts/select_p0a37_transfer.py
  local rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$rc"
}

case "${1:-help}" in
  evaluate) evaluate ;;
  guarded-evaluate)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
      bash scripts/run_p0a37.sh evaluate
    ;;
  structural-check)
    bash -n scripts/run_p0a37.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a11_domain.py \
      scripts/select_p0a37_transfer.py
    ;;
  *) echo "Usage: bash scripts/run_p0a37.sh <guarded-evaluate|structural-check>" ;;
esac
