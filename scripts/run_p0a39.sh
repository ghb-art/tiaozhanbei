#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A39_GPU:-0}"
PORT="${P0A39_PORT:-18503}"
ENDPOINT="http://127.0.0.1:$PORT"
MODEL="models/pretrained/Qwen--Qwen3-1.7B"
AUDIT_ROOT="reports/audit/p0a39"

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

evaluate_one() {
  local protocol="$1" manifest="$2" rows="$3" label="$4"
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol "$protocol" --domain nlp --manifest "$manifest" \
    --endpoint "$ENDPOINT" --model-id p0a39-original --candidate-name p0a39-original \
    --expected-rows "$rows" --workers "${P0A39_WORKERS:-8}" \
    --thinking off --max-tokens 256 --timeout-sec 120 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$AUDIT_ROOT/${label}.json"
}

evaluate() {
  [[ -s "$MODEL/model-00001-of-00002.safetensors" && -s "$MODEL/model-00002-of-00002.safetensors" ]] || {
    echo "Missing original Qwen3-1.7B snapshot" >&2; return 1;
  }
  mkdir -p logs "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPU" --port "$PORT" --model-dir "$MODEL" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a39-original \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a39_original_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" p0a39-original
  evaluate_one p0a39_synth data/p0a36/nlp_validation.jsonl 256 original_synth
  evaluate_one p0a39_ceval data/p0a34/nlp_validation.jsonl 260 original_ceval
  set +e
  "$PYTHON_BIN" scripts/select_p0a39_original.py
  local rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$rc"
}

case "${1:-help}" in
  evaluate) evaluate ;;
  guarded-evaluate)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
      bash scripts/run_p0a39.sh evaluate
    ;;
  structural-check)
    bash -n scripts/run_p0a39.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a11_domain.py \
      scripts/select_p0a39_original.py
    ;;
  *) echo "Usage: bash scripts/run_p0a39.sh <guarded-evaluate|structural-check>" ;;
esac
