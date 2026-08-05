#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPUS="${P0A38_GPUS:-0,1,2,3}"
STUDENT_GPU="${P0A38_STUDENT_GPU:-0}"
BASELINE_PORT="${P0A38_BASELINE_PORT:-18501}"
STUDENT_PORT="${P0A38_STUDENT_PORT:-18502}"
AUDIT_ROOT="reports/audit/p0a38"

wait_endpoint() {
  local pid="$1" endpoint="$2" model_id="$3" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$endpoint" "$model_id" >/dev/null 2>&1 <<'PY'
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
  local endpoint="$1" model_id="$2" label="$3"
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a38_nlp_long_gate --domain nlp \
    --manifest data/p0a31/nlp_gate100.jsonl \
    --endpoint "$endpoint" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 100 --workers "${P0A38_EVAL_WORKERS:-8}" \
    --thinking off --max-tokens 768 --timeout-sec 180 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" \
    --audit "$AUDIT_ROOT/${label}.json"
}

run_baseline() {
  [[ -f "$AUDIT_ROOT/baseline14b.json" ]] && return 0
  local endpoint="http://127.0.0.1:$BASELINE_PORT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port "$BASELINE_PORT" \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ \
    --quantization awq --tensor-parallel-size 4 \
    --served-model-name p0a38-baseline14b --max-model-len 1536 \
    --gpu-memory-utilization 0.80 >logs/p0a38_baseline_server.log 2>&1 &
  local server_pid=$!
  trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' RETURN
  wait_endpoint "$server_pid" "$endpoint" p0a38-baseline14b
  evaluate_one "$endpoint" p0a38-baseline14b baseline14b
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
}

run_student() {
  [[ -f "$AUDIT_ROOT/student17b.json" ]] && return 0
  local endpoint="http://127.0.0.1:$STUDENT_PORT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$STUDENT_GPU" --port "$STUDENT_PORT" \
    --model-dir models/checkpoints/p0a4/student-shared-merged \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a38-student-base \
    --lora-module "p0a38-student=$ROOT/models/checkpoints/p0a10/nlp-specialist/checkpoint-136" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a38_student_server.log 2>&1 &
  local server_pid=$!
  trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' RETURN
  wait_endpoint "$server_pid" "$endpoint" p0a38-student
  evaluate_one "$endpoint" p0a38-student student17b
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
}

evaluate() {
  mkdir -p logs "$AUDIT_ROOT"
  if [[ -e reports/audit/gate_p0a38_nlp_retention.json ]]; then
    echo "P0-A38 retention artifact already exists; repeated frozen run refused." >&2
    return 1
  fi
  run_baseline
  run_student
  "$PYTHON_BIN" scripts/p0a38_retention_gate.py
}

case "${1:-help}" in
  evaluate) evaluate ;;
  guarded-evaluate)
    MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
      bash scripts/run_p0a38.sh evaluate
    ;;
  status)
    for path in "$AUDIT_ROOT/baseline14b.json" "$AUDIT_ROOT/student17b.json" \
      reports/audit/gate_p0a38_nlp_retention.json; do
      if [[ -f "$path" ]]; then
        "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("correct_count",d.get("nlp_retention","")))' "$path"
      else echo "$path missing"; fi
    done
    ;;
  structural-check)
    bash -n scripts/run_p0a38.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a11_domain.py \
      scripts/p0a38_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a38.sh <guarded-evaluate|status|structural-check>" ;;
esac
