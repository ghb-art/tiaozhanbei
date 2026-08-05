#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A20_SERVE_GPU:-0}"
PORT="${P0A20_PORT:-18474}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
ADAPTER="models/checkpoints/p0a19/code-mixed-specialist/checkpoint-256"
AUDIT_ROOT="reports/audit/p0a20"

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
  "$PYTHON_BIN" model_compression/build_p0a20_data.py build
  require_status reports/audit/gate_p0a20_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a20_data.json passed
  require_status reports/audit/p0a19/code_selection.json failed
  [[ -d "$BASE_DIR" && -f "$ADAPTER/adapter_config.json" ]] || {
    echo "Missing P0-A20 model asset" >&2; return 1;
  }
  "$PYTHON_BIN" - <<'PY'
import json,hashlib
from pathlib import Path
cfg=json.load(open('configs/p0a20_code_runtime.json'))
data=json.load(open('reports/audit/gate_p0a20_data.json'))
if cfg.get('protocol')!='P0-A20-CODE-THINKING-RUNTIME': raise SystemExit('protocol mismatch')
if data['output']['rows']!=239 or data['context']['status']!='passed': raise SystemExit('data/context mismatch')
report={'gate':'P0-A20-PREFLIGHT','status':'passed','candidate_count':2,
 'baseline_thinking':'off','candidate_thinking':'on','max_tokens':768,
 'server_context':2048,'formal_full_opened':False,
 'data_audit_hash':hashlib.sha256(Path('reports/audit/gate_p0a20_data.json').read_bytes()).hexdigest()}
Path('reports/audit/gate_p0a20_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('P0-A20 preflight passed.')
PY
  require_status reports/audit/gate_p0a20_preflight.json passed
}

wait_endpoint() {
  local pid="$1" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
raise SystemExit(0 if {'p0a20-base','p0a20-step256'}.issubset(ids) else 1)
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
    --protocol p0a20 --domain code --manifest data/p0a20/code_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$label" \
    --expected-rows 239 --workers "${P0A20_WORKERS:-8}" \
    --thinking "$thinking" --max-tokens 768 --timeout-sec 120 --code-timeout-sec 5 \
    --output-trace "$AUDIT_ROOT/${label}_trace.jsonl" --audit "$audit"
  require_status "$audit" passed
}

validation() {
  preflight
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a20-base \
    --lora-module "p0a20-step256=$ROOT/$ADAPTER" \
    --max-model-len 2048 --gpu-memory-utilization 0.80 \
    >logs/p0a20_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  evaluate_once p0a20-base base_off off
  evaluate_once p0a20-base base_thinking on
  evaluate_once p0a20-step256 step256_thinking on
  set +e
  "$PYTHON_BIN" scripts/select_p0a20_code.py \
    --base-audit "$AUDIT_ROOT/base_off.json" \
    --candidate "base-thinking=$AUDIT_ROOT/base_thinking.json" \
    --candidate "step256-thinking=$AUDIT_ROOT/step256_thinking.json" \
    --output "$AUDIT_ROOT/code_runtime_selection.json"
  local selection_rc=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$selection_rc"
}

guarded_validation() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a20.sh validation
}

status() {
  local path
  for path in reports/audit/gate_p0a20_data.json reports/audit/gate_p0a20_preflight.json \
    "$AUDIT_ROOT/base_off.json" "$AUDIT_ROOT/base_thinking.json" \
    "$AUDIT_ROOT/step256_thinking.json" "$AUDIT_ROOT/code_runtime_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("selected_runtime","")))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  validation) validation ;;
  guarded-validation) guarded_validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a20.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a20_data.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a20_code.py
    ;;
  *) echo "Usage: bash scripts/run_p0a20.sh <data-build|preflight|validation|guarded-validation|status|structural-check>" ;;
esac
