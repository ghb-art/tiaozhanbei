#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A18_SERVE_GPU:-0}"
PORT="${P0A18_PORT:-18472}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
CODE_250="models/checkpoints/p0a11/code-specialist/checkpoint-250"
CODE_500="models/checkpoints/p0a11/code-specialist/checkpoint-500"
AUDIT_ROOT="reports/audit/p0a18"

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

require_checkpoint() {
  local directory="$1"
  [[ -f "$directory/adapter_config.json" ]] || { echo "Missing adapter config: $directory" >&2; return 1; }
  [[ -s "$directory/adapter_model.safetensors" || -s "$directory/adapter_model.bin" ]] || {
    echo "Missing adapter weights: $directory" >&2
    return 1
  }
}

build_data() {
  "$PYTHON_BIN" model_compression/build_p0a18_data.py build
  require_status reports/audit/gate_p0a18_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a18_data.json passed
  require_status reports/audit/gate_p0a17_code_nlp_retention.json failed
  [[ -d "$BASE_DIR" ]] || { echo "Missing shared base: $BASE_DIR" >&2; return 1; }
  require_checkpoint "$CODE_250"
  require_checkpoint "$CODE_500"
  "$PYTHON_BIN" - <<'PY'
import hashlib,json
from pathlib import Path
cfg=json.loads(Path('configs/p0a18_code_transfer.json').read_text(encoding='utf-8'))
p17=json.loads(Path('reports/audit/gate_p0a17_code_nlp_retention.json').read_text(encoding='utf-8'))
if cfg.get('protocol') != 'P0-A18-CODE-TRANSFER-AUDIT':
 raise SystemExit('P0-A18 protocol identity mismatch')
if p17.get('retention_ratios', {}).get('nlp', 0) < 0.78:
 raise SystemExit('Frozen P0-A17 NLP route is not passing')
if p17.get('retention_ratios', {}).get('code', 1) >= 0.78:
 raise SystemExit('P0-A17 Code is already passing; P0-A18 should not run')
if cfg['validation'] != {
 'repo':'google-research-datasets/mbpp',
 'revision':'4bb6404fdc6cacfda99d4ac4205087b89d32030c',
 'config':'full','split':'validation','rows':90,'public_tests_per_problem':3,
 'minimum_absolute_gain':0.03,'candidate_canonical_not_below_base':True,
 'generation_errors':0,'maximum_candidate_evaluations':2,
}:
 raise SystemExit('P0-A18 validation registration changed')
report={
 'gate':'P0-A18-PREFLIGHT','check_version':'1.0','status':'passed',
 'data_audit':'reports/audit/gate_p0a18_data.json',
 'data_audit_sha256':hashlib.sha256(Path('reports/audit/gate_p0a18_data.json').read_bytes()).hexdigest(),
 'base':cfg['shared_base'],'candidate_steps':[250,500],
 'math_route_frozen':True,'nlp_route_frozen':True,
 'mbpp_validation_used_for_training':False,'formal_full_opened':False,
}
Path('reports/audit/gate_p0a18_preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('P0-A18 preflight passed: base + Code steps 250/500, MBPP validation 90.')
PY
  require_status reports/audit/gate_p0a18_preflight.json passed
}

wait_endpoint() {
  local pid="$1" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "P0-A18 vLLM exited before readiness" >&2; return 1; }
    if "$PYTHON_BIN" - "$ENDPOINT" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
needed={'p0a18-base','p0a18-code-250','p0a18-code-500'}
raise SystemExit(0 if needed.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  echo "Timed out waiting for $ENDPOINT" >&2
  return 1
}

evaluate_once() {
  local model_id="$1" label="$2"
  local trace="$AUDIT_ROOT/${label}_trace.jsonl"
  local audit="$AUDIT_ROOT/${label}.json"
  if [[ -f "$audit" ]]; then
    require_status "$audit" passed
    return 0
  fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a18 --domain code \
    --manifest data/p0a18/code_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model_id" --candidate-name "$model_id" \
    --expected-rows 90 --workers "${P0A18_WORKERS:-8}" \
    --thinking off --max-tokens 768 --timeout-sec 120 --code-timeout-sec 5 \
    --output-trace "$trace" --audit "$audit"
  require_status "$audit" passed
}

validation() {
  preflight
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a18-base \
    --lora-module "p0a18-code-250=$ROOT/$CODE_250" \
    --lora-module "p0a18-code-500=$ROOT/$CODE_500" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a18_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  evaluate_once p0a18-base base_code
  evaluate_once p0a18-code-250 code_250
  evaluate_once p0a18-code-500 code_500
  set +e
  "$PYTHON_BIN" scripts/select_p0a18_code.py \
    --base-audit "$AUDIT_ROOT/base_code.json" \
    --candidate "250=$AUDIT_ROOT/code_250.json" \
    --candidate "500=$AUDIT_ROOT/code_500.json" \
    --output "$AUDIT_ROOT/code_selection.json"
  local selection_rc=$?
  set -e
  cleanup
  trap - EXIT INT TERM
  return "$selection_rc"
}

guarded_validation() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a18.sh validation
}

status() {
  local path
  for path in reports/audit/gate_p0a18_data.json \
    reports/audit/gate_p0a18_preflight.json \
    "$AUDIT_ROOT/base_code.json" "$AUDIT_ROOT/code_250.json" \
    "$AUDIT_ROOT/code_500.json" "$AUDIT_ROOT/code_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("selected_step","")))' "$path"
    else
      echo "$path missing"
    fi
  done
}

case "${1:-help}" in
  data-build) build_data ;;
  preflight) preflight ;;
  validation) validation ;;
  guarded-validation) guarded_validation ;;
  auto) build_data; preflight; guarded_validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a18.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a18_data.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a18_code.py
    ;;
  *)
    echo "Usage: bash scripts/run_p0a18.sh <data-build|preflight|validation|guarded-validation|auto|status|structural-check>"
    ;;
esac
