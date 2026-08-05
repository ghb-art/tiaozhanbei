#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A30_SERVE_GPU:-0}"
PORT="${P0A30_PORT:-18490}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE="models/checkpoints/p0a4/student-shared-merged"
FULL="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
LOW="models/adapters/p0a30/nlp-step136-scale-0p75"
HIGH="models/adapters/p0a30/nlp-step136-scale-1p25"
AUDIT_ROOT="reports/audit/p0a30"

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
  "$PYTHON_BIN" model_compression/build_p0a30_nlp_scale_data.py
  require_status reports/audit/gate_p0a30_data.json passed
}

preflight() {
  require_status reports/audit/gate_p0a30_data.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  require_status reports/audit/gate_p0a17_code_nlp_retention.json failed
  [[ -d "$BASE" && -d "$FULL" && -d "$LOW" && -d "$HIGH" ]] || {
    echo "Missing P0-A30 model asset" >&2; return 1;
  }
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
ret=json.load(open('reports/audit/gate_p0a17_code_nlp_retention.json'))
if ret['retention_ratios']['nlp'] < 0.78: raise SystemExit('Frozen NLP evidence is below 78%')
for path,alpha in (
 ('models/adapters/p0a30/nlp-step136-scale-0p75/adapter_config.json',24),
 ('models/checkpoints/p0a10/nlp-specialist/checkpoint-136/adapter_config.json',32),
 ('models/adapters/p0a30/nlp-step136-scale-1p25/adapter_config.json',40),
):
 d=json.load(open(path));
 if (d.get('r'),d.get('lora_alpha')) != (16,alpha): raise SystemExit(f'Adapter mismatch: {path}')
report={'gate':'P0-A30-PREFLIGHT','status':'passed','validation_rows':235,
 'candidate_scales':[0.75,1.25],'reference_scale':1.0,
 'minimum_gain_questions':3,'formal_test_opened':False}
p=Path('reports/audit/gate_p0a30_preflight.json');p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('P0-A30 preflight passed.')
PY
}

wait_endpoint() {
  local pid="$1" ids="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$ids" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as r:
 actual={str(x.get('id')) for x in json.load(r).get('data',[])}
raise SystemExit(0 if set(sys.argv[2].split(',')).issubset(actual) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

evaluate_once() {
  local model="$1" name="$2" audit="$AUDIT_ROOT/$2.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a11_domain.py \
    --protocol p0a30 --domain nlp --manifest data/p0a30/nlp_validation.jsonl \
    --endpoint "$ENDPOINT" --model-id "$model" --candidate-name "$name" \
    --expected-rows 235 --workers 8 --max-tokens 256 --thinking off \
    --output-trace "$AUDIT_ROOT/${name}_trace.jsonl" --audit "$audit"
}

validation() {
  preflight
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPU" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a30-base \
    --lora-module "p0a30-nlp-0p75=$ROOT/$LOW" \
    --lora-module "p0a30-nlp-1p0=$ROOT/$FULL" \
    --lora-module "p0a30-nlp-1p25=$ROOT/$HIGH" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a30_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a30-nlp-0p75,p0a30-nlp-1p0,p0a30-nlp-1p25"
  evaluate_once p0a30-nlp-1p0 scale_1p0
  evaluate_once p0a30-nlp-0p75 scale_0p75
  evaluate_once p0a30-nlp-1p25 scale_1p25
  set +e
  "$PYTHON_BIN" scripts/select_p0a30_nlp_scale.py
  local result=$?
  set -e
  cleanup; trap - EXIT INT TERM
  return "$result"
}

status() {
  local path
  for path in reports/audit/gate_p0a30_data.json reports/audit/gate_p0a30_preflight.json \
    "$AUDIT_ROOT/scale_1p0.json" "$AUDIT_ROOT/scale_0p75.json" \
    "$AUDIT_ROOT/scale_1p25.json" "$AUDIT_ROOT/nlp_scale_selection.json"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("selected_scale","")))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  validation) validation ;;
  auto) data_build; preflight; validation ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a30.sh
    "$PYTHON_BIN" -m py_compile model_compression/build_p0a30_nlp_scale_data.py \
      scripts/evaluate_p0a11_domain.py scripts/select_p0a30_nlp_scale.py
    ;;
  *) echo "Usage: bash scripts/run_p0a30.sh <data-build|preflight|validation|auto|status|structural-check>" ;;
esac
