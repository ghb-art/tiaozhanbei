#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVE_GPU="${P0A17_SERVE_GPU:-0}"
PORT="${P0A17_PORT:-18471}"
ENDPOINT="http://127.0.0.1:$PORT"
BASE_DIR="models/checkpoints/p0a4/student-shared-merged"
CODE_DIR="models/checkpoints/p0a11/code-specialist/checkpoint-250"
NLP_DIR="models/checkpoints/p0a10/nlp-specialist/checkpoint-136"

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

preflight() {
  require_status reports/audit/p0a16/math_selection.json failed
  require_status reports/audit/p0a11/code_selection.json passed
  require_status reports/audit/p0a10/nlp_selection.json passed
  require_status reports/audit/gate_p0a4_official_full_retention.json failed
  [[ -d "$BASE_DIR" && -d "$CODE_DIR" && -d "$NLP_DIR" ]] || { echo "Missing frozen model asset" >&2; return 1; }
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
cfg=json.load(open('configs/p0a17_code_nlp_gate.json'))
math=json.load(open('reports/audit/gate_p0a4_official_full_retention.json'))
if float(math['ratios']['math_ratio']) < 0.80: raise SystemExit('Frozen full Math retention is below 80%')
if cfg['counts'] != {'code':100,'nlp':100} or cfg['maximum_runs'] != 1:
 raise SystemExit('P0-A17 gate protocol changed')
report={'gate':'P0-A17-PREFLIGHT','status':'passed',
 'math_full_retention':math['ratios']['math_ratio'],'code_rows':100,'nlp_rows':100,
 'maximum_runs':1,'formal_full_opened':False}
Path('reports/audit/gate_p0a17_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('P0-A17 preflight passed.')
PY
  require_status reports/audit/gate_p0a17_preflight.json passed
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
raise SystemExit(0 if {'p0a17-base','p0a17-code-250','p0a17-nlp-136'}.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

gate() {
  preflight
  if [[ -e data/eval/p0a17_code_nlp_gate200.jsonl || \
        -e reports/audit/gate_p0a17_code_nlp_gate200_eval.json || \
        -e reports/audit/gate_p0a17_code_nlp_retention.json ]]; then
    echo "P0-A17 gate artifacts already exist; repeated run refused." >&2
    return 1
  fi
  mkdir -p logs runtime
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$SERVE_GPU" --port "$PORT" --model-dir "$BASE_DIR" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a17-base \
    --lora-module "p0a17-code-250=$ROOT/$CODE_DIR" \
    --lora-module "p0a17-nlp-136=$ROOT/$NLP_DIR" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 \
    >logs/p0a17_gate_server.log 2>&1 &
  local server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid"
  "$PYTHON_BIN" scripts/evaluate_p0a17_code_nlp_gate.py \
    --endpoint "$ENDPOINT" --model-id-code p0a17-code-250 --model-id-nlp p0a17-nlp-136
  "$PYTHON_BIN" scripts/p0a17_retention_gate.py
  local result=$?
  cleanup; trap - EXIT INT TERM
  return "$result"
}

status() {
  for path in reports/audit/gate_p0a17_preflight.json \
    reports/audit/gate_p0a17_code_nlp_gate200_eval.json \
    reports/audit/gate_p0a17_code_nlp_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("retention_ratios",d.get("accuracy_by_domain","")))' "$path"
    else echo "$path missing"; fi
  done
}

case "${1:-help}" in
  preflight) preflight ;;
  gate) gate ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a17.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a17_code_nlp_gate.py scripts/p0a17_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a17.sh <preflight|gate|status|structural-check>" ;;
esac
