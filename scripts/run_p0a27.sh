#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${P0A27_GPU:-0}"
PORT="${P0A27_PORT:-18482}"
ENDPOINT="http://127.0.0.1:$PORT"
MODEL="models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf"
ADAPTER="models/adapters/p0a25/code-failure-repair-checkpoint-192-f16.gguf"
MODEL_ID="p0a27-code-q4-lora"
AUDIT_ROOT="reports/audit/p0a27"

require_status() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit(f"Missing audit: {p}")
s=json.loads(p.read_text(encoding="utf-8")).get("status")
if s != "passed": raise SystemExit(f"Audit rejected: {p} status={s}")
print(f"Audit guard passed: {p} status={s}")
PY
}

preflight() {
  require_status reports/audit/gate_p0a4_student_q4_prepare.json
  require_status reports/audit/gate_p0a26_code_retention.json
  [[ -s "$MODEL" && -s "$ADAPTER" ]] || {
    echo "Missing P0-A27 model or adapter" >&2; return 1;
  }
  "$PYTHON_BIN" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('.')
model=root/'models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf'
adapter=root/'models/adapters/p0a25/code-failure-repair-checkpoint-192-f16.gguf'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
q4=json.loads(Path('reports/audit/gate_p0a4_student_q4_prepare.json').read_text())
if sha(model) != q4['q4_gguf_hash']: raise SystemExit('Q4 base hash changed')
if sha(adapter) != 'aec4c537a99a8e132b4c3cfb5763c64e3915cb61c25217024e9a5a45f2d14d60':
 raise SystemExit('Code LoRA GGUF hash changed')
report={
 'gate':'P0-A27-PREFLIGHT','status':'passed','weight_type':'Q4_K_M',
 'kv_cache_type':'q8_0','thinking':'off','gpu_layers':'all',
 'base_model':str(model),'base_sha256':sha(model),
 'code_adapter':str(adapter),'adapter_sha256':sha(adapter),
 'adapter_bytes':adapter.stat().st_size,'formal_full_opened':False,
}
p=Path('reports/audit/gate_p0a27_preflight.json'); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('P0-A27 preflight passed.')
PY
}

wait_endpoint() {
  local pid="$1" attempt
  for attempt in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$MODEL_ID" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

evaluate() {
  preflight
  if [[ -e "$AUDIT_ROOT/candidate.json" || -e "$AUDIT_ROOT/candidate_trace.jsonl" || \
        -e reports/audit/gate_p0a27_quantized_code_retention.json ]]; then
    echo "P0-A27 final artifacts already exist; repeated run refused." >&2
    return 1
  fi
  mkdir -p logs runtime "$AUDIT_ROOT"
  env CUDA_VISIBLE_DEVICES="$GPU" \
    external/llama.cpp/build/bin/llama-server \
      --model "$ROOT/$MODEL" --lora "$ROOT/$ADAPTER" --alias "$MODEL_ID" \
      --host 127.0.0.1 --port "$PORT" \
      --ctx-size 2048 --threads 8 --parallel 1 \
      --batch-size 32 --ubatch-size 16 \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --flash-attn on --n-gpu-layers all --no-repack \
      --cache-ram 0 --no-cache-idle-slots \
      --reasoning off --reasoning-format none \
      >logs/p0a27_q4_code_server.log 2>&1 &
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
    --protocol p0a27_quantized_gate --domain code \
    --manifest data/p0a25/code_gate100.jsonl \
    --endpoint "$ENDPOINT" --model-id "$MODEL_ID" \
    --candidate-name p0a25-code-192-q4-k-m-lora \
    --expected-rows 100 --workers 1 --thinking off \
    --max-tokens 768 --timeout-sec 180 --code-timeout-sec 5 \
    --output-trace "$AUDIT_ROOT/candidate_trace.jsonl" \
    --audit "$AUDIT_ROOT/candidate.json"
  require_status "$AUDIT_ROOT/candidate.json"
  set +e
  "$PYTHON_BIN" scripts/p0a27_retention_gate.py
  local result=$?
  set -e
  cleanup
  trap - EXIT INT TERM
  return "$result"
}

guarded_evaluate() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a27.sh evaluate
}

status() {
  local path
  for path in reports/audit/gate_p0a27_preflight.json \
    "$AUDIT_ROOT/candidate.json" \
    reports/audit/gate_p0a27_quantized_code_retention.json; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1],d.get("status"),d.get("accuracy",d.get("code_retention","")))' "$path"
    else
      echo "$path missing"
    fi
  done
}

case "${1:-help}" in
  preflight) preflight ;;
  evaluate) evaluate ;;
  guarded-evaluate) guarded_evaluate ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a27.sh
    "$PYTHON_BIN" -m py_compile scripts/evaluate_p0a11_domain.py scripts/p0a27_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a27.sh <preflight|evaluate|guarded-evaluate|status|structural-check>" ;;
esac
