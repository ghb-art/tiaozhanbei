#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SERVER="$ROOT/external/llama.cpp/build/bin/llama-server"
CONFIG="configs/p0a43_edge_full.json"
BASE_HF="$ROOT/models/checkpoints/p0a4/student-shared-merged"
BASE_GGUF="$ROOT/models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf"
CODE_LORA="$ROOT/models/adapters/p0a25/code-failure-repair-checkpoint-192-f16.gguf"
NLP_HF="$ROOT/models/checkpoints/p0a10/nlp-specialist/checkpoint-136"
NLP_LORA="$ROOT/models/adapters/p0a10/nlp-specialist-checkpoint-136-f16.gguf"
SPLIT_DIR="data/splits/p0a43_edge_best_full"
SHARD_DIR="reports/sealed/p0a43/shards"
AUDIT_DIR="reports/audit/p0a43"

require_status() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit(f"Missing audit: {p}")
d=json.loads(p.read_text(encoding="utf-8"))
if d.get("status") != "passed": raise SystemExit(f"Audit rejected: {p} status={d.get('status')}")
print(f"Audit guard passed: {p} status=passed")
PY
}

adapter_prepare() {
  require_status reports/audit/gate_p0a42_domain_selection.json
  [[ -x "$SERVER" ]] || { echo "Missing llama-server: $SERVER" >&2; return 1; }
  [[ -s "$BASE_GGUF" && -s "$CODE_LORA" ]] || { echo "Missing Q4 base or Code GGUF LoRA" >&2; return 1; }
  if [[ ! -s "$NLP_LORA" ]]; then
    mkdir -p "$(dirname "$NLP_LORA")"
    "$PYTHON_BIN" external/llama.cpp/convert_lora_to_gguf.py \
      "$NLP_HF" --base "$BASE_HF" --outtype f16 --outfile "$NLP_LORA"
  fi
  "$PYTHON_BIN" - <<'PY'
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
root=Path('.')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base=root/'models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf'
code=root/'models/adapters/p0a25/code-failure-repair-checkpoint-192-f16.gguf'
nlp=root/'models/adapters/p0a10/nlp-specialist-checkpoint-136-f16.gguf'
hf=root/'models/checkpoints/p0a10/nlp-specialist/checkpoint-136/adapter_model.safetensors'
q4=json.loads((root/'reports/audit/gate_p0a4_student_q4_prepare.json').read_text())
if sha(base) != q4['q4_gguf_hash']: raise SystemExit('Q4 base hash changed')
p27=json.loads((root/'reports/audit/gate_p0a27_preflight.json').read_text())
if sha(code) != p27['adapter_sha256']: raise SystemExit('Code GGUF LoRA hash changed')
report={'gate':'P0-A43-NLP-GGUF-PREPARE','check_version':'1.0','created_ts':datetime.now(timezone.utc).isoformat(),
 'status':'passed','base_gguf':str(base),'base_gguf_hash':sha(base),'weight_type':'Q4_K_M',
 'nlp_hf_adapter':str(hf.parent),'nlp_hf_adapter_hash':sha(hf),'nlp_gguf_adapter':str(nlp),
 'nlp_gguf_adapter_hash':sha(nlp),'nlp_gguf_adapter_bytes':nlp.stat().st_size,'outtype':'f16',
 'code_gguf_adapter':str(code),'code_gguf_adapter_hash':sha(code),'kv_cache_type':'q8_0'}
report['report_hash']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
out=root/'reports/audit/gate_p0a43_nlp_gguf_prepare.json'; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n')
print(f"Wrote {out} nlp_bytes={nlp.stat().st_size}")
PY
}

preflight() {
  adapter_prepare
  "$PYTHON_BIN" scripts/prepare_p0a43_edge_full.py
  require_status reports/audit/gate_p0a43_preflight.json
  bash -n scripts/run_p0a43.sh
  "$PYTHON_BIN" -m py_compile scripts/prepare_p0a43_edge_full.py scripts/p0a43_retention_gate.py
}

wait_endpoint() {
  local pid="$1" port="$2" model_id="$3" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$port" "$model_id" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models',timeout=2) as r:
 ids={str(x.get('id')) for x in json.load(r).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

start_server() {
  local gpu="$1" port="$2" alias="$3" adapter="$4" log="$5"
  local lora_args=()
  if [[ -n "$adapter" ]]; then lora_args=(--lora "$adapter"); fi
  env CUDA_VISIBLE_DEVICES="$gpu" "$SERVER" \
    --model "$BASE_GGUF" "${lora_args[@]}" --alias "$alias" \
    --host 127.0.0.1 --port "$port" --ctx-size 2048 \
    --threads 8 --parallel 1 --batch-size 32 --ubatch-size 16 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
    --n-gpu-layers all --no-repack --cache-ram 0 --no-cache-idle-slots \
    --reasoning off --reasoning-format none >"$log" 2>&1 &
  LAST_SERVER_PID=$!
}

evaluate_dataset() {
  local dataset="$1" port="$2" model_id="$3" shards="$4" index="$5" name="$6"
  "$PYTHON_BIN" scripts/evaluate_chapter2_capability.py \
    --student-url "http://127.0.0.1:$port" --student-model-id "$model_id" \
    --local-model-dir models/checkpoints/p0a4/student-shared-merged \
    --dataset "$dataset" --use-frozen-final --sample-limit-per-dataset 0 \
    --split-dir "$SPLIT_DIR" --num-shards "$shards" --shard-index "$index" \
    --output-trace "$SHARD_DIR/${name}.jsonl" --audit "$AUDIT_DIR/${name}.json" \
    --disable-thinking --kv-cache-type q8_0 --prompt-style v15 \
    --max-new-tokens-map cmmlu=16,gsm8k=512,humaneval=512 \
    --timeout-sec 180 --humaneval-timeout-sec 10 --min-accuracy 0
}

full() {
  preflight
  local final_trace="reports/sealed/p0a43/edge_best_router_q4_full.jsonl"
  local final_audit="reports/audit/gate_p0a43_edge_best_router_q4_full.json"
  local retention="reports/audit/gate_p0a43_edge_best_router_q4_full_retention.json"
  if [[ -e "$final_trace" || -e "$final_audit" || -e "$retention" ]]; then
    echo "P0-A43 official-full artifacts already exist; repeated run refused." >&2
    return 1
  fi
  mkdir -p logs runtime "$SHARD_DIR" "$AUDIT_DIR"
  local server_pids=() eval_pids=()
  cleanup() {
    local pid
    for pid in "${eval_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    for pid in "${server_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    for pid in "${eval_pids[@]}" "${server_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  }
  trap cleanup EXIT INT TERM

  start_server 0 18520 p0a43-math-base "" logs/p0a43_math_server.log; server_pids+=("$LAST_SERVER_PID")
  start_server 1 18521 p0a43-code-best "$CODE_LORA" logs/p0a43_code_server.log; server_pids+=("$LAST_SERVER_PID")
  start_server 2 18522 p0a43-nlp-best-0 "$NLP_LORA" logs/p0a43_nlp0_server.log; server_pids+=("$LAST_SERVER_PID")
  start_server 3 18523 p0a43-nlp-best-1 "$NLP_LORA" logs/p0a43_nlp1_server.log; server_pids+=("$LAST_SERVER_PID")
  wait_endpoint "${server_pids[0]}" 18520 p0a43-math-base
  wait_endpoint "${server_pids[1]}" 18521 p0a43-code-best
  wait_endpoint "${server_pids[2]}" 18522 p0a43-nlp-best-0
  wait_endpoint "${server_pids[3]}" 18523 p0a43-nlp-best-1
  echo "P0-A43 four edge routes ready; starting official-full evaluation."

  evaluate_dataset gsm8k 18520 p0a43-math-base 1 0 math >logs/p0a43_math_eval.log 2>&1 & eval_pids+=("$!")
  evaluate_dataset humaneval 18521 p0a43-code-best 1 0 code >logs/p0a43_code_eval.log 2>&1 & eval_pids+=("$!")
  evaluate_dataset cmmlu 18522 p0a43-nlp-best-0 2 0 nlp_shard_0 >logs/p0a43_nlp0_eval.log 2>&1 & eval_pids+=("$!")
  evaluate_dataset cmmlu 18523 p0a43-nlp-best-1 2 1 nlp_shard_1 >logs/p0a43_nlp1_eval.log 2>&1 & eval_pids+=("$!")
  echo "P0-A43 eval_pids=${eval_pids[*]}"
  local failures=0 pid rc
  set +e
  for pid in "${eval_pids[@]}"; do
    wait "$pid"; rc=$?
    if (( rc != 0 )); then echo "P0-A43 evaluator pid=$pid failed rc=$rc" >&2; failures=$((failures+1)); fi
  done
  set -e
  if (( failures != 0 )); then return 1; fi
  require_status "$AUDIT_DIR/math.json"
  require_status "$AUDIT_DIR/code.json"
  require_status "$AUDIT_DIR/nlp_shard_0.json"
  require_status "$AUDIT_DIR/nlp_shard_1.json"
  cleanup; trap - EXIT INT TERM

  "$PYTHON_BIN" scripts/merge_capability_shards.py \
    --input "$SHARD_DIR/math.jsonl" --input "$SHARD_DIR/code.jsonl" \
    --input "$SHARD_DIR/nlp_shard_0.jsonl" --input "$SHARD_DIR/nlp_shard_1.jsonl" \
    --split-dir "$SPLIT_DIR" --role student \
    --model-name "Edge-Best-Router-Q4_K_M-Q8KV-CodeP0A25-NLPP0A10" \
    --output "$final_trace" --audit "$final_audit"
  require_status "$final_audit"
  set +e
  "$PYTHON_BIN" scripts/p0a43_retention_gate.py
  local result=$?
  set -e
  return "$result"
}

guarded_full() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 \
    bash scripts/run_with_memory_guard.sh bash scripts/run_p0a43.sh full
}

status() {
  local p
  for p in reports/audit/gate_p0a43_nlp_gguf_prepare.json \
    reports/audit/gate_p0a43_preflight.json \
    "$AUDIT_DIR/math.json" "$AUDIT_DIR/code.json" \
    "$AUDIT_DIR/nlp_shard_0.json" "$AUDIT_DIR/nlp_shard_1.json" \
    reports/audit/gate_p0a43_edge_best_router_q4_full.json \
    reports/audit/gate_p0a43_edge_best_router_q4_full_retention.json; do
    if [[ -f "$p" ]]; then
      "$PYTHON_BIN" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[1],d.get("status"),d.get("accuracy_by_dataset",d.get("retention_ratios","")),d.get("sample_count",""))' "$p"
    else echo "$p missing"; fi
  done
}

case "${1:-help}" in
  adapter-prepare) adapter_prepare ;;
  preflight) preflight ;;
  full) full ;;
  guarded-full) guarded_full ;;
  status) status ;;
  structural-check)
    bash -n scripts/run_p0a43.sh
    "$PYTHON_BIN" -m py_compile scripts/prepare_p0a43_edge_full.py scripts/p0a43_retention_gate.py
    ;;
  *) echo "Usage: bash scripts/run_p0a43.sh <adapter-prepare|preflight|full|guarded-full|status|structural-check>" ;;
esac
