#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A44_GPUS:-0,1,2,3}"
BASE="models/checkpoints/p0a4/student-shared-merged"
PORT="${P0A44_PORT:-18540}"
ENDPOINT="http://127.0.0.1:$PORT"
AUDIT_ROOT="reports/audit/p0a44"
LLAMA_SERVER="$ROOT/external/llama.cpp/build/bin/llama-server"
BASE_GGUF="$ROOT/models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf"

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not p.is_file(): raise SystemExit(f'Missing audit: {p}')
s=json.loads(p.read_text(encoding='utf-8')).get('status')
if s not in allowed: raise SystemExit(f'Audit rejected: {p} status={s}')
print(f'Audit guard passed: {p} status={s}')
PY
}

cfg() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
d=json.load(open('configs/p0a44_aligned_retrain.json'))[sys.argv[1]]
print(d[sys.argv[2]])
PY
}

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
x=[v.strip() for v in sys.argv[1].split(',') if v.strip()]
if len(x)!=4 or len(set(x))!=4 or not all(v.isdigit() for v in x):
 raise SystemExit(f'P0-A44 requires four distinct GPU ids: {x}')
print('P0-A44 GPU group:',','.join(x))
PY
}

data_build() {
  if [[ ! -f reports/audit/gate_p0a44_data.json ]]; then
    "$PYTHON_BIN" model_compression/build_p0a44_aligned_data.py
  fi
  require_status reports/audit/gate_p0a44_data.json passed
}

preflight_domain() {
  local domain="$1" max checkpoint rank alpha lr
  max="$(cfg "$domain" max_steps)"; checkpoint="$(cfg "$domain" checkpoint_steps)"
  rank="$(cfg "$domain" rank)"; alpha="$(cfg "$domain" alpha)"; lr="$(cfg "$domain" learning_rate)"
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE" --train-data "data/p0a44/${domain}_train.jsonl" \
    --output-dir "models/checkpoints/p0a44/$domain" \
    --audit "reports/audit/gate_p0a44_${domain}_preflight.json" \
    --max-steps "$max" --checkpoint-steps "$checkpoint" --focus-domain "$domain" \
    --learning-rate "$lr" --lora-rank "$rank" --lora-alpha "$alpha" --dry-run
  require_status "reports/audit/gate_p0a44_${domain}_preflight.json" dry_run_passed
}

preflight() {
  data_build; validate_gpus
  [[ -d "$BASE" ]] || { echo "Missing base: $BASE" >&2; return 1; }
  preflight_domain code
  preflight_domain nlp
  bash -n scripts/run_p0a44.sh
  "$PYTHON_BIN" -m py_compile model_compression/build_p0a44_aligned_data.py \
    scripts/evaluate_p0a44_aligned.py scripts/select_p0a44_hf.py
}

checkpoint_ok() {
  local domain="$1" step="$2" path="models/checkpoints/p0a44/$1/checkpoint-$2"
  [[ -s "$path/adapter_config.json" && -s "$path/trainer_state.json" ]] && \
    [[ -s "$path/adapter_model.safetensors" || -s "$path/adapter_model.bin" ]]
}

train_domain() {
  local domain="$1" max checkpoint rank alpha lr audit output resume=()
  data_build; validate_gpus; preflight_domain "$domain"
  max="$(cfg "$domain" max_steps)"; checkpoint="$(cfg "$domain" checkpoint_steps)"
  rank="$(cfg "$domain" rank)"; alpha="$(cfg "$domain" alpha)"; lr="$(cfg "$domain" learning_rate)"
  audit="reports/audit/gate_p0a44_train_${domain}.json"; output="models/checkpoints/p0a44/$domain"
  if [[ -f "$audit" ]] && [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$audit")" == passed ]]; then
    checkpoint_ok "$domain" "$checkpoint"; checkpoint_ok "$domain" "$max"
    echo "P0-A44 $domain training already complete."
    return 0
  fi
  if [[ -d "$output" ]] && find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then
    resume=(--resume-from-checkpoint auto)
  fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a6_student.py \
      --model-dir "$BASE" --train-data "data/p0a44/${domain}_train.jsonl" \
      --output-dir "$output" --audit "$audit" \
      --max-steps "$max" --checkpoint-steps "$checkpoint" --focus-domain "$domain" \
      --learning-rate "$lr" --lora-rank "$rank" --lora-alpha "$alpha" \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 "${resume[@]}" \
    2>&1 | tee "logs/p0a44_${domain}_train.log"
  require_status "$audit" passed
  checkpoint_ok "$domain" "$checkpoint"; checkpoint_ok "$domain" "$max"
}

train_all() {
  train_domain code
  train_domain nlp
}

wait_endpoint() {
  local pid="$1" required="$2" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$ENDPOINT" "$required" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(x.get('id')) for x in json.load(response).get('data',[])}
need={x for x in sys.argv[2].split(',') if x}
raise SystemExit(0 if need.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

eval_aligned() {
  local dataset="$1" manifest="$2" model_id="$3" label="$4" max_tokens="$5"
  local trace="$AUDIT_ROOT/${label}_trace.jsonl" audit="$AUDIT_ROOT/${label}.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a44_aligned.py \
    --dataset "$dataset" --manifest "$manifest" --endpoint "$ENDPOINT" \
    --model-id "$model_id" --candidate-name "$model_id" --workers 8 \
    --timeout-sec 180 --code-timeout-sec 10 --max-tokens "$max_tokens" \
    --output-trace "$trace" --audit "$audit"
  require_status "$audit" passed
}

validate_hf() {
  require_status reports/audit/gate_p0a44_train_code.json passed
  require_status reports/audit/gate_p0a44_train_nlp.json passed
  checkpoint_ok code 176; checkpoint_ok code 352; checkpoint_ok nlp 64; checkpoint_ok nlp 128
  mkdir -p logs runtime "$AUDIT_ROOT"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "${P0A44_SERVE_GPU:-0}" --port "$PORT" --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a44-base \
    --lora-module "p0a44-code-176=$ROOT/models/checkpoints/p0a44/code/checkpoint-176" \
    --lora-module "p0a44-code-352=$ROOT/models/checkpoints/p0a44/code/checkpoint-352" \
    --lora-module "p0a44-nlp-64=$ROOT/models/checkpoints/p0a44/nlp/checkpoint-64" \
    --lora-module "p0a44-nlp-128=$ROOT/models/checkpoints/p0a44/nlp/checkpoint-128" \
    --max-model-len 2048 --gpu-memory-utilization 0.82 \
    >logs/p0a44_hf_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() { if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "p0a44-base,p0a44-code-176,p0a44-code-352,p0a44-nlp-64,p0a44-nlp-128"
  eval_aligned code data/p0a44/code_validation.jsonl p0a44-base hf_code_base 512
  eval_aligned code data/p0a44/code_validation.jsonl p0a44-code-176 hf_code_176 512
  eval_aligned code data/p0a44/code_validation.jsonl p0a44-code-352 hf_code_352 512
  local model step dataset manifest
  for step in base 64 128; do
    model="p0a44-nlp-$step"; [[ "$step" == base ]] && model=p0a44-base
    for dataset in ceval cmmlu; do
      manifest="data/p0a44/nlp_${dataset}_dev.jsonl"
      eval_aligned "$dataset" "$manifest" "$model" "hf_nlp_${step}_${dataset}" 16
    done
  done
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/select_p0a44_hf.py
  require_status reports/audit/gate_p0a44_hf_selection.json passed
}

convert_adapters() {
  require_status reports/audit/gate_p0a44_hf_selection.json passed
  mkdir -p models/adapters/p0a44
  local domain step source target
  while read -r domain step; do
    [[ "$step" == 0 ]] && continue
    source="models/checkpoints/p0a44/$domain/checkpoint-$step"
    target="models/adapters/p0a44/$domain-step-$step-f16.gguf"
    if [[ ! -s "$target" ]]; then
      "$PYTHON_BIN" external/llama.cpp/convert_lora_to_gguf.py \
        "$source" --base "$BASE" --outtype f16 --outfile "$target"
    fi
  done < <("$PYTHON_BIN" - <<'PY'
import json
d=json.load(open('reports/audit/gate_p0a44_hf_selection.json'))['selected']
for domain in ('code','nlp'):
 print(domain, int(d[domain]['step']))
PY
)
  "$PYTHON_BIN" - <<'PY'
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
root=Path('.'); selected=json.load(open('reports/audit/gate_p0a44_hf_selection.json'))['selected']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); outputs={}
for domain in ('code','nlp'):
 step=int(selected[domain]['step'])
 if step:
  p=root/f'models/adapters/p0a44/{domain}-step-{step}-f16.gguf'
  if not p.is_file(): raise SystemExit(f'Missing converted adapter: {p}')
  outputs[domain]={'step':step,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
d={'gate':'P0-A44-GGUF-ADAPTER-PREPARE','created_ts':datetime.now(timezone.utc).isoformat(),'status':'passed','outputs':outputs,'formal_test_items_loaded':0}
d['report_hash']=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
p=root/'reports/audit/gate_p0a44_gguf_prepare.json';p.write_text(json.dumps(d,indent=2)+'\n')
print(f'Wrote {p} status=passed outputs={outputs}')
PY
  require_status reports/audit/gate_p0a44_gguf_prepare.json passed
}

start_q4_server() {
  local gpu="$1" port="$2" alias="$3" adapter="$4" log="$5" lora=()
  [[ -n "$adapter" ]] && lora=(--lora "$ROOT/$adapter")
  env CUDA_VISIBLE_DEVICES="$gpu" "$LLAMA_SERVER" --model "$BASE_GGUF" "${lora[@]}" \
    --alias "$alias" --host 127.0.0.1 --port "$port" --ctx-size 2048 \
    --threads 8 --parallel 1 --batch-size 32 --ubatch-size 16 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --n-gpu-layers all \
    --no-repack --cache-ram 0 --no-cache-idle-slots --reasoning off --reasoning-format none \
    >"$log" 2>&1 &
  LAST_SERVER_PID=$!
}

wait_q4_endpoint() {
  local pid="$1" port="$2" model="$3" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$port" "$model" >/dev/null 2>&1 <<'PY'
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

eval_q4() {
  local dataset="$1" manifest="$2" port="$3" model="$4" label="$5" tokens="$6"
  local saved_endpoint="$ENDPOINT"
  ENDPOINT="http://127.0.0.1:$port"
  eval_aligned "$dataset" "$manifest" "$model" "$label" "$tokens"
  ENDPOINT="$saved_endpoint"
}

validate_q4() {
  convert_adapters
  [[ -x "$LLAMA_SERVER" && -s "$BASE_GGUF" ]] || { echo "Missing CUDA llama-server or Q4 base" >&2; return 1; }
  local code_step nlp_step code_adapter="" nlp_adapter=""
  read -r code_step nlp_step < <("$PYTHON_BIN" - <<'PY'
import json
d=json.load(open('reports/audit/gate_p0a44_hf_selection.json'))['selected']
print(d['code']['step'],d['nlp']['step'])
PY
)
  [[ "$code_step" != 0 ]] && code_adapter="models/adapters/p0a44/code-step-$code_step-f16.gguf"
  [[ "$nlp_step" != 0 ]] && nlp_adapter="models/adapters/p0a44/nlp-step-$nlp_step-f16.gguf"
  mkdir -p logs "$AUDIT_ROOT"
  local pids=() jobs=()
  start_q4_server 0 18541 p0a44-q4-code-base "" logs/p0a44_q4_code_base_server.log; pids+=("$LAST_SERVER_PID")
  start_q4_server 1 18542 p0a44-q4-code-candidate "$code_adapter" logs/p0a44_q4_code_candidate_server.log; pids+=("$LAST_SERVER_PID")
  start_q4_server 2 18543 p0a44-q4-nlp-base "" logs/p0a44_q4_nlp_base_server.log; pids+=("$LAST_SERVER_PID")
  start_q4_server 3 18544 p0a44-q4-nlp-candidate "$nlp_adapter" logs/p0a44_q4_nlp_candidate_server.log; pids+=("$LAST_SERVER_PID")
  cleanup() { local pid; for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; }
  trap cleanup EXIT INT TERM
  wait_q4_endpoint "${pids[0]}" 18541 p0a44-q4-code-base
  wait_q4_endpoint "${pids[1]}" 18542 p0a44-q4-code-candidate
  wait_q4_endpoint "${pids[2]}" 18543 p0a44-q4-nlp-base
  wait_q4_endpoint "${pids[3]}" 18544 p0a44-q4-nlp-candidate
  eval_q4 code data/p0a44/code_validation.jsonl 18541 p0a44-q4-code-base q4_code_base 512 & jobs+=("$!")
  eval_q4 code data/p0a44/code_validation.jsonl 18542 p0a44-q4-code-candidate q4_code_candidate 512 & jobs+=("$!")
  eval_q4 ceval data/p0a44/nlp_ceval_dev.jsonl 18543 p0a44-q4-nlp-base q4_nlp_base_ceval 16 & jobs+=("$!")
  eval_q4 cmmlu data/p0a44/nlp_cmmlu_dev.jsonl 18543 p0a44-q4-nlp-base q4_nlp_base_cmmlu 16 & jobs+=("$!")
  eval_q4 ceval data/p0a44/nlp_ceval_dev.jsonl 18544 p0a44-q4-nlp-candidate q4_nlp_candidate_ceval 16 & jobs+=("$!")
  eval_q4 cmmlu data/p0a44/nlp_cmmlu_dev.jsonl 18544 p0a44-q4-nlp-candidate q4_nlp_candidate_cmmlu 16 & jobs+=("$!")
  local failures=0 pid; set +e
  for pid in "${jobs[@]}"; do wait "$pid" || failures=$((failures+1)); done
  set -e; (( failures == 0 )) || return 1
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/select_p0a44_q4.py
  require_status reports/audit/gate_p0a44_q4_selection.json passed
}

evaluate_full_dataset() {
  local dataset="$1" port="$2" model_id="$3" shards="$4" index="$5" name="$6"
  "$PYTHON_BIN" scripts/evaluate_chapter2_capability.py \
    --student-url "http://127.0.0.1:$port" --student-model-id "$model_id" \
    --local-model-dir "$BASE" --dataset "$dataset" --use-frozen-final \
    --sample-limit-per-dataset 0 --split-dir data/splits/p0a44_aligned_edge_full \
    --num-shards "$shards" --shard-index "$index" \
    --output-trace "reports/sealed/p0a44/shards/${name}.jsonl" \
    --audit "reports/audit/p0a44/full_${name}.json" \
    --disable-thinking --kv-cache-type q8_0 --prompt-style v15 \
    --max-new-tokens-map cmmlu=16,gsm8k=512,humaneval=512 \
    --timeout-sec 180 --humaneval-timeout-sec 10 --min-accuracy 0
}

full() {
  require_status reports/audit/gate_p0a44_q4_selection.json passed
  "$PYTHON_BIN" scripts/prepare_p0a44_edge_full.py
  require_status reports/audit/gate_p0a44_edge_full_preflight.json passed
  local code_adapter nlp_adapter
  read -r code_adapter nlp_adapter < <("$PYTHON_BIN" - <<'PY'
import json
d=json.load(open('reports/audit/gate_p0a44_q4_selection.json'))['selected']
print(d['code']['adapter'] or '-',d['nlp']['adapter'] or '-')
PY
)
  [[ "$code_adapter" == - ]] && code_adapter=""
  [[ "$nlp_adapter" == - ]] && nlp_adapter=""
  mkdir -p logs reports/sealed/p0a44/shards reports/audit/p0a44
  local pids=() jobs=()
  start_q4_server 0 18545 p0a44-math-base "" logs/p0a44_full_math_server.log; pids+=("$LAST_SERVER_PID")
  start_q4_server 1 18546 p0a44-code-route "$code_adapter" logs/p0a44_full_code_server.log; pids+=("$LAST_SERVER_PID")
  start_q4_server 2 18547 p0a44-nlp-route-0 "$nlp_adapter" logs/p0a44_full_nlp0_server.log; pids+=("$LAST_SERVER_PID")
  start_q4_server 3 18548 p0a44-nlp-route-1 "$nlp_adapter" logs/p0a44_full_nlp1_server.log; pids+=("$LAST_SERVER_PID")
  cleanup() { local pid; for pid in "${jobs[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${jobs[@]}" "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; }
  trap cleanup EXIT INT TERM
  wait_q4_endpoint "${pids[0]}" 18545 p0a44-math-base
  wait_q4_endpoint "${pids[1]}" 18546 p0a44-code-route
  wait_q4_endpoint "${pids[2]}" 18547 p0a44-nlp-route-0
  wait_q4_endpoint "${pids[3]}" 18548 p0a44-nlp-route-1
  echo "P0-A44 selected edge routes ready; starting one official-full run."
  evaluate_full_dataset gsm8k 18545 p0a44-math-base 1 0 math >logs/p0a44_full_math_eval.log 2>&1 & jobs+=("$!")
  evaluate_full_dataset humaneval 18546 p0a44-code-route 1 0 code >logs/p0a44_full_code_eval.log 2>&1 & jobs+=("$!")
  evaluate_full_dataset cmmlu 18547 p0a44-nlp-route-0 2 0 nlp_shard_0 >logs/p0a44_full_nlp0_eval.log 2>&1 & jobs+=("$!")
  evaluate_full_dataset cmmlu 18548 p0a44-nlp-route-1 2 1 nlp_shard_1 >logs/p0a44_full_nlp1_eval.log 2>&1 & jobs+=("$!")
  local failures=0 pid; set +e
  for pid in "${jobs[@]}"; do wait "$pid" || failures=$((failures+1)); done
  set -e; (( failures == 0 )) || return 1
  require_status reports/audit/p0a44/full_math.json passed
  require_status reports/audit/p0a44/full_code.json passed
  require_status reports/audit/p0a44/full_nlp_shard_0.json passed
  require_status reports/audit/p0a44/full_nlp_shard_1.json passed
  cleanup; jobs=(); pids=(); trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/merge_capability_shards.py \
    --input reports/sealed/p0a44/shards/math.jsonl \
    --input reports/sealed/p0a44/shards/code.jsonl \
    --input reports/sealed/p0a44/shards/nlp_shard_0.jsonl \
    --input reports/sealed/p0a44/shards/nlp_shard_1.jsonl \
    --split-dir data/splits/p0a44_aligned_edge_full --role student \
    --model-name "Edge-P0A44-Aligned-Router-Q4_K_M-Q8KV" \
    --output reports/sealed/p0a44/edge_aligned_router_q4_full.jsonl \
    --audit reports/audit/gate_p0a44_edge_aligned_router_q4_full.json
  require_status reports/audit/gate_p0a44_edge_aligned_router_q4_full.json passed
  set +e; "$PYTHON_BIN" scripts/p0a44_retention_gate.py; local result=$?; set -e
  return "$result"
}

guarded_full() {
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh bash scripts/run_p0a44.sh full
}

status() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('reports/audit').glob('gate_p0a44*.json')):
 try:
  d=json.loads(p.read_text()); print(p,d.get('status'),d.get('global_step',''),d.get('selected',''))
 except Exception as e: print(p,'INVALID',e)
PY
  find models/checkpoints/p0a44 -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' -printf '%p\n' 2>/dev/null | sort -V || true
}

case "${1:-help}" in
  data) data_build ;;
  preflight) preflight ;;
  train-code) train_domain code ;;
  train-nlp) train_domain nlp ;;
  train-all) train_all ;;
  validate-hf) validate_hf ;;
  convert-adapters) convert_adapters ;;
  validate-q4) validate_q4 ;;
  full) full ;;
  guarded-full) guarded_full ;;
  status) status ;;
  structural-check) bash -n scripts/run_p0a44.sh; "$PYTHON_BIN" -m py_compile model_compression/build_p0a44_aligned_data.py scripts/evaluate_p0a44_aligned.py scripts/select_p0a44_hf.py scripts/select_p0a44_q4.py scripts/prepare_p0a44_edge_full.py scripts/p0a44_retention_gate.py ;;
  *) echo "Usage: bash scripts/run_p0a44.sh <data|preflight|train-code|train-nlp|train-all|validate-hf|convert-adapters|validate-q4|full|guarded-full|status|structural-check>" ;;
esac
