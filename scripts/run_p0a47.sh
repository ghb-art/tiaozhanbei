#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A47_GPUS:-0,1,2,3}"
BASE="models/checkpoints/p0a4/student-shared-merged"
INIT="models/checkpoints/p0a25/code-failure-repair/checkpoint-192"
OUTPUT="models/checkpoints/p0a47/code"
BASE_GGUF="models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf"
LLAMA_SERVER="$ROOT/external/llama.cpp/build/bin/llama-server"

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not p.is_file(): raise SystemExit(f'Missing audit: {p}')
s=json.loads(p.read_text(encoding='utf-8')).get('status')
if s not in allowed: raise SystemExit(f'Audit rejected: {p} status={s} allowed={sorted(allowed)}')
print(f'Audit guard passed: {p} status={s}')
PY
}

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
x=[v.strip() for v in sys.argv[1].split(',') if v.strip()]
if len(x)!=4 or len(set(x))!=4 or not all(v.isdigit() for v in x): raise SystemExit(f'Need four GPUs: {x}')
print('P0-A47 GPU group:',','.join(x))
PY
}

wait_vllm() {
  local pid="$1" endpoint="$2" required="$3" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$endpoint" "$required" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as r: ids={str(x.get('id')) for x in json.load(r).get('data',[])}
need={x for x in sys.argv[2].split(',') if x}; raise SystemExit(0 if need.issubset(ids) else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

mine() {
  require_status reports/audit/gate_p0a25_train_code.json passed
  require_status reports/audit/gate_p0a44_data.json passed
  if [[ -f reports/audit/p0a47/mining.json ]]; then require_status reports/audit/p0a47/mining.json passed; return 0; fi
  mkdir -p logs reports/audit/p0a47
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "${P0A47_SERVE_GPU:-0}" --port 18600 --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a47-base \
    --lora-module "p0a47-initial=$ROOT/$INIT" --max-model-len 2048 --gpu-memory-utilization 0.82 \
    >logs/p0a47_mining_server.log 2>&1 &
  local server_pid=$!
  cleanup() { if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
  trap cleanup EXIT INT TERM
  wait_vllm "$server_pid" http://127.0.0.1:18600 p0a47-initial
  "$PYTHON_BIN" scripts/mine_p0a47_code.py --endpoint http://127.0.0.1:18600 \
    --model-id p0a47-initial --workers 8 --timeout-sec 180 --code-timeout-sec 10
  cleanup; trap - EXIT INT TERM
  require_status reports/audit/p0a47/mining.json passed
}

curriculum() {
  require_status reports/audit/p0a47/mining.json passed
  "$PYTHON_BIN" model_compression/build_p0a47_code_curriculum.py
  require_status reports/audit/gate_p0a47_curriculum.json passed
}

preflight() {
  curriculum; validate_gpus
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE" --train-data data/p0a47/code_curriculum.jsonl \
    --output-dir "$OUTPUT" --audit reports/audit/gate_p0a47_train_preflight.json \
    --max-steps 256 --checkpoint-steps 128 --focus-domain code \
    --learning-rate 3e-7 --lora-rank 16 --lora-alpha 32 --init-adapter "$INIT" --dry-run
  require_status reports/audit/gate_p0a47_train_preflight.json dry_run_passed
}

checkpoint_ok() {
  local path="$OUTPUT/checkpoint-$1"
  [[ -s "$path/adapter_config.json" && -s "$path/trainer_state.json" ]] && \
    [[ -s "$path/adapter_model.safetensors" || -s "$path/adapter_model.bin" ]]
}

train() {
  local audit=reports/audit/gate_p0a47_train_code.json resume=()
  preflight
  if [[ -f "$audit" ]] && [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$audit")" == passed ]]; then
    checkpoint_ok 128; checkpoint_ok 256; echo 'P0-A47 training already complete.'; return 0
  fi
  if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then resume=(--resume-from-checkpoint auto); fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0a6_student.py --model-dir "$BASE" \
      --train-data data/p0a47/code_curriculum.jsonl --output-dir "$OUTPUT" --audit "$audit" \
      --max-steps 256 --checkpoint-steps 128 --focus-domain code \
      --learning-rate 3e-7 --lora-rank 16 --lora-alpha 32 --init-adapter "$INIT" \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 "${resume[@]}" \
    2>&1 | tee logs/p0a47_code_train.log
  require_status "$audit" passed; checkpoint_ok 128; checkpoint_ok 256
}

eval_hf_code() {
  local endpoint="$1" model="$2" label="$3" audit="reports/audit/p0a44/p0a47_${3}.json"
  if [[ -f "$audit" ]]; then require_status "$audit" passed; return 0; fi
  "$PYTHON_BIN" scripts/evaluate_p0a44_aligned.py --dataset code \
    --manifest data/p0a44/code_validation.jsonl --endpoint "$endpoint" --model-id "$model" \
    --candidate-name "$model" --workers 8 --timeout-sec 180 --code-timeout-sec 10 --max-tokens 512 \
    --output-trace "reports/audit/p0a44/p0a47_${label}_trace.jsonl" --audit "$audit"
}

validate_hf() {
  require_status reports/audit/gate_p0a47_train_code.json passed; checkpoint_ok 128; checkpoint_ok 256
  mkdir -p logs reports/audit/p0a44
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py --gpu-group "${P0A47_SERVE_GPU:-0}" --port 18600 \
    --model-dir "$BASE" --quantization none --tensor-parallel-size 1 --served-model-name p0a47-base \
    --lora-module "p0a47-initial=$ROOT/$INIT" \
    --lora-module "p0a47-code-128=$ROOT/$OUTPUT/checkpoint-128" \
    --lora-module "p0a47-code-256=$ROOT/$OUTPUT/checkpoint-256" \
    --max-model-len 2048 --gpu-memory-utilization 0.82 >logs/p0a47_hf_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() { if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
  trap cleanup EXIT INT TERM
  wait_vllm "$server_pid" http://127.0.0.1:18600 p0a47-initial,p0a47-code-128,p0a47-code-256
  eval_hf_code http://127.0.0.1:18600 p0a47-initial hf_initial
  eval_hf_code http://127.0.0.1:18600 p0a47-code-128 hf_128
  eval_hf_code http://127.0.0.1:18600 p0a47-code-256 hf_256
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/select_p0a47_code.py
  require_status reports/audit/gate_p0a47_hf_selection.json passed
}

convert() {
  require_status reports/audit/gate_p0a47_hf_selection.json passed
  local step target source
  step="$("$PYTHON_BIN" -c 'import json;print(json.load(open("reports/audit/gate_p0a47_hf_selection.json"))["selected"]["step"])')"
  source="$OUTPUT/checkpoint-$step"; target="models/adapters/p0a47/code-step-$step-f16.gguf"
  mkdir -p models/adapters/p0a47
  if [[ ! -s "$target" ]]; then "$PYTHON_BIN" external/llama.cpp/convert_lora_to_gguf.py "$source" --base "$BASE" --outtype f16 --outfile "$target"; fi
  "$PYTHON_BIN" - "$step" "$target" <<'PY'
import hashlib,json,sys
from pathlib import Path
step=int(sys.argv[1]);p=Path(sys.argv[2]);sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
d={'gate':'P0-A47-GGUF-PREPARE','status':'passed','selected_step':step,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
o=Path('reports/audit/gate_p0a47_gguf_prepare.json');o.write_text(json.dumps(d,indent=2)+'\n');print('Wrote',o)
PY
  require_status reports/audit/gate_p0a47_gguf_prepare.json passed
}

LAST_SERVER_PID=""
start_q4() {
  local gpu="$1" port="$2" alias="$3" adapter="$4" log="$5" lora=()
  [[ -n "$adapter" ]] && lora=(--lora "$ROOT/$adapter")
  env CUDA_VISIBLE_DEVICES="$gpu" "$LLAMA_SERVER" --model "$ROOT/$BASE_GGUF" "${lora[@]}" \
    --alias "$alias" --host 127.0.0.1 --port "$port" --ctx-size 2048 --threads 8 --parallel 1 \
    --batch-size 32 --ubatch-size 16 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
    --n-gpu-layers all --no-repack --cache-ram 0 --no-cache-idle-slots --reasoning off --reasoning-format none \
    >"$log" 2>&1 & LAST_SERVER_PID=$!
}

wait_q4() {
  local pid="$1" port="$2" model="$3" attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$port" "$model" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models',timeout=2) as r: ids={str(x.get('id')) for x in json.load(r).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

validate_q4() {
  convert
  local step adapter pids=() jobs=()
  step="$("$PYTHON_BIN" -c 'import json;print(json.load(open("reports/audit/gate_p0a47_hf_selection.json"))["selected"]["step"])')"
  adapter="models/adapters/p0a47/code-step-$step-f16.gguf"; mkdir -p logs
  start_q4 0 18601 p0a47-q4-base '' logs/p0a47_q4_base_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 1 18602 p0a47-q4-code "$adapter" logs/p0a47_q4_code_server.log; pids+=("$LAST_SERVER_PID")
  cleanup() { local pid; for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; }
  trap cleanup EXIT INT TERM
  wait_q4 "${pids[0]}" 18601 p0a47-q4-base; wait_q4 "${pids[1]}" 18602 p0a47-q4-code
  eval_hf_code http://127.0.0.1:18601 p0a47-q4-base q4_base & jobs+=("$!")
  eval_hf_code http://127.0.0.1:18602 p0a47-q4-code q4_candidate & jobs+=("$!")
  local failure=0 pid; set +e; for pid in "${jobs[@]}"; do wait "$pid" || failure=1; done; set -e; ((failure==0))
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" - "$step" "$adapter" <<'PY'
import hashlib,json,sys
from pathlib import Path
step=int(sys.argv[1]);adapter=sys.argv[2]
base=json.load(open('reports/audit/p0a44/p0a47_q4_base.json'));cand=json.load(open('reports/audit/p0a44/p0a47_q4_candidate.json'))
d={'gate':'P0-A47-Q4-CODE-VALIDATION','status':'passed','selected_step':step,'adapter':adapter,'base_correct':base['correct_count'],'candidate_correct':cand['correct_count'],'candidate_gain':cand['correct_count']-base['correct_count'],'mandatory_full_authorized':True}
d['report_hash']=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest();p=Path('reports/audit/gate_p0a47_q4_validation.json');p.write_text(json.dumps(d,indent=2)+'\n');print('Wrote',p,d)
PY
  require_status reports/audit/gate_p0a47_q4_validation.json passed
}

evaluate_full() {
  local dataset="$1" port="$2" model="$3" shards="$4" index="$5" name="$6"
  "$PYTHON_BIN" scripts/evaluate_chapter2_capability.py --student-url "http://127.0.0.1:$port" \
    --student-model-id "$model" --local-model-dir "$BASE" --dataset "$dataset" --use-frozen-final \
    --sample-limit-per-dataset 0 --split-dir data/splits/p0a44_aligned_edge_full \
    --num-shards "$shards" --shard-index "$index" --output-trace "reports/sealed/p0a47/shards/$name.jsonl" \
    --audit "reports/audit/p0a47/full_$name.json" --disable-thinking --kv-cache-type q8_0 --prompt-style v15 \
    --max-new-tokens-map cmmlu=16,gsm8k=512,humaneval=512 --timeout-sec 180 --humaneval-timeout-sec 10 --min-accuracy 0
}

full() {
  require_status reports/audit/gate_p0a47_q4_validation.json passed
  local adapter pids=() jobs=()
  adapter="$("$PYTHON_BIN" -c 'import json;print(json.load(open("reports/audit/gate_p0a47_q4_validation.json"))["adapter"])')"
  mkdir -p logs reports/sealed/p0a47/shards reports/audit/p0a47
  start_q4 0 18610 p0a47-math '' logs/p0a47_full_math_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 1 18611 p0a47-code "$adapter" logs/p0a47_full_code_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 2 18612 p0a47-nlp0 '' logs/p0a47_full_nlp0_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 3 18613 p0a47-nlp1 '' logs/p0a47_full_nlp1_server.log; pids+=("$LAST_SERVER_PID")
  cleanup() { local pid; for pid in "${jobs[@]}" "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${jobs[@]}" "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; }
  trap cleanup EXIT INT TERM
  wait_q4 "${pids[0]}" 18610 p0a47-math; wait_q4 "${pids[1]}" 18611 p0a47-code
  wait_q4 "${pids[2]}" 18612 p0a47-nlp0; wait_q4 "${pids[3]}" 18613 p0a47-nlp1
  evaluate_full gsm8k 18610 p0a47-math 1 0 math >logs/p0a47_full_math_eval.log 2>&1 & jobs+=("$!")
  evaluate_full humaneval 18611 p0a47-code 1 0 code >logs/p0a47_full_code_eval.log 2>&1 & jobs+=("$!")
  evaluate_full cmmlu 18612 p0a47-nlp0 2 0 nlp0 >logs/p0a47_full_nlp0_eval.log 2>&1 & jobs+=("$!")
  evaluate_full cmmlu 18613 p0a47-nlp1 2 1 nlp1 >logs/p0a47_full_nlp1_eval.log 2>&1 & jobs+=("$!")
  local failure=0 pid; set +e; for pid in "${jobs[@]}"; do wait "$pid" || failure=1; done; set -e; ((failure==0))
  cleanup; jobs=(); pids=(); trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/merge_capability_shards.py \
    --input reports/sealed/p0a47/shards/math.jsonl --input reports/sealed/p0a47/shards/code.jsonl \
    --input reports/sealed/p0a47/shards/nlp0.jsonl --input reports/sealed/p0a47/shards/nlp1.jsonl \
    --split-dir data/splits/p0a44_aligned_edge_full --role student --model-name P0-A47-Round1-Q4 \
    --output reports/sealed/p0a47/edge_round1_q4_full.jsonl --audit reports/audit/gate_p0a47_edge_round1_q4_full.json
  set +e
  "$PYTHON_BIN" scripts/full_retention_gate.py --stage P0-A47 \
    --candidate reports/sealed/p0a47/edge_round1_q4_full.jsonl \
    --output reports/audit/gate_p0a47_edge_round1_q4_full_retention.json
  local result=$?; set -e; return "$result"
}

structural_check() {
  bash -n scripts/run_p0a47.sh
  "$PYTHON_BIN" -m py_compile scripts/mine_p0a47_code.py scripts/select_p0a47_code.py \
    scripts/full_retention_gate.py model_compression/build_p0a47_code_curriculum.py
}

case "${1:-help}" in
  mine) mine ;; curriculum) curriculum ;; preflight) preflight ;; train) train ;;
  validate-hf) validate_hf ;; convert) convert ;; validate-q4) validate_q4 ;; full) full ;;
  structural-check) structural_check ;;
  *) echo 'Usage: bash scripts/run_p0a47.sh <mine|curriculum|preflight|train|validate-hf|convert|validate-q4|full|structural-check>' ;;
esac
