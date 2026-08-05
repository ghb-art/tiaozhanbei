#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
GPUS="${P0A48_GPUS:-0,1,2,3}"
BASE="models/checkpoints/p0a4/student-shared-merged"
OUTPUT="models/checkpoints/p0a48/nlp"
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

round_two_guard() {
  require_status reports/audit/gate_p0a47_edge_round1_q4_full_retention.json failed
}

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
x=[v.strip() for v in sys.argv[1].split(',') if v.strip()]
if len(x)!=4 or len(set(x))!=4 or not all(v.isdigit() for v in x): raise SystemExit(f'Need four GPUs: {x}')
print('P0-A48 GPU group:',','.join(x))
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
  round_two_guard
  require_status reports/audit/gate_p0a44_data.json passed
  if [[ -f reports/audit/p0a48/mining.json ]]; then require_status reports/audit/p0a48/mining.json passed; return 0; fi
  mkdir -p logs reports/audit/p0a48
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "${P0A48_SERVE_GPU:-0}" --port 18620 --model-dir "$BASE" \
    --quantization none --tensor-parallel-size 1 --served-model-name p0a48-base \
    --max-model-len 1536 --gpu-memory-utilization 0.80 >logs/p0a48_mining_server.log 2>&1 &
  local server_pid=$!
  cleanup() { if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
  trap cleanup EXIT INT TERM
  wait_vllm "$server_pid" http://127.0.0.1:18620 p0a48-base
  "$PYTHON_BIN" scripts/mine_p0a48_nlp.py --endpoint http://127.0.0.1:18620 --model-id p0a48-base --workers 8
  cleanup; trap - EXIT INT TERM
  require_status reports/audit/p0a48/mining.json passed
}

curriculum() {
  require_status reports/audit/p0a48/mining.json passed
  "$PYTHON_BIN" model_compression/build_p0a48_nlp_curriculum.py
  require_status reports/audit/gate_p0a48_curriculum.json passed
}

checkpoint_ok() {
  local path="$OUTPUT/checkpoint-$1"
  [[ -s "$path/adapter_config.json" && -s "$path/trainer_state.json" ]] && \
    [[ -s "$path/adapter_model.safetensors" || -s "$path/adapter_model.bin" ]]
}

preflight() {
  curriculum; validate_gpus
  "$PYTHON_BIN" model_compression/train_p0a6_student.py \
    --model-dir "$BASE" --train-data data/p0a48/nlp_curriculum.jsonl \
    --output-dir "$OUTPUT" --audit reports/audit/gate_p0a48_train_preflight.json \
    --max-steps 128 --checkpoint-steps 64 --focus-domain nlp \
    --learning-rate 5e-7 --lora-rank 16 --lora-alpha 32 --dry-run
  require_status reports/audit/gate_p0a48_train_preflight.json dry_run_passed
}

train() {
  local audit=reports/audit/gate_p0a48_train_nlp.json resume=()
  preflight
  if [[ -f "$audit" ]] && [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$audit")" == passed ]]; then
    checkpoint_ok 64; checkpoint_ok 128; echo 'P0-A48 training already complete.'; return 0
  fi
  if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -print -quit | grep -q .; then resume=(--resume-from-checkpoint auto); fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0a6_student.py --model-dir "$BASE" \
      --train-data data/p0a48/nlp_curriculum.jsonl --output-dir "$OUTPUT" --audit "$audit" \
      --max-steps 128 --checkpoint-steps 64 --focus-domain nlp \
      --learning-rate 5e-7 --lora-rank 16 --lora-alpha 32 \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 "${resume[@]}" \
    2>&1 | tee logs/p0a48_nlp_train.log
  require_status "$audit" passed; checkpoint_ok 64; checkpoint_ok 128
}

eval_nlp() {
  local endpoint="$1" model="$2" label="$3" dataset manifest audit
  for dataset in ceval cmmlu; do
    manifest="data/p0a44/nlp_${dataset}_dev.jsonl"
    audit="reports/audit/p0a44/p0a48_${label}_${dataset}.json"
    if [[ -f "$audit" ]]; then require_status "$audit" passed; continue; fi
    "$PYTHON_BIN" scripts/evaluate_p0a44_aligned.py --dataset "$dataset" --manifest "$manifest" \
      --endpoint "$endpoint" --model-id "$model" --candidate-name "$model" --workers 8 \
      --timeout-sec 180 --max-tokens 16 --output-trace "reports/audit/p0a44/p0a48_${label}_${dataset}_trace.jsonl" --audit "$audit"
  done
}

validate_hf() {
  require_status reports/audit/gate_p0a48_train_nlp.json passed; checkpoint_ok 64; checkpoint_ok 128
  mkdir -p logs reports/audit/p0a44
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py --gpu-group "${P0A48_SERVE_GPU:-0}" --port 18620 \
    --model-dir "$BASE" --quantization none --tensor-parallel-size 1 --served-model-name p0a48-base \
    --lora-module "p0a48-nlp-64=$ROOT/$OUTPUT/checkpoint-64" \
    --lora-module "p0a48-nlp-128=$ROOT/$OUTPUT/checkpoint-128" \
    --max-model-len 1536 --gpu-memory-utilization 0.80 >logs/p0a48_hf_validation_server.log 2>&1 &
  local server_pid=$!
  cleanup() { if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
  trap cleanup EXIT INT TERM
  wait_vllm "$server_pid" http://127.0.0.1:18620 p0a48-base,p0a48-nlp-64,p0a48-nlp-128
  eval_nlp http://127.0.0.1:18620 p0a48-base hf_base
  eval_nlp http://127.0.0.1:18620 p0a48-nlp-64 hf_64
  eval_nlp http://127.0.0.1:18620 p0a48-nlp-128 hf_128
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/select_p0a48_nlp.py
  require_status reports/audit/gate_p0a48_hf_selection.json passed
}

convert() {
  require_status reports/audit/gate_p0a48_hf_selection.json passed
  local step target source
  step="$("$PYTHON_BIN" -c 'import json;print(json.load(open("reports/audit/gate_p0a48_hf_selection.json"))["selected"]["step"])')"
  source="$OUTPUT/checkpoint-$step"; target="models/adapters/p0a48/nlp-step-$step-f16.gguf"
  mkdir -p models/adapters/p0a48
  if [[ ! -s "$target" ]]; then "$PYTHON_BIN" external/llama.cpp/convert_lora_to_gguf.py "$source" --base "$BASE" --outtype f16 --outfile "$target"; fi
  "$PYTHON_BIN" - "$step" "$target" <<'PY'
import hashlib,json,sys
from pathlib import Path
step=int(sys.argv[1]);p=Path(sys.argv[2]);sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
d={'gate':'P0-A48-GGUF-PREPARE','status':'passed','selected_step':step,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
o=Path('reports/audit/gate_p0a48_gguf_prepare.json');o.write_text(json.dumps(d,indent=2)+'\n');print('Wrote',o)
PY
  require_status reports/audit/gate_p0a48_gguf_prepare.json passed
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
  step="$("$PYTHON_BIN" -c 'import json;print(json.load(open("reports/audit/gate_p0a48_hf_selection.json"))["selected"]["step"])')"
  adapter="models/adapters/p0a48/nlp-step-$step-f16.gguf"; mkdir -p logs
  start_q4 0 18621 p0a48-q4-base '' logs/p0a48_q4_base_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 1 18622 p0a48-q4-nlp "$adapter" logs/p0a48_q4_nlp_server.log; pids+=("$LAST_SERVER_PID")
  cleanup() { local pid; for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; }
  trap cleanup EXIT INT TERM
  wait_q4 "${pids[0]}" 18621 p0a48-q4-base; wait_q4 "${pids[1]}" 18622 p0a48-q4-nlp
  eval_nlp http://127.0.0.1:18621 p0a48-q4-base q4_base & jobs+=("$!")
  eval_nlp http://127.0.0.1:18622 p0a48-q4-nlp q4_candidate & jobs+=("$!")
  local failure=0 pid; set +e; for pid in "${jobs[@]}"; do wait "$pid" || failure=1; done; set -e; ((failure==0))
  cleanup; trap - EXIT INT TERM
  "$PYTHON_BIN" - "$step" "$adapter" <<'PY'
import hashlib,json,sys
from pathlib import Path
step=int(sys.argv[1]);adapter=sys.argv[2]
load=lambda label,name:json.load(open(f'reports/audit/p0a44/p0a48_{label}_{name}.json'))
base={n:load('q4_base',n) for n in ('ceval','cmmlu')};cand={n:load('q4_candidate',n) for n in ('ceval','cmmlu')}
bt=sum(int(x['correct_count']) for x in base.values());ct=sum(int(x['correct_count']) for x in cand.values())
selected_adapter=adapter if ct>bt else ''
d={'gate':'P0-A48-Q4-NLP-SELECTION','status':'passed','selected_step':step,'base_correct_total':bt,'candidate_correct_total':ct,'candidate_gain':ct-bt,'selected_route':'candidate' if selected_adapter else 'base','adapter':selected_adapter,'mandatory_second_full_authorized':True}
d['report_hash']=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest();p=Path('reports/audit/gate_p0a48_q4_selection.json');p.write_text(json.dumps(d,indent=2)+'\n');print('Wrote',p,d)
PY
  require_status reports/audit/gate_p0a48_q4_selection.json passed
}

evaluate_full() {
  local dataset="$1" port="$2" model="$3" shards="$4" index="$5" name="$6"
  "$PYTHON_BIN" scripts/evaluate_chapter2_capability.py --student-url "http://127.0.0.1:$port" \
    --student-model-id "$model" --local-model-dir "$BASE" --dataset "$dataset" --use-frozen-final \
    --sample-limit-per-dataset 0 --split-dir data/splits/p0a44_aligned_edge_full \
    --num-shards "$shards" --shard-index "$index" --output-trace "reports/sealed/p0a48/shards/$name.jsonl" \
    --audit "reports/audit/p0a48/full_$name.json" --disable-thinking --kv-cache-type q8_0 --prompt-style v15 \
    --max-new-tokens-map cmmlu=16,gsm8k=512,humaneval=512 --timeout-sec 180 --humaneval-timeout-sec 10 --min-accuracy 0
}

full() {
  require_status reports/audit/gate_p0a48_q4_selection.json passed
  local adapter pids=() jobs=()
  adapter="$("$PYTHON_BIN" -c 'import json;print(json.load(open("reports/audit/gate_p0a48_q4_selection.json"))["adapter"])')"
  mkdir -p logs reports/sealed/p0a48/shards reports/audit/p0a48
  start_q4 0 18630 p0a48-math '' logs/p0a48_full_math_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 1 18631 p0a48-code '' logs/p0a48_full_code_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 2 18632 p0a48-nlp0 "$adapter" logs/p0a48_full_nlp0_server.log; pids+=("$LAST_SERVER_PID")
  start_q4 3 18633 p0a48-nlp1 "$adapter" logs/p0a48_full_nlp1_server.log; pids+=("$LAST_SERVER_PID")
  cleanup() { local pid; for pid in "${jobs[@]}" "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; for pid in "${jobs[@]}" "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; }
  trap cleanup EXIT INT TERM
  wait_q4 "${pids[0]}" 18630 p0a48-math; wait_q4 "${pids[1]}" 18631 p0a48-code
  wait_q4 "${pids[2]}" 18632 p0a48-nlp0; wait_q4 "${pids[3]}" 18633 p0a48-nlp1
  evaluate_full gsm8k 18630 p0a48-math 1 0 math >logs/p0a48_full_math_eval.log 2>&1 & jobs+=("$!")
  evaluate_full humaneval 18631 p0a48-code 1 0 code >logs/p0a48_full_code_eval.log 2>&1 & jobs+=("$!")
  evaluate_full cmmlu 18632 p0a48-nlp0 2 0 nlp0 >logs/p0a48_full_nlp0_eval.log 2>&1 & jobs+=("$!")
  evaluate_full cmmlu 18633 p0a48-nlp1 2 1 nlp1 >logs/p0a48_full_nlp1_eval.log 2>&1 & jobs+=("$!")
  local failure=0 pid; set +e; for pid in "${jobs[@]}"; do wait "$pid" || failure=1; done; set -e; ((failure==0))
  cleanup; jobs=(); pids=(); trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/merge_capability_shards.py \
    --input reports/sealed/p0a48/shards/math.jsonl --input reports/sealed/p0a48/shards/code.jsonl \
    --input reports/sealed/p0a48/shards/nlp0.jsonl --input reports/sealed/p0a48/shards/nlp1.jsonl \
    --split-dir data/splits/p0a44_aligned_edge_full --role student --model-name P0-A48-Round2-Q4 \
    --output reports/sealed/p0a48/edge_round2_q4_full.jsonl --audit reports/audit/gate_p0a48_edge_round2_q4_full.json
  set +e
  "$PYTHON_BIN" scripts/full_retention_gate.py --stage P0-A48 \
    --candidate reports/sealed/p0a48/edge_round2_q4_full.jsonl \
    --output reports/audit/gate_p0a48_edge_round2_q4_full_retention.json
  local result=$?; set -e; return "$result"
}

structural_check() {
  bash -n scripts/run_p0a48.sh
  "$PYTHON_BIN" -m py_compile scripts/mine_p0a48_nlp.py scripts/select_p0a48_nlp.py \
    scripts/full_retention_gate.py model_compression/build_p0a48_nlp_curriculum.py
}

case "${1:-help}" in
  mine) mine ;; curriculum) curriculum ;; preflight) preflight ;; train) train ;;
  validate-hf) validate_hf ;; convert) convert ;; validate-q4) validate_q4 ;; full) full ;;
  all) mine; curriculum; train; validate_hf; convert; validate_q4; full ;;
  structural-check) structural_check ;;
  *) echo 'Usage: bash scripts/run_p0a48.sh <mine|curriculum|preflight|train|validate-hf|convert|validate-q4|full|all|structural-check>' ;;
esac
