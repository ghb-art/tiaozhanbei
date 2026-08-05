#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG="configs/p0b1_converged_shared.json"
GPUS="${P0B1_GPUS:-0,1,2,3}"
SERVE_GPU="${P0B1_SERVE_GPU:-0}"
TEACHER_PORT="${P0B1_TEACHER_PORT:-18600}"
GATE_PORT="${P0B1_GATE_PORT:-18601}"
BASE="models/checkpoints/p0a4/student-shared-merged"
ADAPTER="models/checkpoints/p0b1/shared"
MERGED="models/checkpoints/p0b1/shared-merged"
Q4="models/quantized/p0b1-converged-shared-q4_k_m.gguf"
MODEL_ID="p0b1-converged-shared-q4"
TRAIN_AUDIT="reports/audit/gate_p0b1_train_shared.json"
MERGE_AUDIT="reports/audit/gate_p0b1_merge_shared.json"
IMATRIX_AUDIT="reports/audit/gate_p0b1_imatrix_calibration.json"
QUANT_AUDIT="reports/audit/gate_p0b1_quantize_shared.json"
LLAMA_SERVER="$ROOT/external/llama.cpp/build/bin/llama-server"

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not p.is_file(): raise SystemExit(f'Missing audit: {p}')
status=json.loads(p.read_text(encoding='utf-8')).get('status')
if status not in allowed: raise SystemExit(f'Audit rejected: {p} status={status} allowed={sorted(allowed)}')
print(f'Audit guard passed: {p} status={status}')
PY
}

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[value.strip() for value in sys.argv[1].split(',') if value.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(value.isdigit() for value in values):
 raise SystemExit(f'P0-B1 requires four distinct GPU ids: {values}')
print('P0-B1 GPU group:',','.join(values))
PY
}

wait_endpoint() {
  local pid="$1" endpoint="$2" required="$3" attempt
  for attempt in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || return 1
    if "$PYTHON_BIN" - "$endpoint" "$required" >/dev/null 2>&1 <<'PY'
import json,sys
from urllib.request import urlopen
with urlopen(sys.argv[1]+'/v1/models',timeout=2) as response:
 ids={str(row.get('id','')) for row in json.load(response).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

code_download() {
  "$PYTHON_BIN" model_compression/download_p0b1_code.py
  require_status reports/audit/gate_p0b1_code_download.json passed
}

code_build() {
  code_download
  TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" model_compression/build_p0b1_code_data.py \
    2>&1 | tee -a logs/p0b1_code_build.log
  require_status reports/audit/gate_p0b1_code_data.json passed
}

nlp_requests() {
  "$PYTHON_BIN" model_compression/build_p0b1_nlp_requests.py
  require_status reports/audit/gate_p0b1_nlp_requests.json passed
}

nlp_generate() {
  local server_pid="" endpoint="http://127.0.0.1:$TEACHER_PORT"
  nlp_requests
  if [[ -f reports/audit/gate_p0b1_nlp_teacher_data.json ]]; then
    require_status reports/audit/gate_p0b1_nlp_teacher_data.json passed
    return 0
  fi
  validate_gpus
  mkdir -p logs data/p0b1
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$GPUS" --port "$TEACHER_PORT" \
    --model-dir models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ \
    --quantization awq --tensor-parallel-size 4 \
    --served-model-name p0b1-teacher14b --max-model-len 2048 \
    --gpu-memory-utilization 0.82 >logs/p0b1_teacher_server.log 2>&1 &
  server_pid=$!
  cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$endpoint" p0b1-teacher14b
  "$PYTHON_BIN" model_compression/generate_p0a7_mmlu_chinese.py generate \
    --requests data/distill/p0b1_nlp_teacher_requests.jsonl \
    --trace data/distill/p0b1_nlp_teacher_trace.jsonl \
    --train-output data/p0b1/nlp_new_train.jsonl \
    --validation-output data/p0b1/nlp_new_validation.jsonl \
    --audit reports/audit/gate_p0b1_nlp_teacher_data.json \
    --train-target 20000 --validation-target 1500 \
    --endpoint "$endpoint" --model-id p0b1-teacher14b \
    --fallback-model-id p0b1-teacher14b --workers "${P0B1_TEACHER_WORKERS:-16}" \
    --retries 2 --timeout-sec 180 --seed 20260804 \
    --minimum-domain-equal-quota-ratio 0.05 --minimum-domains 2 \
    2>&1 | tee logs/p0b1_nlp_teacher_generate.log
  require_status reports/audit/gate_p0b1_nlp_teacher_data.json passed
  cleanup; server_pid=""; trap - EXIT INT TERM
}

data_build() {
  require_status reports/audit/gate_p0b1_code_data.json passed
  require_status reports/audit/gate_p0b1_nlp_teacher_data.json passed
  "$PYTHON_BIN" model_compression/build_p0b1_training_data.py
  require_status reports/audit/gate_p0b1_data.json passed
}

preflight() {
  data_build
  validate_gpus
  [[ -d "$BASE" ]] || { echo "Missing base model: $BASE" >&2; return 1; }
  "$PYTHON_BIN" model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role student --candidate-index 1 \
    --model-dir "$BASE" --train-data data/p0b1/train.jsonl \
    --validation-data data/p0b1/internal_validation.jsonl \
    --output-dir "$ADAPTER" \
    --audit reports/audit/gate_p0b1_train_preflight.json --dry-run
  require_status reports/audit/gate_p0b1_train_preflight.json dry_run_passed
}

recover_early_stop() {
  "$PYTHON_BIN" scripts/finalize_p0b1_early_stop.py
  require_status "$TRAIN_AUDIT" passed
}

latest_checkpoint() {
  local output="$1" checkpoint
  [[ -d "$output" ]] || return 0
  while IFS= read -r checkpoint; do
    if [[ -s "$checkpoint/trainer_state.json" \
       && -s "$checkpoint/adapter_config.json" \
       && -s "$checkpoint/adapter_model.safetensors" \
       && -s "$checkpoint/optimizer.pt" \
       && -s "$checkpoint/scheduler.pt" ]]; then
      printf '%s\n' "$checkpoint"
    fi
  done < <(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' | sort -V)
}

train() {
  local resume=() checkpoint=""
  preflight
  if [[ -f "$TRAIN_AUDIT" ]] && \
     [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$TRAIN_AUDIT")" == passed ]]; then
    echo "P0-B1 converged shared training already complete."
    return 0
  fi
  checkpoint="$(latest_checkpoint "$ADAPTER" | tail -n 1)"
  if [[ -n "$checkpoint" ]]; then
    resume=(--resume-from-checkpoint "$checkpoint")
    echo "Resuming P0-B1 from $checkpoint"
  elif [[ -d "$ADAPTER" ]] && find "$ADAPTER" -mindepth 1 -print -quit | grep -q .; then
    echo "P0-B1 output is non-empty but has no complete checkpoint: $ADAPTER" >&2
    return 2
  fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0a5_lora.py \
      --config "$CONFIG" --role student --candidate-index 1 \
      --model-dir "$BASE" --train-data data/p0b1/train.jsonl \
      --validation-data data/p0b1/internal_validation.jsonl \
      --output-dir "$ADAPTER" --audit "$TRAIN_AUDIT" \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
      "${resume[@]}" \
    2>&1 | tee -a logs/p0b1_shared_train.log
  require_status "$TRAIN_AUDIT" passed
}

merge() {
  require_status "$TRAIN_AUDIT" passed
  if [[ -f "$MERGE_AUDIT" && -d "$MERGED" ]]; then
    require_status "$MERGE_AUDIT" passed
    return 0
  fi
  "$PYTHON_BIN" model_compression/merge_lora_adapter.py \
    --base-model "$BASE" --adapter "$ADAPTER" --output "$MERGED" --audit "$MERGE_AUDIT"
  require_status "$MERGE_AUDIT" passed
}

imatrix_corpus() {
  require_status reports/audit/gate_p0b1_data.json passed
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source data/p0b1/train.jsonl --output data/p0b1/imatrix_calibration.txt \
    --audit "$IMATRIX_AUDIT" --stratify-key dataset_key \
    --stratum gsm8k --stratum opencodeinstruct --stratum cmmlu \
    --rows-per-stratum 128 --rows-per-source 384 --seed 20260804
  require_status "$IMATRIX_AUDIT" passed
}

quantize() {
  require_status "$TRAIN_AUDIT" passed
  require_status "$MERGE_AUDIT" passed
  require_status "$IMATRIX_AUDIT" passed
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    "$PYTHON_BIN" scripts/prepare_p0a5_quantized_student.py \
      --candidate 1 --config "$CONFIG" --merged-dir "$MERGED" \
      --corpus data/p0b1/imatrix_calibration.txt --merge-audit "$MERGE_AUDIT" \
      --corpus-audit "$IMATRIX_AUDIT" --train-audit "$TRAIN_AUDIT" \
      --audit "$QUANT_AUDIT" \
      --f16-output models/quantized/p0b1-converged-shared-f16.gguf \
      --imatrix-output models/quantized/p0b1-converged-shared-imatrix.gguf \
      --q4-output "$Q4" --gate-name P0-B1-CONVERGED-SHARED-QUANTIZATION \
      --gpu "$SERVE_GPU" --chunks "${P0B1_IMATRIX_CHUNKS:-170}"
  require_status "$QUANT_AUDIT" passed
}

start_server() {
  local gpu="$1" port="$2" alias="$3" log="$4"
  env CUDA_VISIBLE_DEVICES="$gpu" "$LLAMA_SERVER" \
    --model "$ROOT/$Q4" --alias "$alias" --host 127.0.0.1 --port "$port" \
    --ctx-size 1536 --threads 8 --parallel 1 --batch-size 32 --ubatch-size 16 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --n-gpu-layers all \
    --no-repack --cache-ram 0 --no-cache-idle-slots \
    --reasoning off --reasoning-format none >"$log" 2>&1 &
  LAST_SERVER_PID=$!
}

gate300() {
  local server_pid="" result=0 endpoint="http://127.0.0.1:$GATE_PORT"
  require_status "$QUANT_AUDIT" passed
  require_status reports/audit/gate_p0a5_baseline14b_gate300_eval.json passed
  mkdir -p logs data/eval reports/audit
  start_server "$SERVE_GPU" "$GATE_PORT" "$MODEL_ID" logs/p0b1_gate300_server.log
  server_pid="$LAST_SERVER_PID"
  cleanup() { kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$endpoint" "$MODEL_ID"
  "$PYTHON_BIN" scripts/evaluate_p0a5_gate.py \
    --endpoint "$endpoint" --model-id "$MODEL_ID" --candidate-name "$MODEL_ID" \
    --output-trace data/eval/p0b1_converged_shared_gate300.jsonl \
    --audit reports/audit/gate_p0b1_gate300_eval.json \
    2>&1 | tee logs/p0b1_gate300_eval.log
  set +e
  "$PYTHON_BIN" scripts/p0a5_gate.py --config "$CONFIG" \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace data/eval/p0b1_converged_shared_gate300.jsonl \
    --candidate-name "$MODEL_ID" --output reports/audit/gate_p0b1_gate300_retention.json
  result=$?
  set -e
  cleanup; server_pid=""; trap - EXIT INT TERM
  echo "P0-B1 300-item test completed with decision_rc=$result"
  return "$result"
}

evaluate_full_dataset() {
  local dataset="$1" port="$2" model_id="$3" shards="$4" index="$5" name="$6"
  "$PYTHON_BIN" scripts/evaluate_chapter2_capability.py \
    --student-url "http://127.0.0.1:$port" --student-model-id "$model_id" \
    --local-model-dir "$BASE" --dataset "$dataset" --use-frozen-final \
    --sample-limit-per-dataset 0 --split-dir data/splits/p0a43_edge_best_full \
    --num-shards "$shards" --shard-index "$index" \
    --output-trace "reports/sealed/p0b1/shards/${name}.jsonl" \
    --audit "reports/audit/p0b1/full_${name}.json" \
    --disable-thinking --kv-cache-type q8_0 --prompt-style v15 \
    --max-new-tokens-map cmmlu=16,gsm8k=512,humaneval=512 \
    --timeout-sec 180 --humaneval-timeout-sec 10 --min-accuracy 0
}

full() {
  require_status "$QUANT_AUDIT" passed
  [[ ! -e reports/sealed/p0b1/edge_converged_q4_full.jsonl ]] || {
    echo "P0-B1 sealed full result already exists; repeat refused." >&2; return 2;
  }
  mkdir -p logs reports/sealed/p0b1/shards reports/audit/p0b1
  local pids=() jobs=() alias
  cleanup() {
    local pid
    for pid in "${jobs[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${jobs[@]}" "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  }
  trap cleanup EXIT INT TERM
  for spec in "0 18610 math" "1 18611 code" "2 18612 nlp0" "3 18613 nlp1"; do
    read -r gpu port alias <<<"$spec"
    start_server "$gpu" "$port" "p0b1-$alias" "logs/p0b1_full_${alias}_server.log"
    pids+=("$LAST_SERVER_PID")
  done
  wait_endpoint "${pids[0]}" http://127.0.0.1:18610 p0b1-math
  wait_endpoint "${pids[1]}" http://127.0.0.1:18611 p0b1-code
  wait_endpoint "${pids[2]}" http://127.0.0.1:18612 p0b1-nlp0
  wait_endpoint "${pids[3]}" http://127.0.0.1:18613 p0b1-nlp1
  evaluate_full_dataset gsm8k 18610 p0b1-math 1 0 math >logs/p0b1_full_math_eval.log 2>&1 & jobs+=("$!")
  evaluate_full_dataset humaneval 18611 p0b1-code 1 0 code >logs/p0b1_full_code_eval.log 2>&1 & jobs+=("$!")
  evaluate_full_dataset cmmlu 18612 p0b1-nlp0 2 0 nlp_shard_0 >logs/p0b1_full_nlp0_eval.log 2>&1 & jobs+=("$!")
  evaluate_full_dataset cmmlu 18613 p0b1-nlp1 2 1 nlp_shard_1 >logs/p0b1_full_nlp1_eval.log 2>&1 & jobs+=("$!")
  local failures=0 pid
  set +e
  for pid in "${jobs[@]}"; do wait "$pid" || failures=$((failures+1)); done
  set -e
  (( failures == 0 )) || return 1
  cleanup; jobs=(); pids=(); trap - EXIT INT TERM
  "$PYTHON_BIN" scripts/merge_capability_shards.py \
    --input reports/sealed/p0b1/shards/math.jsonl \
    --input reports/sealed/p0b1/shards/code.jsonl \
    --input reports/sealed/p0b1/shards/nlp_shard_0.jsonl \
    --input reports/sealed/p0b1/shards/nlp_shard_1.jsonl \
    --split-dir data/splits/p0a43_edge_best_full --role student \
    --model-name "Edge-P0B1-Converged-Shared-Q4_K_M-Q8KV" \
    --output reports/sealed/p0b1/edge_converged_q4_full.jsonl \
    --audit reports/audit/gate_p0b1_edge_full.json
  require_status reports/audit/gate_p0b1_edge_full.json passed
  set +e
  "$PYTHON_BIN" scripts/full_retention_gate.py --stage P0-B1 \
    --candidate reports/sealed/p0b1/edge_converged_q4_full.jsonl \
    --output reports/audit/gate_p0b1_edge_full_retention.json
  local result=$?
  set -e
  return "$result"
}

pipeline() {
  code_build
  nlp_generate
  data_build
  train
  merge
  imatrix_corpus
  quantize
  local gate_rc=0 full_rc=0
  gate300 || gate_rc=$?
  full || full_rc=$?
  echo "P0-B1 terminal evaluations complete: gate300_rc=$gate_rc full_rc=$full_rc"
  [[ "$full_rc" -eq 0 ]]
}

status() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
paths=[
 'reports/audit/gate_p0b1_code_download.json','reports/audit/gate_p0b1_code_data.json',
 'reports/audit/gate_p0b1_nlp_requests.json','reports/audit/gate_p0b1_nlp_teacher_data.json',
 'reports/audit/gate_p0b1_data.json','reports/audit/gate_p0b1_train_preflight.json',
 'reports/audit/gate_p0b1_train_shared.json','reports/audit/gate_p0b1_merge_shared.json',
 'reports/audit/gate_p0b1_imatrix_calibration.json','reports/audit/gate_p0b1_quantize_shared.json',
 'reports/audit/gate_p0b1_gate300_retention.json','reports/audit/gate_p0b1_edge_full_retention.json']
for value in paths:
 p=Path(value)
 if not p.is_file(): print(value,'missing'); continue
 d=json.loads(p.read_text()); print(value,d.get('status'),d.get('global_step',''),d.get('retention_ratios',''))
trace=Path('data/distill/p0b1_nlp_teacher_trace.jsonl')
if trace.is_file(): print(trace,'rows',sum(1 for _ in trace.open()))
PY
}

structural_check() {
  bash -n scripts/run_p0b1.sh
  "$PYTHON_BIN" -m py_compile \
    model_compression/download_p0b1_code.py \
    model_compression/build_p0b1_code_data.py \
    model_compression/build_p0b1_nlp_requests.py \
    model_compression/build_p0b1_training_data.py \
    model_compression/train_p0a5_lora.py \
    scripts/finalize_p0b1_early_stop.py
}

case "${1:-help}" in
  code-download) code_download ;;
  code-build) code_build ;;
  nlp-requests) nlp_requests ;;
  nlp-generate) nlp_generate ;;
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  recover-early-stop) recover_early_stop ;;
  merge) merge ;;
  imatrix-corpus) imatrix_corpus ;;
  quantize) quantize ;;
  gate300) gate300 ;;
  full) full ;;
  pipeline) pipeline ;;
  status) status ;;
  structural-check) structural_check ;;
  *) echo "Usage: bash scripts/run_p0b1.sh <code-download|code-build|nlp-requests|nlp-generate|data-build|preflight|train|recover-early-stop|merge|imatrix-corpus|quantize|gate300|full|pipeline|status|structural-check>" ;;
esac
