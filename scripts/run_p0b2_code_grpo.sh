#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG="configs/p0b2_code_grpo.json"
GPUS="${P0B2_GPUS:-0,1,2,3}"
BASE="models/checkpoints/p0a4/student-shared-merged"
ADAPTER="models/checkpoints/p0b2/code-grpo"
DATA_AUDIT="reports/audit/gate_p0b2_code_data.json"
PREFLIGHT_AUDIT="reports/audit/gate_p0b2_code_preflight.json"
TRAIN_AUDIT="reports/audit/gate_p0b2_code_train.json"
MERGE_AUDIT="reports/audit/gate_p0b2_code_merge.json"
IMATRIX_AUDIT="reports/audit/gate_p0b2_code_imatrix.json"
QUANT_AUDIT="reports/audit/gate_p0b2_code_quantize.json"
GATE_EVAL_AUDIT="reports/audit/gate_p0b2_code_gate300_eval.json"
GATE_RETENTION_AUDIT="reports/audit/gate_p0b2_code_gate300_retention.json"
BEST="$ADAPTER/best"
MERGED="models/checkpoints/p0b2/code-grpo-merged"
Q4="models/quantized/p0b2-code-grpo-q4_k_m.gguf"
F16="models/quantized/p0b2-code-grpo-f16.gguf"
IMATRIX="data/p0b2/imatrix_calibration.txt"
GATE_PORT="${P0B2_GATE_PORT:-18602}"
MODEL_ID="p0b2-code-grpo-q4"
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
 raise SystemExit(f'P0-B2 requires four distinct GPU ids: {values}')
print('P0-B2 GPU group:', ','.join(values))
PY
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0b2_code_data.py --config "$CONFIG"
  require_status "$DATA_AUDIT" passed
}

preflight() {
  data_build
  validate_gpus
  [[ -d "$BASE" ]] || { echo "Missing base model: $BASE" >&2; return 1; }
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0b2_code_grpo.py \
      --config "$CONFIG" --smoke \
      --audit "$PREFLIGHT_AUDIT" \
    2>&1 | tee logs/p0b2_code_grpo_preflight.log
  require_status "$PREFLIGHT_AUDIT" dry_run_passed
}

latest_checkpoint() {
  local output="$1" checkpoint
  [[ -d "$output" ]] || return 0
  while IFS= read -r checkpoint; do
    if [[ -s "$checkpoint/adapter_config.json" \
       && -s "$checkpoint/adapter_model.safetensors" \
       && -s "$checkpoint/optimizer.pt" ]]; then
      printf '%s\n' "$checkpoint"
    fi
  done < <(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-[0-9]*' | sort -V)
}

train() {
  local resume=() checkpoint=""
  preflight
  if [[ -f "$TRAIN_AUDIT" ]] && \
     [[ "$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$TRAIN_AUDIT")" == passed ]]; then
    echo "P0-B2 Code GRPO training already complete."
    return 0
  fi
  checkpoint="$(latest_checkpoint "$ADAPTER" | tail -n 1)"
  if [[ -n "$checkpoint" ]]; then
    resume=(--resume-from-checkpoint "$checkpoint")
    echo "Resuming P0-B2 Code GRPO from $checkpoint"
  fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0b2_code_grpo.py \
      --config "$CONFIG" \
      --audit "$TRAIN_AUDIT" \
      "${resume[@]}" \
    2>&1 | tee -a logs/p0b2_code_grpo_train.log
  require_status "$TRAIN_AUDIT" passed
}

merge() {
  require_status "$TRAIN_AUDIT" passed
  [[ -d "$BEST" ]] || { echo "Missing best adapter: $BEST" >&2; return 1; }
  if [[ -f "$MERGE_AUDIT" && -d "$MERGED" ]]; then
    require_status "$MERGE_AUDIT" passed
    return 0
  fi
  "$PYTHON_BIN" model_compression/merge_lora_adapter.py \
    --base-model "$BASE" --adapter "$BEST" --output "$MERGED" --audit "$MERGE_AUDIT"
  require_status "$MERGE_AUDIT" passed
}

imatrix_corpus() {
  require_status "$TRAIN_AUDIT" passed
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source data/p0b2/code_grpo_train.jsonl --output "$IMATRIX" \
    --audit "$IMATRIX_AUDIT" --stratify-key dataset_key \
    --stratum opencodeinstruct --rows-per-stratum 128 \
    --rows-per-source 384 --seed 20260805
  require_status "$IMATRIX_AUDIT" passed
}

quantize() {
  require_status "$TRAIN_AUDIT" passed
  require_status "$MERGE_AUDIT" passed
  require_status "$IMATRIX_AUDIT" passed
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    "$PYTHON_BIN" scripts/prepare_p0a5_quantized_student.py \
      --candidate 1 --config "$CONFIG" --merged-dir "$MERGED" \
      --corpus "$IMATRIX" --merge-audit "$MERGE_AUDIT" \
      --corpus-audit "$IMATRIX_AUDIT" --train-audit "$TRAIN_AUDIT" \
      --audit "$QUANT_AUDIT" \
      --f16-output "$F16" --imatrix-output data/p0b2/imatrix.gguf \
      --q4-output "$Q4" --gate-name P0-B2-CODE-GRPO-QUANTIZATION \
      --gpu 0 --chunks "${P0B2_IMATRIX_CHUNKS:-170}"
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

gate300() {
  local server_pid="" result=0 endpoint="http://127.0.0.1:$GATE_PORT"
  require_status "$QUANT_AUDIT" passed
  require_status reports/audit/gate_p0a5_baseline14b_gate300_eval_v3.json passed
  mkdir -p logs data/eval reports/audit
  start_server 0 "$GATE_PORT" "$MODEL_ID" logs/p0b2_code_gate300_server.log
  server_pid="$LAST_SERVER_PID"
  cleanup() { kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
  wait_endpoint "$server_pid" "$endpoint" "$MODEL_ID"
  "$PYTHON_BIN" scripts/evaluate_p0a5_gate.py \
    --endpoint "$endpoint" --model-id "$MODEL_ID" --candidate-name "$MODEL_ID" \
    --output-trace data/eval/p0b2_code_grpo_gate300.jsonl \
    --audit "$GATE_EVAL_AUDIT" \
    2>&1 | tee logs/p0b2_code_gate300_eval.log
  set +e
  "$PYTHON_BIN" scripts/p0a5_gate.py --config "$CONFIG" \
    --baseline-trace data/eval/p0a5_baseline14b_gate300_v3.jsonl \
    --student-trace data/eval/p0b2_code_grpo_gate300.jsonl \
    --candidate-name "$MODEL_ID" --output "$GATE_RETENTION_AUDIT"
  result=$?
  set -e
  cleanup; server_pid=""; trap - EXIT INT TERM
  return $result
}

status() {
  echo "P0-B2 Code GRPO status:"
  for audit in "$DATA_AUDIT" "$PREFLIGHT_AUDIT" "$TRAIN_AUDIT" \
               "$MERGE_AUDIT" "$IMATRIX_AUDIT" "$QUANT_AUDIT" \
               "$GATE_EVAL_AUDIT" "$GATE_RETENTION_AUDIT"; do
    if [[ -f "$audit" ]]; then
      "$PYTHON_BIN" - "$audit" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
print(f"  {sys.argv[1]}: status={data.get('status')}")
PY
    else
      echo "  $audit: missing"
    fi
  done
}

case "${1:-}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  merge) merge ;;
  imatrix) imatrix_corpus ;;
  quantize) quantize ;;
  gate300) gate300 ;;
  status) status ;;
  *)
    echo "Usage: $0 {data-build|preflight|train|merge|imatrix|quantize|gate300|status}" >&2
    return 1
    ;;
esac
