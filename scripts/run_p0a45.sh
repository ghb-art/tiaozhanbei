#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
CONFIG="configs/p0a45_simple_shared.json"
GPUS="${P0A45_GPUS:-0,1,2,3}"
SERVE_GPU="${P0A45_SERVE_GPU:-0}"
PORT="${P0A45_PORT:-18550}"
ENDPOINT="http://127.0.0.1:$PORT"
MODEL_ID="p0a45-simple-shared-q4"
BASE="models/checkpoints/p0a4/student-shared-merged"
ADAPTER="models/checkpoints/p0a45/shared"
MERGED="models/checkpoints/p0a45/shared-merged"
Q4="models/quantized/p0a45-simple-shared-q4_k_m.gguf"
TRAIN_AUDIT="reports/audit/gate_p0a45_train_shared.json"
MERGE_AUDIT="reports/audit/gate_p0a45_merge_shared.json"
IMATRIX_AUDIT="reports/audit/gate_p0a45_imatrix_calibration.json"
QUANT_AUDIT="reports/audit/gate_p0a45_quantize_shared.json"
EVAL_AUDIT="reports/audit/gate_p0a45_simple_shared_gate300_eval.json"
RETENTION_AUDIT="reports/audit/gate_p0a45_simple_shared_gate300_retention.json"
TRACE="data/eval/p0a45_simple_shared_gate300.jsonl"

require_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); allowed=set(sys.argv[2].split(','))
if not p.is_file(): raise SystemExit(f"Missing audit: {p}")
status=json.loads(p.read_text(encoding='utf-8')).get('status')
if status not in allowed:
 raise SystemExit(f"Audit rejected: {p} status={status} allowed={sorted(allowed)}")
print(f"Audit guard passed: {p} status={status}")
PY
}

validate_gpus() {
  "$PYTHON_BIN" - "$GPUS" <<'PY'
import sys
values=[x.strip() for x in sys.argv[1].split(',') if x.strip()]
if len(values)!=4 or len(set(values))!=4 or not all(x.isdigit() for x in values):
 raise SystemExit(f"P0-A45 requires four distinct GPU ids: {values}")
print("P0-A45 GPU group:", ",".join(values))
PY
}

data_build() {
  "$PYTHON_BIN" model_compression/build_p0a45_simple_data.py
  require_status reports/audit/gate_p0a45_data.json passed
}

preflight() {
  data_build
  validate_gpus
  [[ -d "$BASE" ]] || { echo "Missing base model: $BASE" >&2; return 1; }
  "$PYTHON_BIN" model_compression/train_p0a5_lora.py \
    --config "$CONFIG" --role student --candidate-index 1 \
    --model-dir "$BASE" --train-data data/p0a45/train.jsonl \
    --validation-data data/p0a45/internal_validation.jsonl \
    --output-dir "$ADAPTER" \
    --audit reports/audit/gate_p0a45_train_preflight.json --dry-run
  require_status reports/audit/gate_p0a45_train_preflight.json dry_run_passed
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
    echo "P0-A45 shared training already complete."
    return 0
  fi
  checkpoint="$(latest_checkpoint "$ADAPTER" | tail -n 1)"
  if [[ -n "$checkpoint" ]]; then
    resume=(--resume-from-checkpoint "$checkpoint")
    echo "Resuming P0-A45 from $checkpoint"
  elif [[ -d "$ADAPTER" ]] && find "$ADAPTER" -mindepth 1 -print -quit | grep -q .; then
    echo "P0-A45 output is non-empty but has no complete checkpoint: $ADAPTER" >&2
    return 2
  fi
  mkdir -p logs
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    env CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
      --standalone --nproc_per_node=4 \
      model_compression/train_p0a5_lora.py \
      --config "$CONFIG" --role student --candidate-index 1 \
      --model-dir "$BASE" --train-data data/p0a45/train.jsonl \
      --validation-data data/p0a45/internal_validation.jsonl \
      --output-dir "$ADAPTER" --audit "$TRAIN_AUDIT" \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
      "${resume[@]}" \
    2>&1 | tee logs/p0a45_shared_train.log
  require_status "$TRAIN_AUDIT" passed
}

merge() {
  require_status "$TRAIN_AUDIT" passed
  if [[ -f "$MERGE_AUDIT" ]] && [[ -d "$MERGED" ]]; then
    require_status "$MERGE_AUDIT" passed
    echo "P0-A45 merged model already exists."
    return 0
  fi
  "$PYTHON_BIN" model_compression/merge_lora_adapter.py \
    --base-model "$BASE" --adapter "$ADAPTER" --output "$MERGED" \
    --audit "$MERGE_AUDIT"
  require_status "$MERGE_AUDIT" passed
}

imatrix_corpus() {
  require_status reports/audit/gate_p0a45_data.json passed
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source data/p0a45/train.jsonl \
    --output data/p0a45/imatrix_calibration.txt \
    --audit "$IMATRIX_AUDIT" \
    --stratify-key dataset_key \
    --stratum gsm8k --stratum opencodeinstruct --stratum cmmlu \
    --rows-per-stratum 128 --rows-per-source 384 --seed 20260803
  require_status "$IMATRIX_AUDIT" passed
}

quantize() {
  require_status "$TRAIN_AUDIT" passed
  require_status "$MERGE_AUDIT" passed
  require_status "$IMATRIX_AUDIT" passed
  MEMORY_GUARD_THRESHOLD_PERCENT=60 bash scripts/run_with_memory_guard.sh \
    "$PYTHON_BIN" scripts/prepare_p0a5_quantized_student.py \
      --candidate 1 --config "$CONFIG" \
      --merged-dir "$MERGED" --corpus data/p0a45/imatrix_calibration.txt \
      --merge-audit "$MERGE_AUDIT" --corpus-audit "$IMATRIX_AUDIT" \
      --train-audit "$TRAIN_AUDIT" --audit "$QUANT_AUDIT" \
      --f16-output models/quantized/p0a45-simple-shared-f16.gguf \
      --imatrix-output models/quantized/p0a45-simple-shared-imatrix.gguf \
      --q4-output "$Q4" --gate-name P0-A45-SIMPLE-SHARED-QUANTIZATION \
      --gpu "$SERVE_GPU" --chunks "${P0A45_IMATRIX_CHUNKS:-170}"
  require_status "$QUANT_AUDIT" passed
}

wait_endpoint() {
  local pid="$1"
  "$PYTHON_BIN" - "$ENDPOINT" "$pid" "$MODEL_ID" <<'PY'
import json,os,sys,time
from urllib.request import urlopen
endpoint,pid_text,model_id=sys.argv[1:]
pid=int(pid_text); deadline=time.monotonic()+600
while time.monotonic()<deadline:
 try: os.kill(pid,0)
 except ProcessLookupError: raise SystemExit("P0-A45 server exited before readiness")
 try:
  with urlopen(endpoint+"/v1/models",timeout=2) as response:
   ids={str(x.get("id","")) for x in json.load(response).get("data",[])}
  if model_id in ids:
   print(f"P0-A45 Student ready: {sorted(ids)}")
   raise SystemExit(0)
 except Exception: time.sleep(2)
raise SystemExit("P0-A45 server readiness timeout")
PY
}

gate300() {
  local server_pid="" result=0
  require_status "$QUANT_AUDIT" passed
  require_status reports/audit/gate_p0a5_baseline14b_gate300_eval.json passed
  [[ -s "$Q4" ]] || { echo "Missing Q4 model: $Q4" >&2; return 1; }
  mkdir -p logs data/eval reports/audit
  cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      kill -INT "$server_pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 1
      done
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  env CUDA_VISIBLE_DEVICES="$SERVE_GPU" \
    external/llama.cpp/build/bin/llama-server \
      --model "$ROOT/$Q4" --alias "$MODEL_ID" \
      --host 127.0.0.1 --port "$PORT" \
      --ctx-size 1536 --threads 8 --parallel 1 \
      --batch-size 32 --ubatch-size 16 \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --flash-attn on --n-gpu-layers all --no-repack \
      --cache-ram 0 --no-cache-idle-slots \
      --reasoning off --reasoning-format none \
      >logs/p0a45_gate300_server.log 2>&1 &
  server_pid=$!
  wait_endpoint "$server_pid"
  "$PYTHON_BIN" scripts/evaluate_p0a5_gate.py \
    --endpoint "$ENDPOINT" --model-id "$MODEL_ID" \
    --candidate-name p0a45-simple-shared-q4 \
    --output-trace "$TRACE" --audit "$EVAL_AUDIT" \
    2>&1 | tee logs/p0a45_gate300_eval.log
  set +e
  "$PYTHON_BIN" scripts/p0a5_gate.py \
    --config "$CONFIG" \
    --baseline-trace data/eval/p0a5_baseline14b_gate300.jsonl \
    --student-trace "$TRACE" --candidate-name p0a45-simple-shared-q4 \
    --output "$RETENTION_AUDIT"
  result=$?
  set -e
  cleanup; server_pid=""; trap - EXIT INT TERM
  if [[ "$result" -eq 2 ]]; then return 2; fi
  echo "P0-A45 300-item test completed; capability_status=$([[ $result -eq 0 ]] && echo passed || echo failed)"
}

all() {
  train
  merge
  imatrix_corpus
  quantize
  gate300
}

status() {
  local path
  for path in \
    reports/audit/gate_p0a45_data.json \
    reports/audit/gate_p0a45_train_preflight.json \
    "$TRAIN_AUDIT" "$MERGE_AUDIT" "$IMATRIX_AUDIT" "$QUANT_AUDIT" \
    "$EVAL_AUDIT" "$RETENTION_AUDIT"; do
    if [[ -f "$path" ]]; then
      "$PYTHON_BIN" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[1],d.get("status"),d.get("retention_ratios",d.get("train_counts","")))' "$path"
    else
      echo "$path missing"
    fi
  done
}

structural_check() {
  bash -n scripts/run_p0a45.sh
  "$PYTHON_BIN" -m py_compile \
    model_compression/build_p0a45_simple_data.py \
    scripts/prepare_p0a5_quantized_student.py
}

case "${1:-help}" in
  data-build) data_build ;;
  preflight) preflight ;;
  train) train ;;
  merge) merge ;;
  imatrix-corpus) imatrix_corpus ;;
  quantize) quantize ;;
  gate300) gate300 ;;
  all) all ;;
  status) status ;;
  structural-check) structural_check ;;
  *)
    echo "Usage: bash scripts/run_p0a45.sh <data-build|preflight|train|merge|imatrix-corpus|quantize|gate300|all|status|structural-check>"
    ;;
esac
