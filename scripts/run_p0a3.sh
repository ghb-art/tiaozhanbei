#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/external/llama.cpp}"
P0A3_EVAL_GPU="${P0A3_EVAL_GPU:-auto}"
P0A3_TEACHER_URL="${P0A3_TEACHER_URL:-http://127.0.0.1:8000}"
P0A3_TEACHER_MODEL_ID="${P0A3_TEACHER_MODEL_ID:-auto}"

DEV_DATA="data/distill/p0a2_recovery_validation.jsonl"
TRAIN_ONLY_DATA="data/distill/p0a2_recovery_train.jsonl"
TEACHER_MODEL="models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ"
TEACHER_TRACE="data/eval/p0a3_qwen14_teacher_dev.jsonl"
TEACHER_AUDIT="reports/audit/gate_p0a3_qwen14_teacher_dev.json"

QWEN3_MODEL="models/pretrained/Qwen--Qwen3-1.7B"
QWEN3_F16="models/quantized/qwen3-1.7b-f16.gguf"
QWEN3_IMATRIX="models/quantized/qwen3-1.7b-q3_k_m.imatrix"
QWEN3_Q3="models/quantized/qwen3-1.7b-q3_k_m.gguf"
QWEN3_HF_TRACE="data/eval/p0a3_qwen3_1p7b_hf_dev.jsonl"
QWEN3_F16_GGUF_TRACE="data/eval/p0a3_qwen3_1p7b_f16_gguf_f16kv_dev.jsonl"
QWEN3_Q3_TRACE="data/eval/p0a3_qwen3_1p7b_q3_q8kv_dev.jsonl"

QWEN25_MODEL="models/pretrained/Qwen--Qwen2.5-1.5B-Instruct"
QWEN25_F16="models/quantized/qwen2.5-1.5b-f16.gguf"
QWEN25_IMATRIX="models/quantized/qwen2.5-1.5b-q3_k_m.imatrix"
QWEN25_Q3="models/quantized/qwen2.5-1.5b-q3_k_m.gguf"
QWEN25_HF_TRACE="data/eval/p0a3_qwen2p5_1p5b_hf_dev.jsonl"
QWEN25_Q3_TRACE="data/eval/p0a3_qwen2p5_1p5b_q3_dev.jsonl"

CALIBRATION="data/distill/p0a3_q3_imatrix_calibration.txt"
CALIBRATION_AUDIT="reports/audit/gate_p0a3_q3_imatrix_calibration.json"
TOKEN_LIMITS="cmmlu=256,gsm8k=512,humaneval=512"

require_runtime() {
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python runtime: $PYTHON_BIN" >&2; exit 1; }
  [[ -f "$DEV_DATA" ]] || { echo "Missing frozen Dev: $DEV_DATA" >&2; exit 1; }
}

require_path() {
  [[ -e "$1" ]] || { echo "Missing required artifact: $1" >&2; exit 1; }
}

reset_outputs() {
  local path
  for path in "$@"; do
    [[ ! -e "$path" ]] || rm -f -- "$path"
  done
}

resolve_eval_gpu() {
  if [[ "$P0A3_EVAL_GPU" != "auto" ]]; then
    printf '%s\n' "$P0A3_EVAL_GPU"
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "P0A3_EVAL_GPU=auto requires nvidia-smi" >&2
    return 1
  }
  local selected
  selected="$({ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits || true; } | \
    awk -F, 'BEGIN {best=-1; id=""} {gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2+0)>best) {best=$2+0; id=$1}} END {print id}')"
  [[ -n "$selected" ]] || { echo "No CUDA GPU detected" >&2; return 1; }
  echo "Auto-selected evaluation GPU: $selected" >&2
  printf '%s\n' "$selected"
}

ensure_audit_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"Missing audit: {path}")
value = json.loads(path.read_text(encoding="utf-8"))
actual = str(value.get("status", ""))
if actual != expected:
    raise SystemExit(f"Audit status mismatch: {path} actual={actual} expected={expected}")
print(f"Audit guard passed: {path} status={actual}")
PY
}

ensure_memory_margin() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing memory audit: {path}")
value = json.loads(path.read_text(encoding="utf-8"))
peak = value.get("peak_total_memory_mb_decimal")
metric = value.get("memory_gate_metric")
if metric != "peak_process_tree_rss_plus_device_memory_mb_decimal" or peak is None:
    raise SystemExit(f"Incomplete peak-total-memory metric: {path}")
if value.get("status") != "passed" or float(peak) > 1400:
    raise SystemExit(f"P0-A3 Dev memory margin failed: {peak}MB > 1400MB or audit failed")
print(f"P0-A3 Dev memory margin passed: {peak}MB <= 1400MB")
PY
}

build_data() {
  "$PYTHON_BIN" model_compression/build_p0a2_recovery_data.py --config configs/p0a2_recovery.json
}

preflight() {
  require_runtime
  [[ -f "$TRAIN_ONLY_DATA" ]] || build_data
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
data = json.loads((root / "reports/audit/gate_p0a2_recovery_data.json").read_text(encoding="utf-8"))
deepseek = json.loads((root / "reports/audit/gate_p0a2_deepseek_upper_bound.json").read_text(encoding="utf-8"))
rejection = json.loads((root / "reports/audit/gate_p0a2_deepseek_rejection.json").read_text(encoding="utf-8"))
config = json.loads((root / "configs/p0a3_reselection.json").read_text(encoding="utf-8"))
errors = []
if data.get("status") != "passed" or data.get("train_validation_group_overlap_count") != 0:
    errors.append("frozen recovery data gate is not clean")
if data.get("formal_test_reference_count") or data.get("formal_humaneval_prompt_overlap_count"):
    errors.append("formal-test leakage was detected")
expected_counts = {"cmmlu": 64, "gsm8k": 64, "humaneval": 42}
if deepseek.get("dataset_counts") != expected_counts or deepseek.get("sample_count") != 170:
    errors.append("DeepSeek rejection baseline is incomplete")
accuracy = deepseek.get("accuracy_by_dataset", {})
if not (accuracy.get("cmmlu", 1) < 0.25 and accuracy.get("humaneval", 1) < 0.30):
    errors.append("P0-A3 requires the frozen DeepSeek Code/NLP rejection evidence")
if rejection.get("status") != "failed" or rejection.get("decision") != "close_deepseek_recovery_route":
    errors.append("explicit DeepSeek route rejection audit is missing")
if config.get("decision_policy", {}).get("train_before_hf_dev_pass") is not False:
    errors.append("training must remain disabled before HF Dev passes")
if errors:
    raise SystemExit("; ".join(errors))
print(
    "P0-A3 preflight passed: frozen_dev=170 "
    f"deepseek_math={accuracy['gsm8k']:.4f} "
    f"deepseek_code={accuracy['humaneval']:.4f} "
    f"deepseek_nlp={accuracy['cmmlu']:.4f}"
)
PY
}

download_primary() {
  "$PYTHON_BIN" scripts/download_models.py --model primary_edge_candidate
}

download_fallback() {
  fallback_guard
  "$PYTHON_BIN" scripts/download_models.py --model fallback_edge_candidate
}

teacher_dev() {
  require_runtime
  require_path "$TEACHER_MODEL"
  reset_outputs "$TEACHER_TRACE" "$TEACHER_AUDIT"
  "$PYTHON_BIN" scripts/evaluate_edge_candidate_dev.py \
    --endpoint "$P0A3_TEACHER_URL" \
    --endpoint-model-id "$P0A3_TEACHER_MODEL_ID" \
    --model-artifact "$TEACHER_MODEL" \
    --candidate-name qwen2p5-14b-teacher \
    --validation-data "$DEV_DATA" \
    --output-trace "$TEACHER_TRACE" \
    --audit "$TEACHER_AUDIT" \
    --max-new-tokens-map "$TOKEN_LIMITS" \
    --request-timeout-sec 240
}

hf_dev() {
  local candidate_name="$1"
  local model_dir="$2"
  local trace="$3"
  local audit="$4"
  local retention_audit="$5"
  local disable_thinking="$6"
  local eval_status=0
  local eval_gpu
  require_runtime
  require_path "$model_dir"
  require_path "$TEACHER_TRACE"
  ensure_audit_status "$TEACHER_AUDIT" passed
  eval_gpu="$(resolve_eval_gpu)"
  reset_outputs "$trace" "$audit" "$retention_audit"
  local thinking_args=()
  [[ "$disable_thinking" == "true" ]] && thinking_args+=(--disable-thinking)
  CUDA_VISIBLE_DEVICES="$eval_gpu" "$PYTHON_BIN" scripts/evaluate_edge_candidate_dev.py \
    --local-model-dir "$model_dir" \
    --candidate-name "$candidate_name" \
    --validation-data "$DEV_DATA" \
    --output-trace "$trace" \
    --audit "$audit" \
    --max-new-tokens-map "$TOKEN_LIMITS" \
    "${thinking_args[@]}" \
    --device cuda \
    --dtype bfloat16 || eval_status=$?
  [[ -f "$trace" ]] || return "$eval_status"
  "$PYTHON_BIN" scripts/summarize_edge_candidate_dev.py \
    --teacher-trace "$TEACHER_TRACE" \
    --candidate-trace "$trace" \
    --candidate-name "$candidate_name" \
    --min-ratio 0.8 \
    --output "$retention_audit"
}

qwen3_hf_smoke() {
  require_runtime
  require_path "$QWEN3_MODEL"
  local eval_gpu
  local smoke_trace="reports/audit/p0a3_qwen3_1p7b_hf_smoke.jsonl"
  local smoke_audit="reports/audit/gate_p0a3_qwen3_1p7b_hf_smoke.json"
  eval_gpu="$(resolve_eval_gpu)"
  reset_outputs "$smoke_trace" "$smoke_audit"
  CUDA_VISIBLE_DEVICES="$eval_gpu" "$PYTHON_BIN" scripts/evaluate_edge_candidate_dev.py \
    --local-model-dir "$QWEN3_MODEL" \
    --candidate-name qwen3-1p7b-hf-smoke \
    --validation-data "$DEV_DATA" \
    --output-trace "$smoke_trace" \
    --audit "$smoke_audit" \
    --sample-limit-per-dataset 1 \
    --max-new-tokens-map "$TOKEN_LIMITS" \
    --disable-thinking \
    --device cuda \
    --dtype bfloat16
}

qwen3_hf() {
  hf_dev \
    qwen3-1p7b-hf "$QWEN3_MODEL" "$QWEN3_HF_TRACE" \
    reports/audit/gate_p0a3_qwen3_1p7b_hf_dev.json \
    reports/audit/gate_p0a3_qwen3_1p7b_hf_retention.json true
}

fallback_guard() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

candidates = [
    Path("reports/audit/gate_p0a3_qwen3_1p7b_hf_retention.json"),
    Path("reports/audit/gate_p0a3_qwen3_1p7b_q3_q8kv_retention.json"),
]
for path in candidates:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        valid_capability_rejection = (
            value.get("status") == "failed"
            and value.get("complete_frozen_dev") is True
            and value.get("matched_sample_ids") is True
            and value.get("generation_error_count") == 0
            and bool(value.get("ratio_failures"))
        )
        if valid_capability_rejection:
            print(f"Fallback guard passed: primary rejection={path}")
            raise SystemExit(0)
        if value.get("status") == "failed":
            print(
                f"Ignoring incomplete/errored primary audit: {path} "
                f"complete={value.get('complete_frozen_dev')} "
                f"matched={value.get('matched_sample_ids')} "
                f"generation_errors={value.get('generation_error_count')}"
            )
memory = Path("reports/audit/g0_memory_p0a3_qwen3_1p7b_q3_k_m.json")
if memory.is_file():
    value = json.loads(memory.read_text(encoding="utf-8"))
    peak = value.get("peak_total_memory_mb_decimal")
    if value.get("status") == "failed" or (peak is not None and float(peak) > 1400):
        print(f"Fallback guard passed: primary memory rejection={memory} peak={peak}")
        raise SystemExit(0)
raise SystemExit("Fallback is locked until Qwen3 fails HF Dev, Q3 Dev, or the 1400MB Dev memory gate")
PY
}

qwen25_hf() {
  fallback_guard
  hf_dev \
    qwen2p5-1p5b-hf "$QWEN25_MODEL" "$QWEN25_HF_TRACE" \
    reports/audit/gate_p0a3_qwen2p5_1p5b_hf_dev.json \
    reports/audit/gate_p0a3_qwen2p5_1p5b_hf_retention.json false
}

build_calibration() {
  require_path "$TRAIN_ONLY_DATA"
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source "$TRAIN_ONLY_DATA" \
    --output "$CALIBRATION" \
    --audit "$CALIBRATION_AUDIT" \
    --rows-per-source 768 \
    --seed 202606
}

build_candidate_imatrix() {
  local model_dir="$1"
  local f16="$2"
  local imatrix="$3"
  require_path "$model_dir"
  [[ -f "$CALIBRATION" ]] || build_calibration
  if [[ ! -f "$f16" ]]; then
    "$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$model_dir" \
      --outfile "$f16" --outtype f16
  fi
  if [[ ! -f "$imatrix" ]]; then
    local partial="${imatrix}.partial"
    [[ ! -e "$partial" ]] || rm -f "$partial"
    "$LLAMA_CPP_DIR/build/bin/llama-imatrix" \
      --model "$f16" \
      --file "$CALIBRATION" \
      --output "$partial" \
      --chunks 64 \
      --ctx-size 512 \
      --threads 16 \
      --no-ppl \
      --output-frequency 16
    mv "$partial" "$imatrix"
  fi
}

prepare_candidate() {
  local model_dir="$1"
  local f16="$2"
  local imatrix="$3"
  local q3="$4"
  local audit="$5"
  local max_bytes="$6"
  build_candidate_imatrix "$model_dir" "$f16" "$imatrix"
  "$PYTHON_BIN" scripts/prepare_edge_gguf.py \
    --merged-hf-dir "$model_dir" \
    --llama-cpp-dir "$LLAMA_CPP_DIR" \
    --f16-gguf "$f16" \
    --quantized-gguf "$q3" \
    --quant-type Q3_K_M \
    --imatrix "$imatrix" \
    --max-quantized-bytes "$max_bytes" \
    --audit "$audit"
}

prepare_qwen3() {
  ensure_audit_status reports/audit/gate_p0a3_qwen3_1p7b_hf_retention.json passed
  prepare_candidate "$QWEN3_MODEL" "$QWEN3_F16" "$QWEN3_IMATRIX" "$QWEN3_Q3" \
    reports/audit/gate_p0a3_qwen3_1p7b_q3_prepare.json 1150000000
}

prepare_qwen25() {
  fallback_guard
  ensure_audit_status reports/audit/gate_p0a3_qwen2p5_1p5b_hf_retention.json passed
  prepare_candidate "$QWEN25_MODEL" "$QWEN25_F16" "$QWEN25_IMATRIX" "$QWEN25_Q3" \
    reports/audit/gate_p0a3_qwen2p5_1p5b_q3_prepare.json 900000000
}

wait_for_server() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import os
import sys
import time
from urllib.request import Request, urlopen

base_url = sys.argv[1].rstrip("/")
pid = int(sys.argv[2])
deadline = time.time() + 120
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"llama-server exited before becoming healthy: {exc}")
    try:
        with urlopen(Request(base_url + "/health", method="GET"), timeout=2) as response:
            if 200 <= response.status < 300:
                print(f"llama-server healthy: {base_url}")
                raise SystemExit(0)
    except Exception:
        time.sleep(0.5)
raise SystemExit("llama-server health timeout")
PY
}

quantized_dev() {
  local candidate_name="$1"
  local gguf="$2"
  local port="$3"
  local trace="$4"
  local audit="$5"
  local retention_audit="$6"
  local cache_type="${7:-q8_0}"
  require_path "$gguf"
  require_path "$TEACHER_TRACE"
  ensure_audit_status "$TEACHER_AUDIT" passed
  local server_bin="$LLAMA_CPP_DIR/build/bin/llama-server"
  require_path "$server_bin"
  local base_url="http://127.0.0.1:${port}"
  local server_log="logs/p0a3/${candidate_name}_server.log"
  reset_outputs "$trace" "$audit" "$retention_audit"
  mkdir -p "$(dirname "$server_log")"
  "$server_bin" \
    --model "$gguf" \
    --alias "$candidate_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --ctx-size 1536 \
    --threads 8 \
    --parallel 1 \
    --batch-size 32 \
    --ubatch-size 16 \
    --cache-type-k "$cache_type" \
    --cache-type-v "$cache_type" \
    --flash-attn on \
    --n-gpu-layers all \
    --no-repack \
    --reasoning off \
    --reasoning-format none \
    --skip-chat-parsing >"$server_log" 2>&1 &
  local server_pid=$!
  local eval_status=0
  wait_for_server "$base_url" "$server_pid" || eval_status=$?
  if [[ "$eval_status" -eq 0 ]]; then
    "$PYTHON_BIN" scripts/evaluate_edge_candidate_dev.py \
      --endpoint "$base_url" \
      --endpoint-model-id "$candidate_name" \
      --model-artifact "$gguf" \
      --candidate-name "$candidate_name" \
      --validation-data "$DEV_DATA" \
      --output-trace "$trace" \
      --audit "$audit" \
      --max-new-tokens-map "$TOKEN_LIMITS" \
      --kv-cache-type "$cache_type" \
      --disable-thinking \
      --request-timeout-sec 240 || eval_status=$?
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  [[ -f "$trace" ]] || return "$eval_status"
  "$PYTHON_BIN" scripts/summarize_edge_candidate_dev.py \
    --teacher-trace "$TEACHER_TRACE" \
    --candidate-trace "$trace" \
    --candidate-name "$candidate_name" \
    --min-ratio 0.8 \
    --output "$retention_audit"
}

qwen3_q3_dev() {
  quantized_dev qwen3-1p7b-q3-k-m-q8kv "$QWEN3_Q3" 18431 "$QWEN3_Q3_TRACE" \
    reports/audit/gate_p0a3_qwen3_1p7b_q3_q8kv_dev.json \
    reports/audit/gate_p0a3_qwen3_1p7b_q3_q8kv_retention.json q8_0
}

qwen3_f16_control() {
  # Diagnostic only: F16 weights plus F16 KV validate GGUF conversion and the
  # llama.cpp backend without the rejected Q4 KV path. This is not a deployable
  # candidate and does not unlock any formal gate.
  quantized_dev qwen3-1p7b-f16-gguf-f16kv-control "$QWEN3_F16" 18430 \
    "$QWEN3_F16_GGUF_TRACE" \
    reports/audit/gate_p0a3_qwen3_1p7b_f16_gguf_f16kv_dev.json \
    reports/audit/gate_p0a3_qwen3_1p7b_f16_gguf_f16kv_retention.json f16
}

qwen25_q3_dev() {
  fallback_guard
  quantized_dev qwen2p5-1p5b-q3-k-m "$QWEN25_Q3" 18432 "$QWEN25_Q3_TRACE" \
    reports/audit/gate_p0a3_qwen2p5_1p5b_q3_dev.json \
    reports/audit/gate_p0a3_qwen2p5_1p5b_q3_retention.json q8_0
}

memory_gate() {
  local candidate="$1"
  local output="$2"
  "$PYTHON_BIN" scripts/run_g0_capmem.py \
    --config configs/g0_capmem_candidates.json \
    --candidate "$candidate" \
    --execute-memory \
    --output "$output"
  "$PYTHON_BIN" - "$output" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidate = report["candidates"][0]
peak = candidate.get("memory", {}).get("peak_total_memory_mb_decimal")
if peak is None:
    raise SystemExit("Missing complete peak-total-memory metric")
if float(peak) > 1400:
    raise SystemExit(f"P0-A3 Dev memory margin failed: {peak}MB > 1400MB")
print(f"P0-A3 Dev memory margin passed: {peak}MB <= 1400MB")
PY
}

qwen3_memory() {
  ensure_audit_status reports/audit/gate_p0a3_qwen3_1p7b_q3_q8kv_retention.json passed
  memory_gate qwen3-1p7b-q3-k-m reports/audit/gate_g0_capmem_p0a3_qwen3_memory.json
}

qwen25_memory() {
  fallback_guard
  ensure_audit_status reports/audit/gate_p0a3_qwen2p5_1p5b_q3_retention.json passed
  memory_gate qwen2p5-1p5b-q3-k-m reports/audit/gate_g0_capmem_p0a3_qwen2p5_memory.json
}

formal_g0() {
  local candidate="$1"
  local dev_retention="$2"
  local memory_audit="$3"
  local output="$4"
  ensure_audit_status "$dev_retention" passed
  ensure_memory_margin "$memory_audit"
  "$PYTHON_BIN" scripts/run_g0_capmem.py \
    --config configs/g0_capmem_candidates.json \
    --candidate "$candidate" \
    --execute-capability-smoke \
    --output "$output" \
    --require-feasible
}

qwen3_formal_g0() {
  formal_g0 qwen3-1p7b-q3-k-m \
    reports/audit/gate_p0a3_qwen3_1p7b_q3_q8kv_retention.json \
    reports/audit/g0_memory_p0a3_qwen3_1p7b_q3_k_m.json \
    reports/audit/gate_g0_capmem_p0a3_qwen3_final.json
}

qwen25_formal_g0() {
  fallback_guard
  formal_g0 qwen2p5-1p5b-q3-k-m \
    reports/audit/gate_p0a3_qwen2p5_1p5b_q3_retention.json \
    reports/audit/g0_memory_p0a3_qwen2p5_1p5b_q3_k_m.json \
    reports/audit/gate_g0_capmem_p0a3_qwen2p5_final.json
}

checks() {
  "$PYTHON_BIN" scripts/validate_project_structure.py
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'
  preflight
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p0a3.sh <command>

Commands, in order:
  preflight             Verify frozen Dev isolation and DeepSeek rejection evidence.
  download-primary      Download Qwen3-1.7B only.
  download-fallback     Download Qwen2.5-1.5B only, after primary rejection.
  teacher-dev           Evaluate Qwen2.5-14B on the identical frozen 170-row Dev set.
  qwen3-hf-smoke        Run one Dev sample per task with Qwen3 non-thinking mode.
  qwen3-hf              Run full Qwen3 HF Dev and the per-task 80% teacher-relative gate.
  prepare-qwen3         Build train-only imatrix and Q3_K_M only after HF Dev passes.
  qwen3-q3-dev          Evaluate Qwen3 Q3 + Q8 KV on the same 170 rows.
  qwen3-f16-control     Diagnose GGUF with F16 weights and F16 KV on the same 170 rows.
  qwen3-memory          Run 20+100 memory gate and require the 1400MB Dev margin.
  qwen3-formal-g0       Run the single frozen formal capability gate after Dev gates pass.
  qwen25-hf             Fallback HF Dev; requires a recorded Qwen3 Dev/memory rejection.
  prepare-qwen25        Build fallback Q3 only after its HF Dev passes.
  qwen25-q3-dev         Evaluate fallback Q3 on the same 170 rows.
  qwen25-memory         Run fallback 20+100 memory gate with 1400MB Dev margin.
  qwen25-formal-g0      Run fallback formal G0 after every Dev gate passes.
  checks                Run project checks, tests and P0-A3 preflight.

No command trains on the frozen 170-row Dev set. DeepSeek P0-A2 is closed and
formal G1 samples are reached only after a single candidate passes all Dev gates.
EOF
}

case "${1:-}" in
  preflight) preflight ;;
  download-primary) download_primary ;;
  download-fallback) download_fallback ;;
  teacher-dev) teacher_dev ;;
  qwen3-hf-smoke) qwen3_hf_smoke ;;
  qwen3-hf) qwen3_hf ;;
  prepare-qwen3) prepare_qwen3 ;;
  qwen3-q3-dev) qwen3_q3_dev ;;
  qwen3-f16-control) qwen3_f16_control ;;
  qwen3-memory) qwen3_memory ;;
  qwen3-formal-g0) qwen3_formal_g0 ;;
  qwen25-hf) qwen25_hf ;;
  prepare-qwen25) prepare_qwen25 ;;
  qwen25-q3-dev) qwen25_q3_dev ;;
  qwen25-memory) qwen25_memory ;;
  qwen25-formal-g0) qwen25_formal_g0 ;;
  checks) checks ;;
  *) usage; exit 2 ;;
esac
