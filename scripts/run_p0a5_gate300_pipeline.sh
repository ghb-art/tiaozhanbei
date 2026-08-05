#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANDIDATE="${1:-1}"
[[ "$CANDIDATE" == "1" || "$CANDIDATE" == "2" ]] || {
  echo "Candidate must be 1 or 2" >&2
  exit 2
}

BASELINE_PID=""
STUDENT_PID=""

stop_service() {
  local pid="${1:-}"
  local label="${2:-service}"
  local pgid=""
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  echo "Stopping $label pid=$pid pgid=${pgid:-unknown}"
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    kill -INT -- "-$pgid" 2>/dev/null || true
  else
    kill -INT "$pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 90); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  sleep 5
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  stop_service "$STUDENT_PID" "Student"
  stop_service "$BASELINE_PID" "Baseline"
}
trap cleanup EXIT INT TERM

wait_endpoint() {
  local endpoint="$1"
  local pid="$2"
  local label="$3"
  "$ROOT/.venv/bin/python" - "$endpoint" "$pid" "$label" <<'PY'
import json
import os
import sys
import time
from urllib.request import urlopen

endpoint, pid_text, label = sys.argv[1:]
pid = int(pid_text)
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit(f"{label} exited before becoming ready")
    try:
        with urlopen(endpoint.rstrip("/") + "/v1/models", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = [str(item.get("id", "")) for item in payload.get("data", [])]
        if any(ids):
            print(f"{label} ready: {ids}")
            raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit(f"{label} readiness timeout")
PY
}

BASELINE_TRACE="data/eval/p0a5_baseline14b_gate300.jsonl"
BASELINE_AUDIT="reports/audit/gate_p0a5_baseline14b_gate300_eval.json"

baseline_result_is_reusable() {
  "$ROOT/.venv/bin/python" - "$BASELINE_TRACE" "$BASELINE_AUDIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

trace_path, audit_path = map(Path, sys.argv[1:])
if not trace_path.is_file() or not audit_path.is_file():
    raise SystemExit(1)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if (
    audit.get("status") != "passed"
    or audit.get("check_version") != "1.1"
    or audit.get("generation_error_count") != 0
):
    raise SystemExit(1)
rows = [
    json.loads(line)
    for line in trace_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 300:
    raise SystemExit(1)
if any(row.get("capability_eval_version") != "p0a5-gate300-v2" for row in rows):
    raise SystemExit(1)
trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
if trace_hash != audit.get("output_trace_hash"):
    raise SystemExit(1)
PY
}

if ! baseline_result_is_reusable; then
  echo "Starting frozen Baseline-14B-AWQ service."
  P0A5_GPUS="${P0A5_GPUS:-0,1,2,3}" \
    setsid bash scripts/run_p0a5.sh baseline-serve \
    > logs/p0a5_baseline_gate300_server.log 2>&1 &
  BASELINE_PID=$!
  wait_endpoint "http://127.0.0.1:8001" "$BASELINE_PID" "Baseline"
  bash scripts/run_p0a5.sh baseline-gate \
    2>&1 | tee logs/p0a5_baseline_gate300_eval.log
  stop_service "$BASELINE_PID" "Baseline"
  BASELINE_PID=""
else
  echo "Reusing validated frozen baseline gate trace: $BASELINE_TRACE"
fi

echo "Starting Candidate $CANDIDATE Q4_K_M + Q8 KV service."
P0A5_STUDENT_GPU="${P0A5_STUDENT_GPU:-0}" \
  setsid bash scripts/run_p0a5.sh student-serve "$CANDIDATE" \
  > "logs/p0a5_student_candidate_${CANDIDATE}_gate300_server.log" 2>&1 &
STUDENT_PID=$!
wait_endpoint "http://127.0.0.1:18450" "$STUDENT_PID" "Student"
bash scripts/run_p0a5.sh student-gate "$CANDIDATE" \
  2>&1 | tee "logs/p0a5_student_candidate_${CANDIDATE}_gate300_eval.log"
stop_service "$STUDENT_PID" "Student"
STUDENT_PID=""
echo "P0-A5 Candidate $CANDIDATE gate300 pipeline completed."
