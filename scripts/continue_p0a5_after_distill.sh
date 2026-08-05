#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DISTILL_PID="${1:?Usage: continue_p0a5_after_distill.sh DISTILL_PID TEACHER_LAUNCHER_PID}"
TEACHER_LAUNCHER_PID="${2:?Usage: continue_p0a5_after_distill.sh DISTILL_PID TEACHER_LAUNCHER_PID}"
GPUS="${P0A5_GPUS:-0,1,2,3}"
EXPECTED_ROWS=36673

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

if [[ ! -r "/proc/$DISTILL_PID/cmdline" ]] \
  || ! tr '\0' ' ' <"/proc/$DISTILL_PID/cmdline" \
    | grep -q 'generate_p0a5_distill.py'; then
  echo "PID $DISTILL_PID is not the active P0-A5 distillation process." >&2
  exit 2
fi

if [[ ! -r "/proc/$TEACHER_LAUNCHER_PID/cmdline" ]] \
  || ! tr '\0' ' ' <"/proc/$TEACHER_LAUNCHER_PID/cmdline" \
    | grep -q 'serve_vllm_teachers.py'; then
  echo "PID $TEACHER_LAUNCHER_PID is not the active Teacher launcher." >&2
  exit 2
fi

log "Waiting for distillation pid=$DISTILL_PID."
while kill -0 "$DISTILL_PID" 2>/dev/null; do
  rows=0
  if [[ -f data/capability_v2/teacher_trace.jsonl ]]; then
    rows="$(wc -l < data/capability_v2/teacher_trace.jsonl)"
  fi
  log "Distillation progress: $rows/$EXPECTED_ROWS."
  sleep 60
done

log "Distillation process exited; validating artifacts."
.venv/bin/python - <<'PY'
import json
from pathlib import Path

expected_rows = 36_673
expected_counts = {
    "gsm8k": 7_173,
    "opencodeinstruct": 20_000,
    "cmmlu": 9_500,
}
audit_path = Path("reports/audit/gate_p0a5_distill.json")
if not audit_path.is_file():
    raise SystemExit(f"Missing distillation audit: {audit_path}")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
checks = {
    "status": audit.get("status") == "passed",
    "rows": audit.get("rows") == expected_rows,
    "counts": audit.get("counts") == expected_counts,
    "teacher_request_error_count": audit.get("teacher_request_error_count") == 0,
    "formal_test_reference_count": audit.get("formal_test_reference_count") == 0,
}
for key, passed in checks.items():
    print(f"{key}: {'passed' if passed else 'failed'}")
if not all(checks.values()):
    raise SystemExit(f"Distillation audit rejected: {checks}")

for name in (
    "data/capability_v2/teacher_trace.jsonl",
    "data/capability_v2/distill_train.jsonl",
):
    path = Path(name)
    if not path.is_file():
        raise SystemExit(f"Missing distillation artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != expected_rows:
        raise SystemExit(f"Unexpected row count: {path} rows={rows}")
    print(f"{path}: {rows} rows")
PY

log "Distillation gate passed; stopping Teacher launcher pid=$TEACHER_LAUNCHER_PID."
if kill -0 "$TEACHER_LAUNCHER_PID" 2>/dev/null; then
  kill -INT "$TEACHER_LAUNCHER_PID"
fi
for _ in $(seq 1 120); do
  if ! kill -0 "$TEACHER_LAUNCHER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$TEACHER_LAUNCHER_PID" 2>/dev/null; then
  echo "Teacher launcher did not stop within 120 seconds." >&2
  exit 1
fi
log "Teacher stopped and GPU resources released."

log "Running Student Candidate 1 preflight."
bash scripts/run_p0a5.sh student-preflight 1

log "Starting four-GPU Student Candidate 1 training."
P0A5_GPUS="$GPUS" \
  bash scripts/run_p0a5.sh student-train 1 \
  2>&1 | tee logs/p0a5_student_candidate_1_train.log

log "Student Candidate 1 training finished."
