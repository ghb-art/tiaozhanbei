#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
P0A_GPUS="${P0A_GPUS:-0,1,2,3}"

checks() {
  "$PYTHON_BIN" scripts/validate_project_structure.py
  "$PYTHON_BIN" scripts/validate_splits.py --check-leakage
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'
}

gpu_preflight() {
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  "$PYTHON_BIN" - "$P0A_GPUS" <<'PY'
import os
import subprocess
import sys

selected = [item.strip() for item in sys.argv[1].replace(" ", ",").split(",") if item.strip()]
rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()
usage = {index.strip(): int(value.strip()) for index, value in (row.split(",") for row in rows)}
missing = [index for index in selected if index not in usage]
busy = {index: usage[index] for index in selected if index in usage and usage[index] > 1024}
if missing or busy:
    raise SystemExit(f"GPU preflight failed: missing={missing} busy_over_1GiB={busy}")
print(f"GPU preflight passed: selected={selected}")
PY
}

db_up() {
  docker compose -f docker/docker-compose.kwdb.yml up -d
}

db_verify() {
  "$PYTHON_BIN" scripts/verify_gate_db.py
}

teacher_serve() {
  gpu_preflight
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$P0A_GPUS" \
    --tensor-parallel-size auto
}

teacher_plan() {
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$P0A_GPUS" \
    --tensor-parallel-size auto \
    --dry-run
}

teacher_stop() {
  "$PYTHON_BIN" - "$ROOT/scripts/serve_vllm_teachers.py" <<'PY'
import os
import signal
import sys
import time
from pathlib import Path

target = str(Path(sys.argv[1]).resolve())
matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if pid == os.getpid():
        continue
    try:
        args = [
            value.decode("utf-8", errors="replace")
            for value in (entry / "cmdline").read_bytes().split(b"\0")
            if value
        ]
        cwd = (entry / "cwd").resolve()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    resolved_args = {
        str((Path(value) if Path(value).is_absolute() else cwd / value).resolve())
        for value in args
    }
    if target in resolved_args:
        matches.append(pid)
if not matches:
    print("Teacher launcher is not running.")
    raise SystemExit(0)
if len(matches) != 1:
    raise SystemExit(f"Refusing to stop ambiguous Teacher launchers: {matches}")
pid = matches[0]
if os.environ.get("P0A_TEACHER_STOP_DRY_RUN") == "1":
    print(f"Teacher launcher matched: pid={pid}")
    raise SystemExit(0)
os.kill(pid, signal.SIGINT)
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Teacher launcher stopped: pid={pid}")
        raise SystemExit(0)
    time.sleep(0.2)
raise SystemExit(f"Teacher launcher did not stop within 30 seconds: pid={pid}")
PY
}

cloud_gate() {
  "$PYTHON_BIN" scripts/verify_gate_cloud.py
}

g0_summary() {
  "$PYTHON_BIN" scripts/run_g0_capmem.py \
    --config configs/g0_capmem_candidates.json \
    --output reports/audit/gate_g0_capmem_current.json
}

g0_run() {
  "$PYTHON_BIN" scripts/run_g0_capmem.py \
    --config configs/g0_capmem_candidates.json \
    --prepare \
    --execute-memory \
    --execute-capability-smoke \
    --output reports/audit/gate_g0_capmem_current.json
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p0a.sh <command>

Current commands:
  checks              Run project, split and unit-test checks.
  gpu-preflight       Ensure selected GPUs exist and are not already occupied.
  db-up               Start the KWDB service.
  db-verify           Verify the live database schema and smoke gate.
  teacher-serve       Start one 14B teacher endpoint tensor-parallel across P0A_GPUS.
  teacher-plan        Print the Teacher GPU/TP/vLLM launch plan without starting it.
  teacher-stop        Gracefully stop the active Teacher launcher and its workers.
  cloud-gate          Verify the Cloud teacher service.
  g0-summary          Summarize current capability-memory candidates.
  g0-run              Prepare and execute current G0 candidates.
  p0a3-<command>      Forward to scripts/run_p0a3.sh; e.g. p0a3-preflight.

The rejected v24-v31 and DeepSeek P0-A2 routes remain immutable audit evidence.
The active capability-first reselection route is documented by run_p0a3.sh.
EOF
}

command="${1:-}"
case "$command" in
  checks) checks ;;
  gpu-preflight) gpu_preflight ;;
  db-up) db_up ;;
  db-verify) db_verify ;;
  teacher-serve) teacher_serve ;;
  teacher-plan) teacher_plan ;;
  teacher-stop) teacher_stop ;;
  cloud-gate) cloud_gate ;;
  g0-summary) g0_summary ;;
  g0-run) g0_run ;;
  p0a3-*) bash scripts/run_p0a3.sh "${command#p0a3-}" ;;
  *) usage; exit 2 ;;
esac
