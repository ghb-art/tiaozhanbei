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
  CUDA_VISIBLE_DEVICES="$P0A_GPUS" "$PYTHON_BIN" scripts/serve_vllm_teachers.py
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
  teacher-serve       Start the configured 14B teacher endpoints.
  cloud-gate          Verify the Cloud teacher service.
  g0-summary          Summarize current capability-memory candidates.
  g0-run              Prepare and execute current G0 candidates.
  p0a2-<command>      Forward to scripts/run_p0a2.sh; e.g. p0a2-preflight.

The rejected v24-v31 launch matrix was removed. Its immutable audit reports remain
under reports/audit; the active recovery route is documented by run_p0a2.sh.
EOF
}

command="${1:-}"
case "$command" in
  checks) checks ;;
  gpu-preflight) gpu_preflight ;;
  db-up) db_up ;;
  db-verify) db_verify ;;
  teacher-serve) teacher_serve ;;
  cloud-gate) cloud_gate ;;
  g0-summary) g0_summary ;;
  g0-run) g0_run ;;
  p0a2-*) bash scripts/run_p0a2.sh "${command#p0a2-}" ;;
  *) usage; exit 2 ;;
esac
