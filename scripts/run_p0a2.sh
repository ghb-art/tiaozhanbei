#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
P0-A2 is closed and must not be resumed.

Frozen rejection evidence:
  reports/audit/gate_p0a2_deepseek_rejection.json
  reports/audit/gate_p0a2_deepseek_upper_bound.json
  Math=84.38%, Code=23.81%, NLP=17.19%

Use the capability-first reselection route instead:
  bash scripts/run_p0a3.sh preflight
  bash scripts/run_p0a3.sh teacher-dev
  bash scripts/run_p0a3.sh qwen3-hf-smoke
  bash scripts/run_p0a3.sh qwen3-hf
EOF
exit 1
