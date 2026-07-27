#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
P0A4R_GPUS="${P0A4R_GPUS:-0,1,2,3}"
P0A4R_EVAL_GPU="${P0A4R_EVAL_GPU:-0}"
CONFIG="configs/p0a4_remediation.json"
TRAINER_CONFIG="configs/p0a4_distillation.json"
BASE_MODEL="models/checkpoints/p0a4/student-shared-v2-merged"
BASE_AUDIT="reports/audit/gate_p0a4_student_shared_v2_merge.json"
CODE_TRAIN="data/distill/p0a4r_code_train.jsonl"
CODE_VALIDATION="data/distill/p0a4r_code_internal_validation.jsonl"
NLP_TRAIN="data/distill/p0a4r_nlp_rationale_train.jsonl"
NLP_VALIDATION="data/distill/p0a4r_nlp_internal_validation.jsonl"
CODE_TRAIN_AUDIT="reports/audit/gate_p0a4r_train_code.json"
CODE_PILOT_TRAIN_AUDIT="reports/audit/gate_p0a4r_train_code_pilot.json"
NLP_TRAIN_AUDIT="reports/audit/gate_p0a4r_train_nlp.json"
CODE_OUTPUT="models/checkpoints/p0a4r/code"
CODE_PILOT_OUTPUT="models/checkpoints/p0a4r/code-pilot"
NLP_OUTPUT="models/checkpoints/p0a4r/nlp"
CODE_SELECTED="models/checkpoints/p0a4r/code-selected"
CODE_PILOT_SELECTED="models/checkpoints/p0a4r/code-pilot-selected"
NLP_SELECTED="models/checkpoints/p0a4r/nlp-selected"
ROUTER_MANIFEST="models/adapters/p0a4r/router_manifest.json"
ROUTER_AUDIT="reports/audit/gate_p0a4r_adapter_router_prepare.json"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; return 1; }
}

require_audit_status() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); allowed=set(sys.argv[2].split(","))
if not path.is_file():
    raise SystemExit(f"Missing audit: {path}")
audit=json.loads(path.read_text(encoding="utf-8"))
if str(audit.get("status")) not in allowed:
    raise SystemExit(f"Audit status is not allowed: {path} status={audit.get('status')} allowed={sorted(allowed)}")
print(f"Audit guard passed: {path} status={audit.get('status')}")
PY
}

config_training_args() {
  local role="$1"
  "$PYTHON_BIN" - "$CONFIG" "$role" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["training"][sys.argv[2]]
print(d["rank"],d["alpha"],d["dropout"],d["learning_rate"],d["epochs"],d["max_seq_length"],d["gradient_accumulation_steps"])
PY
}

preflight() {
  require_file "$CONFIG"
  require_dir "$BASE_MODEL"
  require_audit_status "$BASE_AUDIT" passed
  require_file data/distill/p0a4_train.jsonl
  require_file data/distill/mbpp_v23_dev_gate.jsonl
  require_file reports/audit/p0a4_trial_ledger.json
  "$PYTHON_BIN" - "$CONFIG" reports/audit/p0a4_trial_ledger.json "$BASE_AUDIT" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
config_path,ledger_path,base_audit_path=map(Path,sys.argv[1:])
config=json.loads(config_path.read_text(encoding="utf-8"))
ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
policy=config["policy"]
if policy.get("feedback_source")!="train_only_internal_validation":
    raise SystemExit("P0-A4R feedback must be train-only")
for key in ("smoke96_item_feedback_used","selection170_feedback_used","formal_full_feedback_used"):
    if policy.get(key) is not False:
        raise SystemExit(f"Forbidden remediation feedback enabled: {key}")
trials=[x for x in ledger.get("trials",[]) if x.get("phase")=="selection170"]
if len(trials)!=1 or trials[0].get("version")!="v1-q4_k_m" or trials[0].get("status")!="failed":
    raise SystemExit(f"P0-A4R requires the final 170 slot to remain unused: {trials}")
audit={
 "gate":"P0-A4R-PROTOCOL","check_version":"1.0","status":"passed",
 "created_by":"scripts/run_p0a4r.sh:preflight","created_ts":datetime.now(timezone.utc).isoformat(),
 "policy":policy,"remaining_selection170_slots":1,
 "base_model":config["models"]["student_base"],
 "base_model_audit":str(base_audit_path),
 "base_model_audit_hash":hashlib.sha256(base_audit_path.read_bytes()).hexdigest(),
}
audit["report_hash"]=hashlib.sha256(json.dumps(audit,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
path=Path("reports/audit/gate_p0a4r_protocol.json")
path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"P0-A4R protocol passed: {path}")
PY
}

nlp_generate() {
  require_audit_status reports/audit/gate_p0a4r_protocol.json passed
  local round
  for round in 1 2; do
    echo "P0-A4R NLP verified-rationale pass ${round}/2"
    if "$PYTHON_BIN" model_compression/generate_p0a4r_nlp_rationales.py \
      --config "$CONFIG" --resume --retry-rejected; then
      return 0
    fi
    echo "NLP rationale gate is still below threshold; retrying only incomplete groups." >&2
  done
  return 1
}

code_build() {
  require_audit_status reports/audit/gate_p0a4r_protocol.json passed
  local extra=()
  local value
  if [[ -n "${P0A4R_EXTRA_CODE_SOURCES:-}" ]]; then
    IFS=',' read -r -a values <<<"$P0A4R_EXTRA_CODE_SOURCES"
    for value in "${values[@]}"; do
      [[ -n "$value" ]] && extra+=(--extra-source "$value")
    done
  fi
  "$PYTHON_BIN" model_compression/build_p0a4r_code_data.py \
    --config "$CONFIG" --workers "${P0A4R_CODE_VERIFY_WORKERS:-8}" "${extra[@]}"
}

code_source_download() {
  "$PYTHON_BIN" model_compression/rebuild_p0a4r_apps_data.py download
}

code_source_rebuild() {
  "$PYTHON_BIN" model_compression/rebuild_p0a4r_apps_data.py all \
    --workers "${P0A4R_CODE_VERIFY_WORKERS:-12}" \
    --target-unique "${P0A4R_APPS_TARGET_UNIQUE:-1500}"
}

code_source_status() {
  "$PYTHON_BIN" model_compression/rebuild_p0a4r_apps_data.py status
}

require_code_data() {
  local scope="$1"
  "$PYTHON_BIN" - reports/audit/gate_p0a4r_code_data.json "$scope" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); scope=sys.argv[2]
d=json.loads(path.read_text(encoding="utf-8"))
if d.get("status")!="passed":
    raise SystemExit(f"Code data gate did not pass: {d.get('status')}")
if scope=="promotion" and d.get("promotion_eligible") is not True:
    raise SystemExit(
      "Code data is pilot-only. Provide or rebuild a standardized APPS/CodeContests train-only JSONL "
      f"and rebuild until unique_groups>={d.get('promotion_min_unique_train_groups')}."
    )
print(f"Code data guard passed: scope={d.get('training_scope')} groups={d.get('training_unique_group_count')}")
PY
}

require_router_lineage() {
  "$PYTHON_BIN" - \
    reports/audit/gate_p0a4r_code_data.json \
    "$CODE_TRAIN_AUDIT" \
    reports/audit/gate_p0a4r_code_checkpoint_selection.json \
    reports/audit/gate_p0a4r_nlp_rationale_data.json \
    "$NLP_TRAIN_AUDIT" \
    reports/audit/gate_p0a4r_nlp_checkpoint_selection.json <<'PY'
import json,sys
from pathlib import Path

paths=list(map(Path,sys.argv[1:]))
for path in paths:
    if not path.is_file():
        raise SystemExit(f"Missing router-lineage audit: {path}")
code_data,code_train,code_select,nlp_data,nlp_train,nlp_select=[
    json.loads(path.read_text(encoding="utf-8")) for path in paths
]
for path,audit in zip(paths,(code_data,code_train,code_select,nlp_data,nlp_train,nlp_select)):
    if audit.get("status")!="passed":
        raise SystemExit(f"Router-lineage audit did not pass: {path} status={audit.get('status')}")
if code_data.get("promotion_eligible") is not True:
    raise SystemExit("Pilot-only Code data cannot enter the deployment router")

checks={
    "code training-data hash": (code_train.get("train_data_hash"),code_data.get("train_hash")),
    "code validation hash": (
        code_train.get("validation_data_hash"),code_data.get("internal_validation_hash")
    ),
    "code selection-data hash": (
        code_select.get("selection_data_hash"),code_data.get("internal_validation_hash")
    ),
    "nlp training-data hash": (nlp_train.get("train_data_hash"),nlp_data.get("train_hash")),
    "nlp validation hash": (
        nlp_train.get("validation_data_hash"),nlp_data.get("internal_validation_hash")
    ),
    "nlp selection-data hash": (
        nlp_select.get("selection_data_hash"),nlp_data.get("internal_validation_hash")
    ),
}
for label,(actual,expected) in checks.items():
    if not actual or actual!=expected:
        raise SystemExit(f"Router-lineage mismatch for {label}: {actual!r} != {expected!r}")

for label,train,selection in (
    ("Code",code_train,code_select),
    ("NLP",nlp_train,nlp_select),
):
    selected=selection.get("selected_checkpoint")
    candidates=set(train.get("checkpoint_candidates",[]))
    if not selected or selected not in candidates:
        raise SystemExit(
            f"{label} selected checkpoint is not in the matching training audit: {selected!r}"
        )
print("P0-A4R router lineage passed.")
PY
}

train_task() {
  local task="$1" role="$2" train_data="$3" validation_data="$4" output="$5" audit="$6"
  local args
  read -r -a args <<<"$(config_training_args "$role")"
  CUDA_VISIBLE_DEVICES="$P0A4R_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a4_lora.py \
    --config "$TRAINER_CONFIG" --role student_expert --candidate-index 2 --task "$task" \
    --model-dir "$BASE_MODEL" --train-data "$train_data" --validation-data "$validation_data" \
    --output-dir "$output" --audit "$audit" \
    --rank "${args[0]}" --lora-alpha "${args[1]}" --lora-dropout "${args[2]}" \
    --learning-rate "${args[3]}" --epochs "${args[4]}" --max-seq-length "${args[5]}" \
    --gradient-accumulation-steps "${args[6]}" --external-checkpoint-selection
}

train_code() {
  local scope="${1:-promotion}"
  require_audit_status reports/audit/gate_p0a4r_protocol.json passed
  require_code_data "$scope"
  if [[ "$scope" == "pilot" ]]; then
    train_task humaneval code_adapter "$CODE_TRAIN" "$CODE_VALIDATION" \
      "$CODE_PILOT_OUTPUT" "$CODE_PILOT_TRAIN_AUDIT"
  else
    train_task humaneval code_adapter "$CODE_TRAIN" "$CODE_VALIDATION" \
      "$CODE_OUTPUT" "$CODE_TRAIN_AUDIT"
  fi
}

train_nlp() {
  require_audit_status reports/audit/gate_p0a4r_protocol.json passed
  require_audit_status reports/audit/gate_p0a4r_nlp_rationale_data.json passed
  train_task cmmlu nlp_adapter "$NLP_TRAIN" "$NLP_VALIDATION" "$NLP_OUTPUT" "$NLP_TRAIN_AUDIT"
}

evaluate_one() {
  local task="$1" validation="$2" name="$3" trace="$4" audit="$5"
  shift 5
  CUDA_VISIBLE_DEVICES="$P0A4R_EVAL_GPU" "$PYTHON_BIN" scripts/evaluate_edge_candidate_dev.py \
    --local-model-dir "$BASE_MODEL" --candidate-name "$name" \
    --validation-data "$validation" --output-trace "$trace" --audit "$audit" \
    --device cuda:0 --dtype bfloat16 --disable-thinking \
    --max-new-tokens-map "humaneval=512,cmmlu=16" --min-accuracy-map "$task=0" "$@"
}

evaluate_checkpoints() {
  local task="$1" validation="$2" train_audit="$3" prefix="$4"
  require_audit_status "$train_audit" passed
  mkdir -p data/eval/p0a4r reports/audit/p0a4r
  evaluate_one "$task" "$validation" "${prefix}-base" \
    "data/eval/p0a4r/${prefix}_base.jsonl" "reports/audit/p0a4r/${prefix}_base.json"
  mapfile -t checkpoints < <("$PYTHON_BIN" - "$train_audit" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for value in d.get("checkpoint_candidates",[]): print(value)
PY
)
  [[ "${#checkpoints[@]}" -gt 0 ]] || { echo "No checkpoint candidates in $train_audit" >&2; return 1; }
  local checkpoint step
  for checkpoint in "${checkpoints[@]}"; do
    step="${checkpoint##*-}"
    evaluate_one "$task" "$validation" "${prefix}-checkpoint-${step}" \
      "data/eval/p0a4r/${prefix}_checkpoint_${step}.jsonl" \
      "reports/audit/p0a4r/${prefix}_checkpoint_${step}.json" \
      --adapter-dir "$checkpoint"
  done
}

evaluate_code() {
  evaluate_checkpoints humaneval "$CODE_VALIDATION" "$CODE_TRAIN_AUDIT" code
}

evaluate_code_rescue() {
  require_audit_status "$CODE_TRAIN_AUDIT" passed
  local checkpoint step scale tag
  checkpoint="$("$PYTHON_BIN" - "$CODE_TRAIN_AUDIT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
values=d.get("checkpoint_candidates",[])
if not values:
    raise SystemExit("Code training audit has no checkpoint candidates")
print(values[-1])
PY
)"
  step="${checkpoint##*-}"
  mapfile -t scales < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for value in d["selection"]["code_adapter_scale_candidates"]:
    print(value)
PY
)
  [[ "${#scales[@]}" -eq 2 ]] || {
    echo "Code rescue requires exactly two predeclared adapter scales." >&2
    return 1
  }
  for scale in "${scales[@]}"; do
    tag="${scale/./p}"
    evaluate_one humaneval "$CODE_VALIDATION" "code-rescue-scale-${scale}" \
      "data/eval/p0a4r/code_checkpoint_${step}_scale_${tag}.jsonl" \
      "reports/audit/p0a4r/code_checkpoint_${step}_scale_${tag}.json" \
      --adapter-dir "$checkpoint" --adapter-scale "$scale"
  done
}

evaluate_code_pilot() {
  evaluate_checkpoints humaneval "$CODE_VALIDATION" "$CODE_PILOT_TRAIN_AUDIT" code_pilot
}

evaluate_nlp() {
  evaluate_checkpoints cmmlu "$NLP_VALIDATION" "$NLP_TRAIN_AUDIT" nlp
}

select_task() {
  local task="$1" prefix="$2" output="$3"
  local baseline="reports/audit/p0a4r/${prefix}_base.json"
  mapfile -t audits < <(find reports/audit/p0a4r -maxdepth 1 -type f \
    -name "${prefix}_checkpoint_*.json" -print | sort -V)
  [[ "${#audits[@]}" -gt 0 ]] || { echo "No candidate evaluations for $prefix" >&2; return 1; }
  local candidate_args=() audit
  for audit in "${audits[@]}"; do candidate_args+=(--candidate-audit "$audit"); done
  "$PYTHON_BIN" scripts/select_p0a4r_checkpoint.py --config "$CONFIG" --task "$task" \
    --baseline-audit "$baseline" "${candidate_args[@]}" \
    --output-dir "$output" --audit "reports/audit/gate_p0a4r_${prefix}_checkpoint_selection.json"
}

select_code() {
  select_task humaneval code "$CODE_SELECTED"
}

select_code_pilot() {
  select_task humaneval code_pilot "$CODE_PILOT_SELECTED"
}

select_nlp() {
  select_task cmmlu nlp "$NLP_SELECTED"
}

prepare_router() {
  require_code_data promotion
  require_audit_status reports/audit/gate_p0a4r_code_checkpoint_selection.json passed
  require_audit_status reports/audit/gate_p0a4r_nlp_checkpoint_selection.json passed
  require_router_lineage
  require_dir models/checkpoints/p0a4/student-expert-gsm8k-v2
  "$PYTHON_BIN" scripts/prepare_p0a4_adapters.py \
    --adapter gsm8k=models/checkpoints/p0a4/student-expert-gsm8k-v2 \
    --adapter humaneval="$CODE_SELECTED" --adapter cmmlu="$NLP_SELECTED" \
    --base "$BASE_MODEL" --base-audit "$BASE_AUDIT" \
    --output-dir models/adapters/p0a4r --manifest "$ROUTER_MANIFEST" --audit "$ROUTER_AUDIT"
}

edge_start() {
  require_audit_status "$ROUTER_AUDIT" passed
  P0A4_STUDENT_VERSION=2 \
  P0A4_ADAPTER_ROUTER_MANIFEST="$ROUTER_MANIFEST" \
  P0A4_ADAPTER_PREPARE_AUDIT="$ROUTER_AUDIT" \
  P0A4_ADAPTER_ROUTE_TAG=adapter_remediation \
    bash scripts/run_p0a4.sh edge-start-adapters
}

smoke96() {
  local report="reports/audit/gate_p0a4_edge_student_v2_adapter_remediation_smoke96_retention.json"
  [[ ! -e "$report" ]] || {
    echo "P0-A4R smoke96 aggregate has already been consumed: $report" >&2
    return 2
  }
  P0A4_STUDENT_VERSION=2 \
  P0A4_ADAPTER_ROUTER_MANIFEST="$ROUTER_MANIFEST" \
  P0A4_ADAPTER_PREPARE_AUDIT="$ROUTER_AUDIT" \
  P0A4_ADAPTER_ROUTE_TAG=adapter_remediation \
    bash scripts/run_p0a4.sh student-adapter-smoke96
}

edge_stop() {
  P0A4_STUDENT_VERSION=2 bash scripts/run_p0a4.sh edge-stop
}

checks() {
  bash -n scripts/run_p0a4r.sh
  "$PYTHON_BIN" -m py_compile \
    model_compression/rebuild_p0a4r_apps_data.py \
    model_compression/generate_p0a4r_nlp_rationales.py \
    model_compression/build_p0a4r_code_data.py \
    scripts/select_p0a4r_checkpoint.py
  "$PYTHON_BIN" -m unittest \
    tests.test_p0a4r_apps_rebuild \
    tests.test_p0a4r_remediation
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p0a4r.sh <command>

  preflight             Freeze train-only feedback and verify the final 170 slot is unused.
  nlp-generate          Resume accepted NLP data and retry only incomplete rationale groups.
  code-source-download  Download, hash-check and extract only the official APPS train split.
  code-source-rebuild   Rebuild 1500 unique executable APPS train tasks.
  code-source-status    Show APPS source and rebuilt artifact status.
  code-build            Build unique executable Code train/dev data; missing extra data => pilot-only.
  train-code            Require >=1000 unique Code groups and train a low-rank adapter on 4 GPUs.
  train-code-pilot      Train the immediate 292-group MBPP methodology pilot; cannot be promoted.
  train-nlp             Train rationale/direct-label NLP adapter on 4 GPUs.
  eval-code             Evaluate every Code checkpoint on MBPP dev_gate execution.
  eval-code-rescue      Evaluate two predeclared lower-strength Code LoRA scales.
  eval-code-pilot       Evaluate only the isolated 292-group pilot checkpoints.
  eval-nlp              Evaluate every NLP checkpoint on held-out train-only choices.
  select-code           Publish only a Code checkpoint that improves internal execution.
  select-code-pilot     Select only an isolated pilot checkpoint; router cannot consume it.
  select-nlp            Publish only an NLP checkpoint that improves internal choice accuracy.
  prepare-router        Convert selected Code/NLP plus existing Math adapter to GGUF Top-1 routes.
  edge-start|smoke96|edge-stop
  checks

P0A4R_EXTRA_CODE_SOURCES is a comma-separated list of standardized train-only JSONL files.
No command reads smoke96 item outputs, selection170 outputs, or official-full traces.
EOF
}

case "${1:-}" in
  preflight) preflight ;;
  nlp-generate) nlp_generate ;;
  code-source-download) code_source_download ;;
  code-source-rebuild) code_source_rebuild ;;
  code-source-status) code_source_status ;;
  code-build) code_build ;;
  train-code) train_code promotion ;;
  train-code-pilot) train_code pilot ;;
  train-nlp) train_nlp ;;
  eval-code) evaluate_code ;;
  eval-code-rescue) evaluate_code_rescue ;;
  eval-code-pilot) evaluate_code_pilot ;;
  eval-nlp) evaluate_nlp ;;
  select-code) select_code ;;
  select-code-pilot) select_code_pilot ;;
  select-nlp) select_nlp ;;
  prepare-router) prepare_router ;;
  edge-start) edge_start ;;
  smoke96) smoke96 ;;
  edge-stop) edge_stop ;;
  checks) checks ;;
  *) usage; exit 2 ;;
esac
