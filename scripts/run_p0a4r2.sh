#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
P0A4R2_GPUS="${P0A4R2_GPUS:-0,1,2,3}"
P0A4R2_EVAL_GPU="${P0A4R2_EVAL_GPU:-0}"
CONFIG="configs/p0a4r2_v1_code.json"
TRAINER_CONFIG="configs/p0a4_distillation.json"
BASE_MODEL="models/checkpoints/p0a4/student-shared-merged"
BASE_AUDIT="reports/audit/gate_p0a4_student_shared_merge.json"
CODE_TRAIN="data/distill/p0a4r_code_train.jsonl"
CODE_VALIDATION="data/distill/p0a4r_code_internal_validation.jsonl"
CODE_DATA_AUDIT="reports/audit/gate_p0a4r_code_data.json"
CODE_OUTPUT="models/checkpoints/p0a4r2-v1/code"
CODE_SELECTED="models/checkpoints/p0a4r2-v1/code-selected"
CODE_TRAIN_AUDIT="reports/audit/gate_p0a4r2_v1_train_code.json"
PROTOCOL_AUDIT="reports/audit/gate_p0a4r2_v1_protocol.json"
SELECTION_AUDIT="reports/audit/gate_p0a4r2_v1_code_checkpoint_selection.json"

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
    raise SystemExit(
        f"Audit status is not allowed: {path} status={audit.get('status')} "
        f"allowed={sorted(allowed)}"
    )
print(f"Audit guard passed: {path} status={audit.get('status')}")
PY
}

config_training_args() {
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
from pathlib import Path
d=json.loads(Path("configs/p0a4r2_v1_code.json").read_text(encoding="utf-8"))
t=d["training"]["code_adapter"]
print(
    t["rank"],t["alpha"],t["dropout"],t["learning_rate"],t["epochs"],
    t["max_seq_length"],t["gradient_accumulation_steps"],t["source_balance_key"]
)
PY
}

freeze_nlp() {
  "$PYTHON_BIN" scripts/freeze_p0a4r_nlp.py --config "$CONFIG"
}

preflight() {
  require_file "$CONFIG"
  require_dir "$BASE_MODEL"
  require_audit_status "$BASE_AUDIT" passed
  require_file "$CODE_TRAIN"
  require_file "$CODE_VALIDATION"
  require_audit_status "$CODE_DATA_AUDIT" passed
  require_audit_status reports/audit/gate_p0a4r_nlp_frozen.json passed
  "$PYTHON_BIN" - "$CONFIG" "$CODE_DATA_AUDIT" "$BASE_AUDIT" "$PROTOCOL_AUDIT" <<'PY'
import hashlib,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

config_path,data_audit_path,base_audit_path,output_path=map(Path,sys.argv[1:])
config=json.loads(config_path.read_text(encoding="utf-8"))
data_audit=json.loads(data_audit_path.read_text(encoding="utf-8"))
base_audit=json.loads(base_audit_path.read_text(encoding="utf-8"))
policy=config["policy"]
if policy.get("feedback_source")!="train_only_internal_validation":
    raise SystemExit("P0-A4R2 feedback must be train-only")
for key in ("smoke96_item_feedback_used","selection170_feedback_used","formal_full_feedback_used"):
    if policy.get(key) is not False:
        raise SystemExit(f"Forbidden feedback enabled: {key}")
if data_audit.get("promotion_eligible") is not True:
    raise SystemExit("Code data is not eligible for promotion")
train_path=Path(config["data"]["code_train"])
validation_path=Path(config["data"]["code_internal_validation"])
if data_audit.get("train_hash")!=hashlib.sha256(train_path.read_bytes()).hexdigest():
    raise SystemExit("Code training data changed after its executable-data audit")
if data_audit.get("internal_validation_hash")!=hashlib.sha256(validation_path.read_bytes()).hexdigest():
    raise SystemExit("Code internal validation changed after its executable-data audit")
rows=[json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
settings=config["training"]["code_adapter"]
counts=Counter(str(row.get(settings["source_balance_key"],"")) for row in rows)
expected=set(settings["required_source_markers"].values())
if set(counts)!=expected:
    raise SystemExit(f"Unexpected MBPP/APPS source groups: {dict(counts)}")
weights={source:len(rows)/(len(counts)*count) for source,count in sorted(counts.items())}
weighted_mass={source:counts[source]*weights[source] for source in sorted(counts)}
if len({round(value,8) for value in weighted_mass.values()})!=1:
    raise SystemExit(f"Source loss masses are not balanced: {weighted_mass}")
if base_audit.get("output")!=config["models"]["student_base"]:
    raise SystemExit("v1 base audit points to a different merged model")
audit={
  "gate":"P0-A4R2-V1-PROTOCOL","check_version":"1.0","status":"passed",
  "created_by":"scripts/run_p0a4r2.sh:preflight",
  "created_ts":datetime.now(timezone.utc).isoformat(),
  "policy":policy,"base_model":config["models"]["student_base"],
  "base_model_audit":str(base_audit_path),
  "base_model_audit_sha256":hashlib.sha256(base_audit_path.read_bytes()).hexdigest(),
  "code_train":str(train_path),"code_train_sha256":hashlib.sha256(train_path.read_bytes()).hexdigest(),
  "code_internal_validation":str(validation_path),
  "code_internal_validation_sha256":hashlib.sha256(validation_path.read_bytes()).hexdigest(),
  "unique_train_rows":len(rows),"source_counts":dict(sorted(counts.items())),
  "source_loss_weights":weights,"weighted_mass_by_source":weighted_mass,
  "row_duplication":False,"rank":settings["rank"],"alpha":settings["alpha"],
  "learning_rate":settings["learning_rate"],"epochs":settings["epochs"],
  "routing":config["routing"],"formal_test_reference_count":0,
}
audit["report_hash"]=hashlib.sha256(
    json.dumps(audit,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
).hexdigest()
output_path.parent.mkdir(parents=True,exist_ok=True)
output_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(f"P0-A4R2 v1 preflight passed: {output_path}")
print(f"source_counts={dict(counts)} source_loss_weights={weights}")
PY
}

train_code_common() {
  local dry_run="$1"
  require_audit_status "$PROTOCOL_AUDIT" passed
  local args
  read -r -a args <<<"$(config_training_args)"
  local extra=()
  [[ "$dry_run" == "yes" ]] && extra+=(--dry-run)
  local training_args=(
    model_compression/train_p0a4_lora.py
    --config "$TRAINER_CONFIG" --role student_expert --candidate-index 1 --task humaneval \
    --model-dir "$BASE_MODEL" --train-data "$CODE_TRAIN" \
    --validation-data "$CODE_VALIDATION" --output-dir "$CODE_OUTPUT" \
    --audit "$CODE_TRAIN_AUDIT" \
    --rank "${args[0]}" --lora-alpha "${args[1]}" --lora-dropout "${args[2]}" \
    --learning-rate "${args[3]}" --epochs "${args[4]}" \
    --max-seq-length "${args[5]}" --gradient-accumulation-steps "${args[6]}" \
    --source-balance-key "${args[7]}" --external-checkpoint-selection "${extra[@]}"
  )
  if [[ "$dry_run" == "yes" ]]; then
    "$PYTHON_BIN" "${training_args[@]}"
  else
    CUDA_VISIBLE_DEVICES="$P0A4R2_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      "${training_args[@]}"
  fi
}

train_code_dry_run() {
  train_code_common yes
}

train_code() {
  [[ ! -e "$CODE_OUTPUT" ]] || {
    echo "Refusing to overwrite existing P0-A4R2 training output: $CODE_OUTPUT" >&2
    return 2
  }
  train_code_common no
}

evaluate_one() {
  local name="$1" trace="$2" audit="$3"
  shift 3
  CUDA_VISIBLE_DEVICES="$P0A4R2_EVAL_GPU" "$PYTHON_BIN" scripts/evaluate_edge_candidate_dev.py \
    --local-model-dir "$BASE_MODEL" --candidate-name "$name" \
    --validation-data "$CODE_VALIDATION" --output-trace "$trace" --audit "$audit" \
    --device cuda:0 --dtype bfloat16 --disable-thinking \
    --max-new-tokens-map "humaneval=512" --min-accuracy-map "humaneval=0" "$@"
}

eval_code() {
  require_audit_status "$CODE_TRAIN_AUDIT" passed
  mkdir -p data/eval/p0a4r2 reports/audit/p0a4r2
  evaluate_one p0a4r2-v1-code-base \
    data/eval/p0a4r2/code_base.jsonl reports/audit/p0a4r2/code_base.json
  mapfile -t checkpoints < <("$PYTHON_BIN" - "$CODE_TRAIN_AUDIT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for value in d.get("checkpoint_candidates",[]): print(value)
PY
)
  [[ "${#checkpoints[@]}" -eq 1 ]] || {
    echo "One-epoch P0-A4R2 training must produce exactly one checkpoint." >&2
    return 1
  }
  local checkpoint="${checkpoints[0]}" step="${checkpoints[0]##*-}"
  evaluate_one "p0a4r2-v1-code-checkpoint-${step}" \
    "data/eval/p0a4r2/code_checkpoint_${step}.jsonl" \
    "reports/audit/p0a4r2/code_checkpoint_${step}.json" \
    --adapter-dir "$checkpoint"
}

select_code() {
  require_audit_status "$CODE_TRAIN_AUDIT" passed
  local baseline="reports/audit/p0a4r2/code_base.json"
  mapfile -t audits < <(find reports/audit/p0a4r2 -maxdepth 1 -type f \
    -name 'code_checkpoint_*.json' -print | sort -V)
  [[ "${#audits[@]}" -eq 1 ]] || {
    echo "Expected exactly one P0-A4R2 internal candidate audit." >&2
    return 1
  }
  "$PYTHON_BIN" scripts/select_p0a4r_checkpoint.py \
    --config "$CONFIG" --task humaneval --baseline-audit "$baseline" \
    --candidate-audit "${audits[0]}" --output-dir "$CODE_SELECTED" \
    --audit "$SELECTION_AUDIT"
}

checks() {
  bash -n scripts/run_p0a4r2.sh
  "$PYTHON_BIN" -m py_compile \
    model_compression/train_p0a4_lora.py \
    scripts/freeze_p0a4r_nlp.py
  "$PYTHON_BIN" -m unittest \
    tests.test_p0a4_pipeline \
    tests.test_p0a4r_remediation \
    tests.test_p0a4r2_v1_code
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p0a4r2.sh <command>

  freeze-nlp       Freeze the successful v2 NLP Adapter lineage by hashes; archive only.
  preflight        Verify v1 base, data isolation, Rank-4 settings and MBPP/APPS loss balance.
  train-code-dry   Validate the complete v1 Code training plan without loading the model.
  train-code       Train one Rank-4 Code Adapter epoch on four GPUs; never overwrites output.
  eval-code        Evaluate v1 base and the sole checkpoint on train-only internal MBPP.
  select-code      Publish only an internally improving checkpoint.
  checks           Run shell, Python and unit checks.

NLP and Math use the shared v1 model. The frozen v2 NLP Adapter is not compatible with v1
and remains archive-only. No command reads smoke96 item output, selection170 item output,
or official-full traces.
EOF
}

case "${1:-}" in
  freeze-nlp) freeze_nlp ;;
  preflight) preflight ;;
  train-code-dry) train_code_dry_run ;;
  train-code) train_code ;;
  eval-code) eval_code ;;
  select-code) select_code ;;
  checks) checks ;;
  *) usage; exit 2 ;;
esac
