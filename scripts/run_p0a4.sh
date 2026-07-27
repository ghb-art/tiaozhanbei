#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT/.venv/bin/torchrun}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/external/llama.cpp}"
P0A4_GPUS="${P0A4_GPUS:-0,1,2,3}"
P0A4_BASELINE_URL="${P0A4_BASELINE_URL:-http://127.0.0.1:8000}"
P0A4_TEACHER_URL="${P0A4_TEACHER_URL:-http://127.0.0.1:8000}"
P0A4_EDGE_URL="${P0A4_EDGE_URL:-http://127.0.0.1:18441}"
P0A4_EDGE_PORT="${P0A4_EDGE_PORT:-18441}"
P0A4_EDGE_GPUS="${P0A4_EDGE_GPUS:-0}"
P0A4_EDGE_PARALLEL="${P0A4_EDGE_PARALLEL:-4}"
P0A4_EDGE_CTX_PER_SLOT="${P0A4_EDGE_CTX_PER_SLOT:-1536}"
P0A4_TEACHER_VERSION="${P0A4_TEACHER_VERSION:-1}"
P0A4_STUDENT_VERSION="${P0A4_STUDENT_VERSION:-1}"
P0A4_ADAPTER_ROUTER_MANIFEST="${P0A4_ADAPTER_ROUTER_MANIFEST:-}"
P0A4_ADAPTER_PREPARE_AUDIT="${P0A4_ADAPTER_PREPARE_AUDIT:-}"
P0A4_ADAPTER_ROUTE_TAG="${P0A4_ADAPTER_ROUTE_TAG:-adapter}"

CONFIG="configs/p0a4_distillation.json"
BASELINE_MODEL="models/pretrained/Qwen--Qwen2.5-14B-Instruct-AWQ"
TEACHER_MODEL="models/pretrained/Qwen--Qwen2.5-14B-Instruct"
STUDENT_MODEL="models/pretrained/Qwen--Qwen3-1.7B"
TRAIN_DATA="data/distill/p0a4_train.jsonl"
TEACHER_VALIDATION="data/distill/p0a4_teacher_validation.jsonl"
SMOKE96="data/distill/p0a4_smoke96.jsonl"
SELECTION170="data/distill/p0a2_recovery_validation.jsonl"
DISTILL_DATA="data/distill/p0a4_teacher_verified.jsonl"
FULL_SPLITS="data/splits/p0a4_official_full"
BASELINE_FULL_TRACE="reports/sealed/p0a4/baseline14b_awq_full.jsonl"
BASELINE_FULL_AUDIT="reports/audit/gate_p0a4_baseline14b_awq_full.json"
STUDENT_FULL_TRACE="reports/sealed/p0a4/edge_student_full.jsonl"
STUDENT_FULL_AUDIT="reports/audit/gate_p0a4_edge_student_full.json"
EDGE_QUANT_TYPE="Q4_K_M"
EDGE_QUANT_TAG="q4_k_m"
if [[ "$P0A4_STUDENT_VERSION" == "1" ]]; then
  STUDENT_MERGED_DIR="models/checkpoints/p0a4/student-shared-merged"
  STUDENT_MERGE_AUDIT="reports/audit/gate_p0a4_student_shared_merge.json"
  EDGE_GGUF="models/quantized/p0a4-qwen3-1.7b-q4_k_m.gguf"
  EDGE_F16_GGUF="models/quantized/p0a4-qwen3-1.7b-f16.gguf"
  EDGE_IMATRIX="models/quantized/p0a4-qwen3-1.7b-q3_k_m.imatrix"
  EDGE_QUANT_AUDIT="reports/audit/gate_p0a4_student_q4_prepare.json"
  EDGE_MEMORY_PRECHECK_AUDIT="reports/audit/gate_p0a4_edge_student_q4_k_m_memory_precheck_cache_off.json"
  IMATRIX_CALIBRATION="data/distill/p0a4_imatrix_calibration.txt"
  IMATRIX_CALIBRATION_AUDIT="reports/audit/gate_p0a4_imatrix_calibration.json"
  ADAPTER_DIR="models/adapters/p0a4/v1"
  ADAPTER_MANIFEST="models/adapters/p0a4/v1/router_manifest.json"
  ADAPTER_PREPARE_AUDIT="reports/audit/gate_p0a4_adapter_router_prepare_v1.json"
else
  STUDENT_MERGED_DIR="models/checkpoints/p0a4/student-shared-v${P0A4_STUDENT_VERSION}-merged"
  STUDENT_MERGE_AUDIT="reports/audit/gate_p0a4_student_shared_v${P0A4_STUDENT_VERSION}_merge.json"
  EDGE_GGUF="models/quantized/p0a4-qwen3-1.7b-v${P0A4_STUDENT_VERSION}-q4_k_m.gguf"
  EDGE_F16_GGUF="models/quantized/p0a4-qwen3-1.7b-v${P0A4_STUDENT_VERSION}-f16.gguf"
  EDGE_IMATRIX="models/quantized/p0a4-qwen3-1.7b-v${P0A4_STUDENT_VERSION}.imatrix"
  EDGE_QUANT_AUDIT="reports/audit/gate_p0a4_student_v${P0A4_STUDENT_VERSION}_q4_prepare.json"
  EDGE_MEMORY_PRECHECK_AUDIT="reports/audit/gate_p0a4_edge_student_v${P0A4_STUDENT_VERSION}_q4_k_m_memory_precheck_cache_off.json"
  IMATRIX_CALIBRATION="data/distill/p0a4_imatrix_calibration_v${P0A4_STUDENT_VERSION}.txt"
  IMATRIX_CALIBRATION_AUDIT="reports/audit/gate_p0a4_imatrix_calibration_v${P0A4_STUDENT_VERSION}.json"
  ADAPTER_DIR="models/adapters/p0a4/v${P0A4_STUDENT_VERSION}"
  ADAPTER_MANIFEST="models/adapters/p0a4/v${P0A4_STUDENT_VERSION}/router_manifest.json"
  ADAPTER_PREPARE_AUDIT="reports/audit/gate_p0a4_adapter_router_prepare_v${P0A4_STUDENT_VERSION}.json"
fi
if [[ -n "$P0A4_ADAPTER_ROUTER_MANIFEST" ]]; then
  ADAPTER_MANIFEST="$P0A4_ADAPTER_ROUTER_MANIFEST"
  ADAPTER_DIR="$(dirname "$ADAPTER_MANIFEST")"
fi
if [[ -n "$P0A4_ADAPTER_PREPARE_AUDIT" ]]; then
  ADAPTER_PREPARE_AUDIT="$P0A4_ADAPTER_PREPARE_AUDIT"
fi
TOKEN_LIMITS="gsm8k=512,humaneval=512,cmmlu=16"

build_llama_cuda() {
  local cmake_bin="${CMAKE_BIN:-$ROOT/.venv/bin/cmake}"
  local ninja_bin="${NINJA_BIN:-$ROOT/.venv/bin/ninja}"
  require_file "$cmake_bin"
  require_file "$ninja_bin"
  require_file "$LLAMA_CPP_DIR/CMakeLists.txt"
  "$cmake_bin" -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$ninja_bin" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
  "$cmake_bin" --build "$LLAMA_CPP_DIR/build" --config Release \
    --target llama-cli llama-server llama-quantize llama-imatrix --parallel 8
  grep -q '^GGML_CUDA:BOOL=ON$' "$LLAMA_CPP_DIR/build/CMakeCache.txt" || {
    echo "CUDA llama.cpp build verification failed" >&2
    return 1
  }
  echo "CUDA llama.cpp build passed: compute capability 8.6"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

require_active_student_version() {
  local requested="$1"
  if [[ "$requested" != "$P0A4_STUDENT_VERSION" ]]; then
    echo "Student artifact version mismatch: requested v${requested}, active v${P0A4_STUDENT_VERSION}." >&2
    echo "Run with P0A4_STUDENT_VERSION=${requested}." >&2
    return 2
  fi
}

require_audit_passed() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"Missing audit: {p}")
d = json.loads(p.read_text(encoding="utf-8"))
if d.get("status") != "passed":
    raise SystemExit(f"Gate not passed: {p} status={d.get('status')}")
print(f"Audit guard passed: {p}")
PY
}

require_distill_ready() {
  "$PYTHON_BIN" - "$CONFIG" "$DISTILL_DATA" reports/audit/gate_p0a4_teacher_distill.json <<'PY'
import hashlib,json,sys
from pathlib import Path
config_path,data_path,audit_path=map(Path,sys.argv[1:])
if not data_path.is_file() or not audit_path.is_file():
    raise SystemExit("Missing verified distillation data or audit")
config=json.loads(config_path.read_text(encoding="utf-8"))
audit=json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("status") != "passed":
    raise SystemExit(f"Teacher distillation gate not passed: {audit.get('status')}")
digest=hashlib.sha256(data_path.read_bytes()).hexdigest()
if digest != audit.get("distill_hash"):
    raise SystemExit("Verified distillation data hash does not match its audit")
required=config["gates"]["teacher_distill"]["min_accepted_unique_by_task"]
actual=audit.get("accepted_unique_prompt_counts",{})
failures={task:{"actual":int(actual.get(task,0)),"required":int(count)} for task,count in required.items() if int(actual.get(task,0)) < int(count)}
if failures:
    raise SystemExit(f"Distillation task coverage is insufficient: {failures}")
print(f"Distillation guard passed: {data_path} counts={actual}")
PY
}

require_student_training_ready() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  "$PYTHON_BIN" - "$version" "$DISTILL_DATA" \
    "reports/audit/gate_p0a4_train_student_shared_v${version}.json" \
    "models/checkpoints/p0a4/student-shared-v${version}" <<'PY'
import hashlib,json,sys
from pathlib import Path
version=int(sys.argv[1])
data_path=Path(sys.argv[2]); audit_path=Path(sys.argv[3]); adapter=Path(sys.argv[4])
if not data_path.is_file() or not audit_path.is_file() or not adapter.is_dir():
    raise SystemExit("Missing Student training data, audit, or adapter directory")
audit=json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("status") != "passed" or audit.get("role") != "student_shared":
    raise SystemExit(f"Student training gate not passed: status={audit.get('status')} role={audit.get('role')}")
if int(audit.get("candidate_index",0)) != version:
    raise SystemExit(f"Student version mismatch: requested v{version}, audit={audit.get('candidate_index')}")
data_hash=hashlib.sha256(data_path.read_bytes()).hexdigest()
if data_hash != audit.get("train_data_hash"):
    raise SystemExit("Student training data hash does not match the training audit")
best=Path(str(audit.get("best_checkpoint","")))
if not best.is_dir():
    raise SystemExit(f"Missing selected Student checkpoint: {best}")
required=("adapter_config.json","adapter_model.safetensors")
for name in required:
    published=adapter/name; selected=best/name
    if not published.is_file() or not selected.is_file():
        raise SystemExit(f"Missing published or selected adapter file: {name}")
    if hashlib.sha256(published.read_bytes()).hexdigest() != hashlib.sha256(selected.read_bytes()).hexdigest():
        raise SystemExit(f"Published adapter does not match selected checkpoint: {name}")
if not set(required).issubset(set(audit.get("published_adapter_files",[]))):
    raise SystemExit("Training audit does not record the required published adapter files")
print(f"Student training guard passed: v{version} best={best}")
PY
}

require_student_merge_ready() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_student_training_ready "$version"
  "$PYTHON_BIN" - "$version" "$STUDENT_MERGE_AUDIT" "$STUDENT_MERGED_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
version=int(sys.argv[1])
audit_path=Path(sys.argv[2])
train_path=Path(f"reports/audit/gate_p0a4_train_student_shared_v{version}.json")
adapter=Path(f"models/checkpoints/p0a4/student-shared-v{version}")
merged=Path(sys.argv[3])
if not audit_path.is_file() or not merged.is_dir():
    raise SystemExit("Missing merged Student or merge audit")
audit=json.loads(audit_path.read_text(encoding="utf-8")); train=json.loads(train_path.read_text(encoding="utf-8"))
if audit.get("status") != "passed":
    raise SystemExit(f"Student merge gate not passed: {audit.get('status')}")
if Path(str(audit.get("adapter",""))) != adapter or Path(str(audit.get("output",""))) != merged:
    raise SystemExit("Merge audit points to a different Student adapter or output")
if str(audit.get("created_ts","")) <= str(train.get("created_ts","")):
    raise SystemExit("Merged Student predates the selected training result")
def file_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
def hash_dir(path):
    digest=hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(file_hash(item).encode())
    return digest.hexdigest()
if hash_dir(adapter) != audit.get("adapter_hash"):
    raise SystemExit("Current Student adapter hash does not match merge audit")
if hash_dir(merged) != audit.get("output_hash"):
    raise SystemExit("Current merged Student hash does not match merge audit")
if not (merged/"config.json").is_file() or not any(merged.glob("model*.safetensors")):
    raise SystemExit("Merged Student is incomplete")
print(f"Student merge guard passed: v{version} output={merged}")
PY
}

require_quantized_student_ready() {
  "$PYTHON_BIN" - "$EDGE_GGUF" "$EDGE_QUANT_AUDIT" "$EDGE_QUANT_TYPE" \
    "$STUDENT_MERGE_AUDIT" "$STUDENT_MERGED_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
def file_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
gguf=Path(sys.argv[1]); audit_path=Path(sys.argv[2]); expected_type=sys.argv[3]
merge_path=Path(sys.argv[4]); expected_merged=Path(sys.argv[5])
if not gguf.is_file() or not audit_path.is_file() or not merge_path.is_file():
    raise SystemExit("Missing edge Student, quantization audit, or merge audit")
audit=json.loads(audit_path.read_text(encoding="utf-8")); merge=json.loads(merge_path.read_text(encoding="utf-8"))
if audit.get("status") != "passed" or audit.get("quant_type") != expected_type:
    raise SystemExit(f"Edge preparation gate not passed: status={audit.get('status')} type={audit.get('quant_type')}")
if Path(str(audit.get("quantized_gguf",""))) != gguf:
    raise SystemExit("Edge quantization audit points to a different GGUF")
if file_hash(gguf) != audit.get("quantized_gguf_hash"):
    raise SystemExit("Edge GGUF hash does not match quantization audit")
merged=Path(str(audit.get("merged_hf_dir","")))
if not merged.is_dir() or merged != expected_merged or Path(str(merge.get("output",""))) != merged:
    raise SystemExit("Quantization and merge audits point to different merged models")
prepare_digest=hashlib.sha256(); merge_digest=hashlib.sha256()
for item in sorted(p for p in merged.rglob("*") if p.is_file()):
    relative=item.relative_to(merged).as_posix().encode()
    prepare_digest.update(relative); merge_digest.update(relative)
    item_digest=hashlib.sha256()
    with item.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            prepare_digest.update(chunk); item_digest.update(chunk)
    merge_digest.update(item_digest.hexdigest().encode())
if prepare_digest.hexdigest() != audit.get("merged_hf_hash"):
    raise SystemExit("Current merged model does not match the quantization audit")
if merge_digest.hexdigest() != merge.get("output_hash"):
    raise SystemExit("Current merged model does not match the merge audit")
imatrix=Path(str(audit.get("imatrix","")))
if not imatrix.is_file() or file_hash(imatrix) != audit.get("imatrix_hash"):
    raise SystemExit("Importance matrix is missing or does not match quantization audit")
if not audit.get("artifact_size_pre_gate_passed"):
    raise SystemExit("Edge artifact size pre-gate did not pass")
print(f"Quantized Student guard passed: type={expected_type} {gguf} bytes={gguf.stat().st_size}")
PY
}

require_student_eval_runtime() {
  local audit_path="$1"
  "$PYTHON_BIN" - "$audit_path" "$EDGE_GGUF" "$EDGE_QUANT_AUDIT" <<'PY'
import json,sys
from pathlib import Path
audit_path=Path(sys.argv[1]); gguf=Path(sys.argv[2]); quant_path=Path(sys.argv[3])
if not audit_path.is_file() or not gguf.is_file() or not quant_path.is_file():
    raise SystemExit("Missing Student evaluation, edge model, or quantization audit")
audit=json.loads(audit_path.read_text(encoding="utf-8")); quant=json.loads(quant_path.read_text(encoding="utf-8"))
if audit.get("status") != "passed":
    raise SystemExit(f"Student evaluation did not complete: {audit_path} status={audit.get('status')}")
if audit.get("disable_thinking") is not True:
    raise SystemExit(f"Student evaluation did not explicitly disable thinking: {audit_path}")
if audit.get("kv_cache_type") != "q8_0":
    raise SystemExit(f"Student evaluation did not audit Q8 KV cache: {audit_path}")
model=audit.get("model")
if isinstance(model,dict):
    if model.get("sha256") != quant.get("quantized_gguf_hash"):
        raise SystemExit(f"Student evaluation model hash does not match edge audit: {audit_path}")
else:
    if Path(str(audit.get("local_model_dir",""))) != gguf:
        raise SystemExit(f"Student shard audit points to a different local model: {audit_path}")
print(f"Student runtime audit passed: {audit_path} thinking=off kv=q8_0")
PY
}

require_edge_memory_precheck_ready() {
  "$PYTHON_BIN" - "$EDGE_GGUF" "$EDGE_QUANT_AUDIT" \
    "$EDGE_MEMORY_PRECHECK_AUDIT" <<'PY'
import json,sys
from pathlib import Path
gguf,quant_path,memory_path=map(Path,sys.argv[1:])
if not gguf.is_file() or not quant_path.is_file() or not memory_path.is_file():
    raise SystemExit("Missing edge GGUF, quantization audit, or memory precheck")
quant=json.loads(quant_path.read_text(encoding="utf-8"))
memory=json.loads(memory_path.read_text(encoding="utf-8"))
if memory.get("status") != "passed":
    raise SystemExit(f"Edge memory precheck did not pass: {memory.get('status')}")
if Path(str(memory.get("gguf_path",""))) != gguf:
    raise SystemExit("Memory precheck points to a different edge GGUF")
if memory.get("gguf_hash") != quant.get("quantized_gguf_hash"):
    raise SystemExit("Memory precheck model hash does not match quantization audit")
if memory.get("host_prompt_cache_mib") != 0 or memory.get("cache_idle_slots") is not False:
    raise SystemExit("Memory precheck did not disable the host prompt cache")
if int(memory.get("successful_warmup_requests",0)) != 5 or int(memory.get("measure_requests",0)) != 20:
    raise SystemExit("Memory precheck did not complete the required 5+20 requests")
peak=float(memory.get("peak_total_memory_mb_decimal",float("inf")))
if peak > 1400:
    raise SystemExit(f"Memory precheck exceeds 1400 MB: {peak}")
print(f"Edge memory precheck guard passed: peak={peak:.2f} MB cache=off")
PY
}

require_adapter_router_ready() {
  local manifest="${P0A4_ADAPTER_ROUTER_MANIFEST:-$ADAPTER_MANIFEST}"
  "$PYTHON_BIN" - "$manifest" "$ADAPTER_PREPARE_AUDIT" "$STUDENT_MERGE_AUDIT" \
    "$STUDENT_MERGED_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest_path,audit_path,merge_path,merged=map(Path,sys.argv[1:])
if not all(path.is_file() for path in (manifest_path,audit_path,merge_path)) or not merged.is_dir():
    raise SystemExit("Missing Adapter manifest, preparation audit, merge audit, or merged Student")
def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
audit=json.loads(audit_path.read_text(encoding="utf-8"))
merge=json.loads(merge_path.read_text(encoding="utf-8"))
if audit.get("status") != "passed" or audit.get("manifest_hash") != sha256(manifest_path):
    raise SystemExit("Adapter preparation audit did not pass or manifest hash changed")
if Path(str(manifest.get("base_model",""))) != merged or Path(str(merge.get("output",""))) != merged:
    raise SystemExit("Adapter manifest and merge audit point to different base models")
if manifest.get("base_model_hash") != merge.get("output_hash"):
    raise SystemExit("Adapter manifest base hash does not match merged Student")
adapters=manifest.get("task_adapters",{})
if set(adapters) != {"gsm8k","humaneval","cmmlu"}:
    raise SystemExit("Adapter manifest must contain all three task routes")
for task,value in adapters.items():
    path=Path(str(value.get("path","")))
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise SystemExit(f"Adapter artifact is missing or changed: {task}")
print(f"Adapter router guard passed: {manifest_path}")
PY
}

require_adapter_memory_precheck_ready() {
  local report="reports/audit/gate_p0a4_edge_student_v${P0A4_STUDENT_VERSION}_adapter_memory_precheck.json"
  "$PYTHON_BIN" - "$report" "$EDGE_GGUF" "$EDGE_QUANT_AUDIT" <<'PY'
import json,sys
from pathlib import Path
report_path,gguf,quant_path=map(Path,sys.argv[1:])
if not report_path.is_file() or not gguf.is_file() or not quant_path.is_file():
    raise SystemExit("Missing Adapter memory precheck evidence")
report=json.loads(report_path.read_text(encoding="utf-8"))
quant=json.loads(quant_path.read_text(encoding="utf-8"))
if report.get("status") != "passed" or Path(str(report.get("gguf_path",""))) != gguf:
    raise SystemExit("Adapter memory precheck did not pass for the active GGUF")
if report.get("gguf_hash") != quant.get("quantized_gguf_hash"):
    raise SystemExit("Adapter memory precheck model hash mismatch")
if report.get("host_prompt_cache_mib") != 0 or report.get("cache_idle_slots") is not False:
    raise SystemExit("Adapter memory precheck did not disable host prompt cache")
if int(report.get("successful_warmup_requests",0)) != 5 or int(report.get("measure_requests",0)) != 20:
    raise SystemExit("Adapter memory precheck did not complete 5+20 requests")
peak=float(report.get("peak_total_memory_mb_decimal",float("inf")))
if peak > 1400:
    raise SystemExit(f"Adapter memory precheck exceeds 1400 MB: {peak}")
print(f"Adapter memory precheck guard passed: peak={peak:.2f} MB")
PY
}

require_v2_route_ready() {
  "$PYTHON_BIN" - reports/audit/gate_p0a4_student_v2_route_selection.json \
    "$EDGE_QUANT_AUDIT" "$ADAPTER_MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
route_path,quant_path,manifest_path=map(Path,sys.argv[1:])
if not all(path.is_file() for path in (route_path,quant_path,manifest_path)):
    raise SystemExit("Missing Student v2 route, quantization, or Adapter evidence")
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
route=json.loads(route_path.read_text(encoding="utf-8")); quant=json.loads(quant_path.read_text(encoding="utf-8"))
if route.get("status") != "passed" or route.get("full_test_feedback_used") is not False:
    raise SystemExit("Student v2 route selection did not pass the aggregate-only policy")
if route.get("quantization_audit_hash") != sha256(quant_path) or route.get("quantized_gguf_hash") != quant.get("quantized_gguf_hash"):
    raise SystemExit("Student v2 route does not bind the active quantized model")
if route.get("adapter_manifest_hash") != sha256(manifest_path):
    raise SystemExit("Student v2 route does not bind the active Adapter manifest")
selected=route.get("selected_route")
if selected not in {"shared","adapter_top1"}:
    raise SystemExit(f"Invalid Student v2 selected route: {selected}")
print(selected)
PY
}

require_edge_smoke96_ready() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  local trace="data/eval/p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_smoke96.jsonl"
  local eval_audit="reports/audit/gate_p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_smoke96.json"
  local retention_audit="reports/audit/gate_p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_smoke96_retention.json"
  "$PYTHON_BIN" - "$CONFIG" "$EDGE_GGUF" "$EDGE_QUANT_AUDIT" \
    "$trace" "$eval_audit" "$retention_audit" \
    data/eval/p0a4_baseline14b_awq_smoke96.jsonl \
    reports/audit/gate_p0a4_baseline14b_awq_smoke96.json <<'PY'
import hashlib,json,sys
from pathlib import Path

config_path,gguf,quant_path,trace,eval_path,retention_path,baseline_trace,baseline_path=map(Path,sys.argv[1:])
paths=(config_path,gguf,quant_path,trace,eval_path,retention_path,baseline_trace,baseline_path)
if not all(path.is_file() for path in paths):
    missing=[str(path) for path in paths if not path.is_file()]
    raise SystemExit(f"Missing smoke96 evidence: {missing}")

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

config=json.loads(config_path.read_text(encoding="utf-8"))
quant=json.loads(quant_path.read_text(encoding="utf-8"))
evaluation=json.loads(eval_path.read_text(encoding="utf-8"))
retention=json.loads(retention_path.read_text(encoding="utf-8"))
baseline=json.loads(baseline_path.read_text(encoding="utf-8"))
trace_hash=sha256(trace); baseline_hash=sha256(baseline_trace)

if evaluation.get("status") != "passed" or retention.get("status") != "passed":
    raise SystemExit("Q4 smoke96 evaluation or retention gate did not pass")
if evaluation.get("disable_thinking") is not True or evaluation.get("kv_cache_type") != "q8_0":
    raise SystemExit("Q4 smoke96 runtime is not thinking=off with Q8 KV")
if evaluation.get("model",{}).get("sha256") != quant.get("quantized_gguf_hash"):
    raise SystemExit("Q4 smoke96 model hash does not match quantization audit")
if Path(str(evaluation.get("model",{}).get("path",""))) != gguf:
    raise SystemExit("Q4 smoke96 evaluation points to a different GGUF")
if Path(str(evaluation.get("output_trace",""))) != trace or evaluation.get("output_trace_sha256") != trace_hash:
    raise SystemExit("Q4 smoke96 trace does not match its evaluation audit")
if retention.get("candidate_trace_hash") != trace_hash:
    raise SystemExit("Q4 smoke96 retention report does not bind the evaluated trace")
if baseline.get("output_trace_sha256") != baseline_hash or retention.get("baseline_trace_hash") != baseline_hash:
    raise SystemExit("Q4 smoke96 retention report does not bind the frozen baseline trace")

gate=config["gates"]["smoke96"]
expected={key:int(value) for key,value in gate["expected_counts"].items()}
if evaluation.get("dataset_counts") != expected or retention.get("candidate_counts") != expected:
    raise SystemExit("Q4 smoke96 evidence does not contain the required 32+32+32 rows")
minimum=float(gate["min_ratio_per_task"]); macro_min=float(gate["min_capped_macro_ratio"])
ratios=retention.get("ratios",{})
if any(float(ratios.get(key,0)) < minimum for key in ("math_ratio","code_ratio","nlp_ratio")):
    raise SystemExit(f"Q4 smoke96 per-task retention is below {minimum}: {ratios}")
if float(retention.get("capped_macro_ratio",0)) < macro_min or int(retention.get("generation_error_count",-1)) != 0:
    raise SystemExit("Q4 smoke96 macro retention or generation-error gate failed")
print(f"Edge smoke96 guard passed: ratios={ratios} macro={retention['capped_macro_ratio']:.6f}")
PY
}

require_baseline_selection170_ready() {
  "$PYTHON_BIN" - "$SELECTION170" data/eval/p0a4_baseline14b_awq_selection170.jsonl \
    reports/audit/gate_p0a4_baseline14b_awq_selection170.json <<'PY'
import hashlib,json,sys
from pathlib import Path
validation,trace,audit_path=map(Path,sys.argv[1:])
if not validation.is_file() or not trace.is_file() or not audit_path.is_file():
    raise SystemExit("Missing selection170 data, frozen baseline trace, or baseline audit")
def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
audit=json.loads(audit_path.read_text(encoding="utf-8"))
expected={"cmmlu":64,"gsm8k":64,"humaneval":42}
if audit.get("status") != "passed" or audit.get("dataset_counts") != expected or int(audit.get("sample_count",0)) != 170:
    raise SystemExit("Frozen baseline selection170 audit is incomplete")
if int(audit.get("generation_error_count",-1)) != 0:
    raise SystemExit("Frozen baseline selection170 contains generation errors")
if Path(str(audit.get("output_trace",""))) != trace or audit.get("output_trace_sha256") != sha256(trace):
    raise SystemExit("Frozen baseline selection170 trace hash does not match its audit")
if Path(str(audit.get("validation_data",""))) != validation or audit.get("validation_data_sha256") != sha256(validation):
    raise SystemExit("Frozen selection170 input hash does not match its baseline audit")
print(f"Baseline selection170 guard passed: counts={expected}")
PY
}

require_edge_service_ready() {
  "$PYTHON_BIN" - "$P0A4_EDGE_URL" "$EDGE_GGUF" "$EDGE_QUANT_AUDIT" \
    reports/audit/gate_p0a4_edge_service.json runtime/p0a4_edge.pid \
    "p0a4-edge-student-v${P0A4_STUDENT_VERSION}-${EDGE_QUANT_TAG//_/-}" "$P0A4_EDGE_PARALLEL" \
    "$((P0A4_EDGE_CTX_PER_SLOT * P0A4_EDGE_PARALLEL))" \
    "$P0A4_ADAPTER_ROUTER_MANIFEST" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
from urllib.request import Request,urlopen
url=sys.argv[1].rstrip('/')
gguf=Path(sys.argv[2]); quant_path=Path(sys.argv[3]); service_path=Path(sys.argv[4]); pid_path=Path(sys.argv[5])
expected_alias=sys.argv[6]; expected_parallel=int(sys.argv[7]); expected_ctx=int(sys.argv[8]); manifest_value=sys.argv[9]
if not all(path.is_file() for path in (gguf,quant_path,service_path,pid_path)):
    raise SystemExit("Managed Q4 edge service is not running; run edge-start first")
quant=json.loads(quant_path.read_text(encoding="utf-8")); service=json.loads(service_path.read_text(encoding="utf-8"))
pid=int(pid_path.read_text(encoding="utf-8").strip())
try:
    os.kill(pid,0)
except OSError as exc:
    raise SystemExit(f"Managed edge service pid {pid} is not alive: {exc}")
cmdline=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
required=(str(gguf),"--cache-type-k q8_0","--cache-type-v q8_0","--cache-ram 0","--no-cache-idle-slots",f"--parallel {expected_parallel}",f"--ctx-size {expected_ctx}")
missing=[item for item in required if item not in cmdline]
if missing:
    raise SystemExit(f"Managed edge service command is missing fixed runtime arguments: {missing}")
if service.get("status") != "passed" or int(service.get("pid",-1)) != pid or service.get("endpoint") != url:
    raise SystemExit("Edge service audit does not match the live managed process")
if service.get("model_hash") != quant.get("quantized_gguf_hash") or service.get("alias") != expected_alias:
    raise SystemExit("Edge service audit does not bind the promoted Q4 model")
if service.get("gpu_backend") is not True or int(service.get("parallel",0)) != expected_parallel or int(service.get("ctx_size",0)) != expected_ctx:
    raise SystemExit("Edge service is not the required CUDA/parallel runtime")
expected_manifest_hash=""
if manifest_value:
    manifest_path=Path(manifest_value)
    if not manifest_path.is_file(): raise SystemExit("Configured Adapter manifest is missing")
    expected_manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if service.get("adapter_manifest_hash","") != expected_manifest_hash:
    raise SystemExit("Edge service Adapter route does not match the current request")
try:
    with urlopen(Request(url+'/health'),timeout=3) as response:
        if response.status != 200: raise RuntimeError(f"health={response.status}")
    with urlopen(Request(url+'/v1/models'),timeout=3) as response:
        models=json.load(response)
except Exception as exc:
    raise SystemExit(f"Managed edge endpoint is unavailable: {exc}")
ids={str(item.get('id','')) for item in models.get('data',[])}
if expected_alias not in ids:
    raise SystemExit(f"Edge endpoint model identity mismatch: expected={expected_alias} actual={sorted(ids)}")
print(f"Managed CUDA edge service guard passed: pid={pid} model={expected_alias} parallel={expected_parallel}")
PY
}

preflight() {
  require_file "$PYTHON_BIN"
  "$PYTHON_BIN" scripts/p0a4_protocol.py preflight --config "$CONFIG"
  "$PYTHON_BIN" model_compression/train_p0a4_lora.py \
    --config "$CONFIG" --role teacher --candidate-index 1 \
    --output-dir models/checkpoints/p0a4/teacher-preflight \
    --audit reports/audit/gate_p0a4_train_teacher_preflight.json \
    --deepspeed configs/deepspeed_p0a4_zero3.json --dry-run
}

install_training_deps() {
  "$PYTHON_BIN" -m pip install -r requirements-p0a4.txt
}

download_teacher() {
  "$PYTHON_BIN" scripts/download_models.py --model distill_teacher_base
}

baseline_plan() {
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$P0A4_GPUS" --tensor-parallel-size auto \
    --model-dir "$BASELINE_MODEL" --quantization awq --port 8000 --dry-run
}

baseline_serve() {
  require_dir "$BASELINE_MODEL"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$P0A4_GPUS" --tensor-parallel-size auto \
    --model-dir "$BASELINE_MODEL" --quantization awq --port 8000
}

teacher_plan() {
  local adapter="models/checkpoints/p0a4/teacher-v${P0A4_TEACHER_VERSION}"
  require_dir "$adapter"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$P0A4_GPUS" --tensor-parallel-size auto \
    --model-dir "$TEACHER_MODEL" --quantization none --port 8000 \
    --lora-module "distill-teacher-v${P0A4_TEACHER_VERSION}=$ROOT/$adapter" --dry-run
}

teacher_serve() {
  local adapter="models/checkpoints/p0a4/teacher-v${P0A4_TEACHER_VERSION}"
  require_dir "$TEACHER_MODEL"
  require_dir "$adapter"
  "$PYTHON_BIN" scripts/serve_vllm_teachers.py \
    --gpu-group "$P0A4_GPUS" --tensor-parallel-size auto \
    --model-dir "$TEACHER_MODEL" --quantization none --port 8000 \
    --lora-module "distill-teacher-v${P0A4_TEACHER_VERSION}=$ROOT/$adapter"
}

evaluate_validation_endpoint() {
  local name="$1"
  local url="$2"
  local model_artifact="$3"
  local data="$4"
  local trace="$5"
  local audit="$6"
  shift 6
  "$PYTHON_BIN" scripts/evaluate_edge_candidate_dev.py \
    --endpoint "$url" --endpoint-model-id auto \
    --model-artifact "$model_artifact" --candidate-name "$name" \
    --validation-data "$data" --output-trace "$trace" --audit "$audit" \
    --max-new-tokens-map "$TOKEN_LIMITS" --request-timeout-sec 240 "$@"
}

router_request_lines() {
  local manifest="$1"
  "$PYTHON_BIN" - "$manifest" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for task in ("gsm8k","humaneval","cmmlu"):
    payload={"lora":d["task_adapters"][task]["request_lora"]}
    print(task+"="+json.dumps(payload,separators=(",",":")))
PY
}

baseline_dev() {
  evaluate_validation_endpoint baseline14b-awq-teacher-validation "$P0A4_BASELINE_URL" "$BASELINE_MODEL" \
    "$TEACHER_VALIDATION" data/eval/p0a4_baseline14b_awq_teacher_validation.jsonl \
    reports/audit/gate_p0a4_baseline14b_awq_teacher_validation.json
  evaluate_validation_endpoint baseline14b-awq-smoke96 "$P0A4_BASELINE_URL" "$BASELINE_MODEL" \
    "$SMOKE96" data/eval/p0a4_baseline14b_awq_smoke96.jsonl \
    reports/audit/gate_p0a4_baseline14b_awq_smoke96.json
  evaluate_validation_endpoint baseline14b-awq-selection170 "$P0A4_BASELINE_URL" "$BASELINE_MODEL" \
    "$SELECTION170" data/eval/p0a4_baseline14b_awq_selection170.jsonl \
    reports/audit/gate_p0a4_baseline14b_awq_selection170.json
}

teacher_train() {
  local version="${1:-$P0A4_TEACHER_VERSION}"
  require_dir "$TEACHER_MODEL"
  require_file "$TRAIN_DATA"
  require_file "$TORCHRUN_BIN"
  CUDA_VISIBLE_DEVICES="$P0A4_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a4_lora.py \
    --config "$CONFIG" --role teacher --candidate-index "$version" \
    --output-dir "models/checkpoints/p0a4/teacher-v${version}" \
    --audit "reports/audit/gate_p0a4_train_teacher_v${version}.json" \
    --deepspeed configs/deepspeed_p0a4_zero3.json
}

teacher_validate() {
  local version="${1:-$P0A4_TEACHER_VERSION}"
  evaluate_validation_endpoint "distill-teacher-v${version}" "$P0A4_TEACHER_URL" \
    "models/checkpoints/p0a4/teacher-v${version}" "$TEACHER_VALIDATION" \
    "data/eval/p0a4_teacher_v${version}_validation.jsonl" \
    "reports/audit/gate_p0a4_teacher_v${version}_validation.json" \
    --endpoint-model-id "$ROOT/$TEACHER_MODEL" \
    --endpoint-model-id-map "cmmlu=distill-teacher-v${version}"
}

teacher_select() {
  "$PYTHON_BIN" - <<'PY'
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
baseline_path=Path("reports/audit/gate_p0a4_baseline14b_awq_teacher_validation.json")
if not baseline_path.is_file():
    raise SystemExit("Missing baseline Teacher-validation audit; run baseline-dev first")
baseline_audit=json.loads(baseline_path.read_text(encoding="utf-8"))
baseline=baseline_audit["accuracy_by_dataset"]
candidates=[]
for version in range(1,4):
    p=Path(f"reports/audit/gate_p0a4_teacher_v{version}_validation.json")
    if not p.is_file():
        continue
    d=json.loads(p.read_text(encoding="utf-8"))
    no_regression=all(float(d.get("accuracy_by_dataset",{}).get(task,0)) >= float(baseline[task])-0.01 for task in baseline)
    if d.get("status") == "passed" and d.get("generation_error_count") == 0 and d.get("dataset_counts") == {"cmmlu":32,"gsm8k":32,"humaneval":32} and no_regression:
        candidates.append((float(d["macro_accuracy"]), version, d))
if not candidates:
    raise SystemExit("No complete Teacher validation candidate")
candidates.sort(key=lambda x:(x[0],-x[1]), reverse=True)
macro, version, audit=candidates[0]
out={
  "gate":"P0-A4-TEACHER-SELECTION", "status":"passed",
  "created_by":"scripts/run_p0a4.sh:teacher_select",
  "created_ts":datetime.now(timezone.utc).isoformat(),
  "selection_data":"data/distill/p0a4_teacher_validation.jsonl",
  "best_version":version, "best_macro_accuracy":macro,
  "best_adapter":f"models/checkpoints/p0a4/teacher-v{version}",
  "best_accuracy_by_dataset":audit["accuracy_by_dataset"],
  "best_endpoint_model_id":audit.get("endpoint_model_id",""),
  "best_endpoint_model_id_map":audit.get("endpoint_model_id_map",{}),
  "deployment_endpoint_model_id_map":{"cmmlu":f"distill-teacher-v{version}"},
  "routing_mode":"explicit_task_top1",
  "baseline_accuracy_by_dataset":baseline,
  "baseline_audit_hash":baseline_audit.get("report_hash",""),
  "selected_candidate_audit_hash":audit.get("report_hash",""),
  "candidate_macro_accuracy":{f"v{v}":m for m,v,_ in candidates},
  "max_candidates":3,
}
out["report_hash"]=hashlib.sha256(json.dumps(out,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
p=Path("reports/audit/gate_p0a4_teacher_selection.json")
p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"Selected Teacher v{version} macro={macro:.6f}")
PY
}

teacher_distill() {
  local version="${1:-$P0A4_TEACHER_VERSION}"
  require_audit_passed reports/audit/gate_p0a4_teacher_selection.json
  "$PYTHON_BIN" - "$version" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path("reports/audit/gate_p0a4_teacher_selection.json").read_text())
if int(sys.argv[1]) != int(d["best_version"]):
    raise SystemExit(f"Requested Teacher v{sys.argv[1]} is not selected v{d['best_version']}")
PY
  "$PYTHON_BIN" model_compression/generate_teacher_capability_distill.py \
    --input-jsonl "$TRAIN_DATA" --output-trace data/distill/p0a4_teacher_trace.jsonl \
    --output-distill "$DISTILL_DATA" --audit reports/audit/gate_p0a4_teacher_distill.json \
    --teacher-url "$P0A4_TEACHER_URL" --teacher-model-id "$ROOT/$TEACHER_MODEL" \
    --teacher-model-id-map "cmmlu=distill-teacher-v${version}" \
    --workers 4 --max-new-tokens 512 --code-timeout-sec 10 \
    --retry-count 2 --checkpoint-interval 25 --min-accept-rate 0.80 \
    --min-group-coverage-rate 0.80 \
    --min-selected-count-map "gsm8k=400,humaneval=230,cmmlu=400" \
    --min-accepted-count-map "gsm8k=400,humaneval=220,cmmlu=400" \
    --min-accept-rate-map "gsm8k=0.75,humaneval=0.75,cmmlu=0.75" \
    --resume --retry-rejected-on-resume
}

student_train() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_distill_ready
  if [[ "$version" == "2" ]]; then
    require_active_student_version 2
    require_audit_passed reports/audit/gate_p0a4_student_v2_protocol.json
  fi
  CUDA_VISIBLE_DEVICES="$P0A4_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    model_compression/train_p0a4_lora.py \
    --config "$CONFIG" --role student_shared --candidate-index "$version" \
    --output-dir "models/checkpoints/p0a4/student-shared-v${version}" \
    --audit "reports/audit/gate_p0a4_train_student_shared_v${version}.json"
}

student_v2_preflight() {
  require_active_student_version 2
  require_distill_ready
  "$PYTHON_BIN" - "$CONFIG" reports/audit/gate_p0a4_edge_student_v1_q4_k_m_selection170_retention.json \
    reports/audit/p0a4_trial_ledger.json <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
config_path,selection_path,ledger_path=map(Path,sys.argv[1:])
for path in (config_path,selection_path,ledger_path):
    if not path.is_file(): raise SystemExit(f"Missing Student v2 protocol input: {path}")
config=json.loads(config_path.read_text(encoding="utf-8"))
selection=json.loads(selection_path.read_text(encoding="utf-8"))
ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
if selection.get("feedback_policy") != "aggregate_by_task_only":
    raise SystemExit("Student v1 selection feedback is not aggregate-only")
trials=[item for item in ledger.get("trials",[]) if item.get("phase")=="selection170"]
if len(trials) != 1 or trials[0].get("version") != "v1-q4_k_m" or trials[0].get("status") != "failed":
    raise SystemExit(f"Student v2 requires exactly one failed v1 selection trial: {trials}")
settings=config["training"]["student_shared"]["candidate_overrides"]["2"]
if config["gates"]["v2_promotion"].get("full_test_feedback_used") is not False:
    raise SystemExit("Student v2 protocol must forbid full-test feedback")
audit={
  "gate":"P0-A4-STUDENT-V2-PROTOCOL", "check_version":"1.0", "status":"passed",
  "created_by":"scripts/run_p0a4.sh:student_v2_preflight",
  "created_ts":datetime.now(timezone.utc).isoformat(),
  "feedback_source":"selection170_task_aggregates_only",
  "selection170_ratios":selection.get("ratios",{}),
  "selection170_capped_macro_ratio":selection.get("capped_macro_ratio"),
  "selection170_audit":str(selection_path),
  "selection170_audit_hash":hashlib.sha256(selection_path.read_bytes()).hexdigest(),
  "full_test_feedback_used":False,
  "formal_trace_read":False,
  "v2_shared_settings":settings,
  "v2_promotion_gate":config["gates"]["v2_promotion"],
  "remaining_selection170_slots":1,
}
audit["report_hash"]=hashlib.sha256(json.dumps(audit,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
path=Path("reports/audit/gate_p0a4_student_v2_protocol.json")
path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"Student v2 protocol passed: {path}")
PY
  "$PYTHON_BIN" model_compression/train_p0a4_lora.py --config "$CONFIG" \
    --role student_shared --candidate-index 2 \
    --output-dir models/checkpoints/p0a4/student-shared-v2 \
    --audit reports/audit/gate_p0a4_train_student_shared_v2_preflight.json --dry-run
  local task
  for task in gsm8k humaneval cmmlu; do
    "$PYTHON_BIN" model_compression/train_p0a4_lora.py --config "$CONFIG" \
      --role student_expert --candidate-index 2 --task "$task" \
      --model-dir "$STUDENT_MERGED_DIR" \
      --output-dir "models/checkpoints/p0a4/student-expert-${task}-v2" \
      --audit "reports/audit/gate_p0a4_train_student_expert_${task}_v2_preflight.json" --dry-run
  done
}

student_merge() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_active_student_version "$version"
  require_student_training_ready "$version"
  "$PYTHON_BIN" model_compression/merge_p0a4_adapter.py \
    --base-model "$STUDENT_MODEL" \
    --adapter "models/checkpoints/p0a4/student-shared-v${version}" \
    --output "$STUDENT_MERGED_DIR" \
    --audit "$STUDENT_MERGE_AUDIT"
}

student_check() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_active_student_version "$version"
  require_student_merge_ready "$version"
  if [[ -f "$EDGE_GGUF" || -f "$EDGE_QUANT_AUDIT" ]]; then
    require_quantized_student_ready
    echo "Student v${version} is ready for edge-start and student-smoke96."
  else
    echo "Student v${version} is merged and ready for student-quantize."
  fi
}

student_experts() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_active_student_version "$version"
  require_student_merge_ready "$version"
  if [[ "$version" == "2" ]]; then
    require_audit_passed reports/audit/gate_p0a4_student_v2_protocol.json
  fi
  local task
  for task in gsm8k humaneval cmmlu; do
    CUDA_VISIBLE_DEVICES="$P0A4_GPUS" "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
      model_compression/train_p0a4_lora.py \
      --config "$CONFIG" --role student_expert --candidate-index "$version" --task "$task" \
      --model-dir "$STUDENT_MERGED_DIR" \
      --output-dir "models/checkpoints/p0a4/student-expert-${task}-v${version}" \
      --audit "reports/audit/gate_p0a4_train_student_expert_${task}_v${version}.json"
  done
}

prepare_adapters() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_active_student_version "$version"
  require_student_merge_ready "$version"
  "$PYTHON_BIN" scripts/prepare_p0a4_adapters.py \
    --adapter "gsm8k=models/checkpoints/p0a4/student-expert-gsm8k-v${version}" \
    --adapter "humaneval=models/checkpoints/p0a4/student-expert-humaneval-v${version}" \
    --adapter "cmmlu=models/checkpoints/p0a4/student-expert-cmmlu-v${version}" \
    --base "$STUDENT_MERGED_DIR" --base-audit "$STUDENT_MERGE_AUDIT" \
    --output-dir "$ADAPTER_DIR" \
    --manifest "$ADAPTER_MANIFEST" --audit "$ADAPTER_PREPARE_AUDIT"
  require_adapter_router_ready
}

student_quantize() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_active_student_version "$version"
  require_student_merge_ready "$version"
  "$PYTHON_BIN" scripts/build_imatrix_calibration.py \
    --source "$DISTILL_DATA" --output "$IMATRIX_CALIBRATION" \
    --audit "$IMATRIX_CALIBRATION_AUDIT" \
    --stratify-key dataset_key --stratum gsm8k --stratum humaneval --stratum cmmlu \
    --rows-per-stratum 128 --seed 202606
  "$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$STUDENT_MERGED_DIR" \
    --outfile "$EDGE_F16_GGUF" --outtype f16
  "$LLAMA_CPP_DIR/build/bin/llama-imatrix" \
    --model "$EDGE_F16_GGUF" \
    --file "$IMATRIX_CALIBRATION" --output-frequency 10 \
    --output-file "$EDGE_IMATRIX" --ctx-size 512 --threads 8
  "$PYTHON_BIN" scripts/prepare_edge_gguf.py \
    --merged-hf-dir "$STUDENT_MERGED_DIR" \
    --llama-cpp-dir "$LLAMA_CPP_DIR" \
    --f16-gguf "$EDGE_F16_GGUF" \
    --quantized-gguf "$EDGE_GGUF" --quant-type "$EDGE_QUANT_TYPE" \
    --imatrix "$EDGE_IMATRIX" --max-quantized-bytes 1150000000 \
    --skip-f16-if-exists \
    --audit "$EDGE_QUANT_AUDIT"
  require_quantized_student_ready
}

start_edge_server() {
  require_quantized_student_ready
  mkdir -p logs/p0a4 runtime
  local alias="p0a4-edge-student-v${P0A4_STUDENT_VERSION}-${EDGE_QUANT_TAG//_/-}"
  [[ "$P0A4_EDGE_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid P0A4_EDGE_PARALLEL" >&2; return 2; }
  [[ "$P0A4_EDGE_CTX_PER_SLOT" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid P0A4_EDGE_CTX_PER_SLOT" >&2; return 2; }
  local edge_ctx_size=$((P0A4_EDGE_CTX_PER_SLOT * P0A4_EDGE_PARALLEL))
  grep -q '^GGML_CUDA:BOOL=ON$' "$LLAMA_CPP_DIR/build/CMakeCache.txt" || {
    echo "llama.cpp is not CUDA-enabled; run: bash scripts/run_p0a4.sh build-llama-cuda" >&2
    return 1
  }
  local server_extra=()
  if [[ -n "$P0A4_ADAPTER_ROUTER_MANIFEST" ]]; then
    require_adapter_router_ready
    local adapter_paths
    adapter_paths="$($PYTHON_BIN - "$P0A4_ADAPTER_ROUTER_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
root=Path.cwd()
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(",".join(str(root/d["task_adapters"][task]["path"]) for task in ("gsm8k","humaneval","cmmlu")))
PY
)"
    server_extra+=(--lora "$adapter_paths" --lora-init-without-apply)
  fi
  CUDA_VISIBLE_DEVICES="$P0A4_EDGE_GPUS" nohup "$LLAMA_CPP_DIR/build/bin/llama-server" \
    --model "$EDGE_GGUF" --alias "$alias" \
    --host 127.0.0.1 --port "$P0A4_EDGE_PORT" --ctx-size "$edge_ctx_size" \
    --threads 8 --parallel "$P0A4_EDGE_PARALLEL" --batch-size 32 --ubatch-size 16 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
    --cache-ram 0 --no-cache-idle-slots \
    --n-gpu-layers all --no-repack --reasoning off --reasoning-format none \
    --skip-chat-parsing "${server_extra[@]}" \
    >logs/p0a4/edge_server.log 2>&1 </dev/null &
  local server_pid=$!
  echo "$server_pid" > runtime/p0a4_edge.pid
  "$PYTHON_BIN" - "$P0A4_EDGE_URL" <<'PY'
import sys,time
from urllib.request import Request,urlopen
url=sys.argv[1].rstrip('/')+'/health'
for _ in range(240):
    try:
        with urlopen(Request(url),timeout=2) as r:
            if r.status == 200: print("Edge server healthy"); raise SystemExit(0)
    except Exception: time.sleep(.5)
raise SystemExit("Edge server health timeout")
PY
  if ! "$PYTHON_BIN" - "$server_pid" "$P0A4_EDGE_URL" "$EDGE_GGUF" \
      "$EDGE_QUANT_AUDIT" "$alias" "$P0A4_EDGE_GPUS" "$P0A4_EDGE_PARALLEL" \
      "$edge_ctx_size" "$P0A4_ADAPTER_ROUTER_MANIFEST" <<'PY'
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
pid=int(sys.argv[1]); endpoint=sys.argv[2].rstrip('/'); gguf=Path(sys.argv[3]); quant_path=Path(sys.argv[4]); alias=sys.argv[5]
visible_gpus=sys.argv[6]; parallel=int(sys.argv[7]); ctx_size=int(sys.argv[8]); manifest_value=sys.argv[9]
os.kill(pid,0)
cmdline=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
required=(str(gguf),"--cache-type-k q8_0","--cache-type-v q8_0","--cache-ram 0","--no-cache-idle-slots",f"--parallel {parallel}",f"--ctx-size {ctx_size}")
missing=[item for item in required if item not in cmdline]
if missing:
    raise SystemExit(f"Edge server command is missing fixed runtime arguments: {missing}")
with urlopen(Request(endpoint+'/v1/models'),timeout=3) as response:
    models=json.load(response)
ids={str(item.get('id','')) for item in models.get('data',[])}
if alias not in ids:
    raise SystemExit(f"Edge server alias mismatch: expected={alias} actual={sorted(ids)}")
gpu_rows=subprocess.run(
    ["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],
    text=True,capture_output=True,check=True,
).stdout.splitlines()
gpu_memory=[]
for row in gpu_rows:
    fields=[field.strip() for field in row.split(",")]
    if len(fields)==2 and fields[0]==str(pid):
        gpu_memory.append(int(fields[1]))
if not gpu_memory:
    raise SystemExit(f"llama-server pid {pid} has no active CUDA allocation")
quant=json.loads(quant_path.read_text(encoding="utf-8"))
manifest_hash=""
if manifest_value:
    manifest_path=Path(manifest_value)
    manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
audit={
  "gate":"P0-A4-EDGE-SERVICE", "check_version":"1.0", "status":"passed",
  "created_by":"scripts/run_p0a4.sh:start_edge_server",
  "created_ts":datetime.now(timezone.utc).isoformat(), "pid":pid,
  "endpoint":endpoint, "alias":alias, "model_path":str(gguf),
  "model_hash":quant["quantized_gguf_hash"], "quantization_audit":str(quant_path),
  "quant_type":quant["quant_type"], "kv_cache_type":"q8_0",
  "host_prompt_cache_mib":0, "cache_idle_slots":False,
  "disable_thinking":True, "command_line":cmdline,
  "gpu_backend":True, "cuda_visible_devices":visible_gpus,
  "gpu_memory_mib":gpu_memory, "parallel":parallel, "ctx_size":ctx_size,
  "adapter_manifest":manifest_value, "adapter_manifest_hash":manifest_hash,
}
audit["report_hash"]=hashlib.sha256(json.dumps(audit,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
path=Path("reports/audit/gate_p0a4_edge_service.json")
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"Edge service audit passed: {path} pid={pid} alias={alias}")
PY
  then
    kill "$server_pid" 2>/dev/null || true
    rm -f runtime/p0a4_edge.pid
    return 1
  fi
}

stop_edge_server() {
  if [[ -f runtime/p0a4_edge.pid ]]; then
    kill "$(cat runtime/p0a4_edge.pid)" 2>/dev/null || true
    rm -f runtime/p0a4_edge.pid
  fi
}

start_edge_server_adapters() {
  P0A4_ADAPTER_ROUTER_MANIFEST="$ADAPTER_MANIFEST"
  export P0A4_ADAPTER_ROUTER_MANIFEST
  require_adapter_router_ready
  start_edge_server
}

student_smoke96() {
  local version="${P0A4_STUDENT_VERSION}"
  local trace="data/eval/p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_smoke96.jsonl"
  local eval_audit="reports/audit/gate_p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_smoke96.json"
  local retention_audit="reports/audit/gate_p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_smoke96_retention.json"
  require_quantized_student_ready
  require_edge_service_ready
  require_audit_passed reports/audit/gate_p0a4_baseline14b_awq_smoke96.json
  require_file data/eval/p0a4_baseline14b_awq_smoke96.jsonl
  evaluate_validation_endpoint "edge-student-v${version}-${EDGE_QUANT_TAG//_/-}" "$P0A4_EDGE_URL" "$EDGE_GGUF" \
    "$SMOKE96" "$trace" "$eval_audit" \
    --disable-thinking --kv-cache-type q8_0
  require_student_eval_runtime "$eval_audit"
  "$PYTHON_BIN" scripts/p0a4_retention_gate.py --config "$CONFIG" --gate smoke96 \
    --baseline-trace data/eval/p0a4_baseline14b_awq_smoke96.jsonl \
    --candidate-trace "$trace" \
    --candidate-name "edge-student-v${version}-${EDGE_QUANT_TAG//_/-}" \
    --output "$retention_audit"
  require_edge_smoke96_ready "$version"
}

student_selection170_check() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  local trial_version="v${version}-${EDGE_QUANT_TAG}"
  require_quantized_student_ready
  require_baseline_selection170_ready
  require_edge_smoke96_ready "$version"
  require_edge_memory_precheck_ready
  require_edge_service_ready
  "$PYTHON_BIN" - "$CONFIG" reports/audit/p0a4_trial_ledger.json "$trial_version" <<'PY'
import json,sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ledger_path=Path(sys.argv[2]); version=sys.argv[3]
ledger=json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {"trials":[]}
trials=[item for item in ledger.get("trials",[]) if item.get("phase")=="selection170"]
same=[item for item in trials if item.get("version")==version]
if same and not (len(same)==1 and same[0].get("status")=="reserved"):
    raise SystemExit(f"Selection170 trial already consumed: {same}")
limit=int(config["gates"]["selection170"]["max_student_versions"])
if not same and len(trials) >= limit:
    raise SystemExit(f"Selection170 trial limit reached: {len(trials)}/{limit}")
mode="resume reserved trial" if same else "reserve new trial"
print(f"Selection170 trial guard passed: {mode} {version}; used={len(trials)}/{limit}")
PY
  echo "READY: Q4_K_M Student v${version} may start the 170-row selection gate."
}

student_selection170() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  if [[ "$version" == "2" && "${P0A4_V2_ROUTE_AUTHORIZED:-}" != "shared" ]]; then
    echo "Direct v2 shared 170 is locked; run student-v2-select-route then student-v2-170." >&2
    return 2
  fi
  local candidate="edge-student-v${version}-${EDGE_QUANT_TAG//_/-}"
  local trace="data/eval/p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_selection170.jsonl"
  local eval_audit="reports/audit/gate_p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_selection170_eval.json"
  local retention_audit="reports/audit/gate_p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_selection170_retention.json"
  local trial_version="v${version}-${EDGE_QUANT_TAG}"
  student_selection170_check "$version"
  "$PYTHON_BIN" scripts/p0a4_trials.py reserve --phase selection170 --version "$trial_version" --resume-reserved
  local result=0
  evaluate_validation_endpoint "$candidate" "$P0A4_EDGE_URL" "$EDGE_GGUF" \
    "$SELECTION170" "$trace" "$eval_audit" \
    --disable-thinking --kv-cache-type q8_0 || result=$?
  if [[ "$result" -eq 0 ]]; then
    require_student_eval_runtime "$eval_audit" || result=$?
  fi
  if [[ "$result" -eq 0 ]]; then
    "$PYTHON_BIN" scripts/p0a4_retention_gate.py --config "$CONFIG" --gate selection170 \
      --baseline-trace data/eval/p0a4_baseline14b_awq_selection170.jsonl \
      --candidate-trace "$trace" --candidate-name "$candidate" \
      --output "$retention_audit" || result=$?
  fi
  local action="complete"
  [[ "$result" -eq 0 ]] || action="fail"
  local ledger_audit="$retention_audit"
  [[ -f "$ledger_audit" ]] || ledger_audit="$eval_audit"
  "$PYTHON_BIN" scripts/p0a4_trials.py "$action" --phase selection170 --version "$trial_version" \
    --audit "$ledger_audit"
  return "$result"
}

student_adapter_smoke96() {
  local version="$P0A4_STUDENT_VERSION"
  local route_tag="$P0A4_ADAPTER_ROUTE_TAG"
  [[ "$route_tag" =~ ^[a-z0-9_]+$ ]] || {
    echo "Invalid P0A4_ADAPTER_ROUTE_TAG: $route_tag" >&2
    return 2
  }
  P0A4_ADAPTER_ROUTER_MANIFEST="${P0A4_ADAPTER_ROUTER_MANIFEST:-$ADAPTER_MANIFEST}"
  export P0A4_ADAPTER_ROUTER_MANIFEST
  local manifest="$P0A4_ADAPTER_ROUTER_MANIFEST"
  require_active_student_version "$version"
  require_quantized_student_ready
  require_adapter_router_ready
  require_edge_service_ready
  local extra=()
  while IFS= read -r line; do extra+=(--request-extra-json-map "$line"); done < <(router_request_lines "$manifest")
  evaluate_validation_endpoint "edge-student-v${version}-adapter-top1" \
    "$P0A4_EDGE_URL" "$EDGE_GGUF" "$SMOKE96" \
    "data/eval/p0a4_edge_student_v${version}_${route_tag}_smoke96.jsonl" \
    "reports/audit/gate_p0a4_edge_student_v${version}_${route_tag}_smoke96_eval.json" \
    --disable-thinking --kv-cache-type q8_0 "${extra[@]}"
  require_student_eval_runtime "reports/audit/gate_p0a4_edge_student_v${version}_${route_tag}_smoke96_eval.json"
  "$PYTHON_BIN" scripts/p0a4_retention_gate.py --config "$CONFIG" --gate smoke96 \
    --baseline-trace data/eval/p0a4_baseline14b_awq_smoke96.jsonl \
    --candidate-trace "data/eval/p0a4_edge_student_v${version}_${route_tag}_smoke96.jsonl" \
    --candidate-name "edge-student-v${version}-${route_tag//_/-}-top1" \
    --output "reports/audit/gate_p0a4_edge_student_v${version}_${route_tag}_smoke96_retention.json"
}

student_adapter_selection170() {
  local version="${1:-2}"
  require_active_student_version "$version"
  [[ "$version" == "2" ]] || { echo "The Adapter selection route is reserved for Student v2" >&2; return 2; }
  [[ "${P0A4_V2_ROUTE_AUTHORIZED:-}" == "adapter_top1" ]] || {
    echo "Direct v2 Adapter 170 is locked; run student-v2-select-route then student-v2-170." >&2
    return 2
  }
  local trial_version="v${version}-adapter-${EDGE_QUANT_TAG}"
  P0A4_ADAPTER_ROUTER_MANIFEST="${P0A4_ADAPTER_ROUTER_MANIFEST:-$ADAPTER_MANIFEST}"
  export P0A4_ADAPTER_ROUTER_MANIFEST
  local manifest="$P0A4_ADAPTER_ROUTER_MANIFEST"
  require_quantized_student_ready
  require_adapter_router_ready
  require_edge_service_ready
  require_audit_passed "reports/audit/gate_p0a4_edge_student_v${version}_adapter_smoke96_retention.json"
  "$PYTHON_BIN" scripts/p0a4_trials.py reserve --phase selection170 --version "$trial_version" --resume-reserved
  local extra=()
  while IFS= read -r line; do extra+=(--request-extra-json-map "$line"); done < <(router_request_lines "$manifest")
  local result=0
  evaluate_validation_endpoint "edge-student-v${version}-adapter-top1" "$P0A4_EDGE_URL" "$EDGE_GGUF" \
    "$SELECTION170" "data/eval/p0a4_edge_student_v${version}_adapter_selection170.jsonl" \
    "reports/audit/gate_p0a4_edge_student_v${version}_adapter_selection170_eval.json" \
    --disable-thinking --kv-cache-type q8_0 "${extra[@]}" || result=$?
  if [[ "$result" -eq 0 ]]; then
    require_student_eval_runtime \
      "reports/audit/gate_p0a4_edge_student_v${version}_adapter_selection170_eval.json" || result=$?
  fi
  if [[ "$result" -eq 0 ]]; then
    "$PYTHON_BIN" scripts/p0a4_retention_gate.py --config "$CONFIG" --gate selection170 \
      --baseline-trace data/eval/p0a4_baseline14b_awq_selection170.jsonl \
      --candidate-trace "data/eval/p0a4_edge_student_v${version}_adapter_selection170.jsonl" \
      --candidate-name "edge-student-v${version}-adapter-top1" \
      --output "reports/audit/gate_p0a4_edge_student_v${version}_adapter_selection170_retention.json" || result=$?
  fi
  local action="complete"; [[ "$result" -eq 0 ]] || action="fail"
  local audit="reports/audit/gate_p0a4_edge_student_v${version}_adapter_selection170_retention.json"
  [[ -f "$audit" ]] || audit="reports/audit/gate_p0a4_edge_student_v${version}_adapter_selection170_eval.json"
  "$PYTHON_BIN" scripts/p0a4_trials.py "$action" --phase selection170 \
    --version "$trial_version" --audit "$audit"
  return "$result"
}

student_adapter_memory_precheck() {
  local version="${1:-$P0A4_STUDENT_VERSION}"
  require_active_student_version "$version"
  P0A4_ADAPTER_ROUTER_MANIFEST="${P0A4_ADAPTER_ROUTER_MANIFEST:-$ADAPTER_MANIFEST}"
  export P0A4_ADAPTER_ROUTER_MANIFEST
  require_adapter_router_ready
  require_audit_passed "reports/audit/gate_p0a4_edge_student_v${version}_adapter_smoke96_retention.json"
  local paths request
  paths="$($PYTHON_BIN - "$ADAPTER_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
root=Path.cwd(); d=json.loads(Path(sys.argv[1]).read_text())
print(",".join(str(root/d["task_adapters"][t]["path"]) for t in ("gsm8k","humaneval","cmmlu")))
PY
)"
  request="$($PYTHON_BIN - "$ADAPTER_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({"lora":d["task_adapters"]["humaneval"]["request_lora"]},separators=(",",":")))
PY
)"
  "$PYTHON_BIN" scripts/verify_gate_g3_gguf.py \
    --gguf "$EDGE_GGUF" --llama-cpp-dir "$LLAMA_CPP_DIR" \
    --teacher-trace data/distill/teacher_decision_trace.jsonl \
    --audit "reports/audit/gate_p0a4_edge_student_v${version}_adapter_memory_precheck.json" \
    --memory-trace-csv "reports/audit/gate_p0a4_edge_student_v${version}_adapter_memory_precheck.samples.csv" \
    --warmup-requests 5 --measure-requests 20 --max-tokens 128 --ctx-size 512 \
    --threads 8 --parallel 1 --batch-size 32 --ubatch-size 16 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --no-repack \
    --host-prompt-cache-mib 0 --no-cache-idle-slots --n-gpu-layers 0 \
    --port 18445 --timeout-sec 240 --max-total-memory-mb 1400 --sample-interval-ms 50 \
    --quantization-label "${EDGE_QUANT_TYPE}_Q8KV_V${version}_ADAPTER_PRECHECK" \
    --server-extra-arg=--lora --server-extra-arg="$paths" \
    --server-extra-arg=--lora-init-without-apply --request-extra-json "$request"
  require_adapter_memory_precheck_ready
}

student_v2_select_route() {
  require_active_student_version 2
  require_quantized_student_ready
  require_adapter_router_ready
  "$PYTHON_BIN" scripts/p0a4_select_v2_route.py --config "$CONFIG" \
    --quant-audit "$EDGE_QUANT_AUDIT" --adapter-manifest "$ADAPTER_MANIFEST" \
    --adapter-audit "$ADAPTER_PREPARE_AUDIT" \
    --output reports/audit/gate_p0a4_student_v2_route_selection.json
}

start_edge_server_v2_selected() {
  require_active_student_version 2
  local route
  route="$(require_v2_route_ready)"
  if [[ "$route" == "adapter_top1" ]]; then
    P0A4_ADAPTER_ROUTER_MANIFEST="$ADAPTER_MANIFEST"
    export P0A4_ADAPTER_ROUTER_MANIFEST
  else
    P0A4_ADAPTER_ROUTER_MANIFEST=""
    export P0A4_ADAPTER_ROUTER_MANIFEST
  fi
  start_edge_server
}

student_v2_selection170() {
  require_active_student_version 2
  local route
  route="$(require_v2_route_ready)"
  if [[ "$route" == "shared" ]]; then
    require_edge_memory_precheck_ready
    P0A4_V2_ROUTE_AUTHORIZED=shared student_selection170 2
  else
    P0A4_ADAPTER_ROUTER_MANIFEST="$ADAPTER_MANIFEST"
    export P0A4_ADAPTER_ROUTER_MANIFEST
    require_adapter_memory_precheck_ready
    P0A4_V2_ROUTE_AUTHORIZED=adapter_top1 student_adapter_selection170 2
  fi
}

student_v2_memory() {
  require_active_student_version 2
  local route
  route="$(require_v2_route_ready)"
  if [[ "$route" == "shared" ]]; then
    student_memory
  else
    P0A4_ADAPTER_ROUTER_MANIFEST="$ADAPTER_MANIFEST"
    export P0A4_ADAPTER_ROUTER_MANIFEST
    student_adapter_memory 2
  fi
}

student_memory() {
  local version="${P0A4_STUDENT_VERSION}"
  local audit="reports/audit/gate_p0a4_edge_student_v${version}_memory_dev.json"
  require_audit_passed "reports/audit/gate_p0a4_edge_student_v${version}_${EDGE_QUANT_TAG}_selection170_retention.json"
  "$PYTHON_BIN" scripts/verify_gate_g3_gguf.py \
    --gguf "$EDGE_GGUF" --llama-cpp-dir "$LLAMA_CPP_DIR" \
    --teacher-trace data/distill/teacher_decision_trace.jsonl \
    --audit "$audit" \
    --memory-trace-csv "${audit%.json}.samples.csv" \
    --warmup-requests 20 --measure-requests 100 --max-tokens 128 --ctx-size 512 \
    --batch-size 32 --ubatch-size 16 --cache-type-k q8_0 --cache-type-v q8_0 \
    --host-prompt-cache-mib 0 --no-cache-idle-slots \
    --max-total-memory-mb 1400 --sample-interval-ms 50 --quantization-label "${EDGE_QUANT_TYPE}_Q8KV"
}

student_memory_precheck_q4() {
  local version="${P0A4_STUDENT_VERSION}"
  local eval_audit="reports/audit/gate_p0a4_edge_student_v${version}_q4_k_m_smoke96.json"
  local retention_audit="reports/audit/gate_p0a4_edge_student_v${version}_q4_k_m_smoke96_retention.json"
  require_quantized_student_ready
  require_audit_passed "$eval_audit"
  require_audit_passed "$retention_audit"
  "$PYTHON_BIN" - "$EDGE_GGUF" "$eval_audit" <<'PY'
import hashlib,json,sys
from pathlib import Path
gguf,audit_path=map(Path,sys.argv[1:])
audit=json.loads(audit_path.read_text(encoding="utf-8"))
model=audit.get("model",{})
digest=hashlib.sha256()
with gguf.open("rb") as handle:
    for chunk in iter(lambda:handle.read(1024*1024),b""):
        digest.update(chunk)
if Path(str(model.get("path",""))) != gguf:
    raise SystemExit("Q4 smoke96 audit points to a different model path")
if digest.hexdigest() != model.get("sha256"):
    raise SystemExit("Q4 model hash does not match its smoke96 audit")
print(f"Q4 precheck guard passed: {gguf} bytes={gguf.stat().st_size}")
PY
  "$PYTHON_BIN" scripts/verify_gate_g3_gguf.py \
    --gguf "$EDGE_GGUF" --llama-cpp-dir "$LLAMA_CPP_DIR" \
    --teacher-trace data/distill/teacher_decision_trace.jsonl \
    --audit "$EDGE_MEMORY_PRECHECK_AUDIT" \
    --memory-trace-csv "${EDGE_MEMORY_PRECHECK_AUDIT%.json}.samples.csv" \
    --warmup-requests 5 --measure-requests 20 --max-tokens 128 --ctx-size 512 \
    --threads 8 --parallel 1 --batch-size 32 --ubatch-size 16 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --no-repack \
    --host-prompt-cache-mib 0 --no-cache-idle-slots \
    --port 18444 --timeout-sec 240 --max-total-memory-mb 1400 \
    --n-gpu-layers 0 --sample-interval-ms 50 \
    --quantization-label Q4_K_M_Q8KV_CACHE_OFF_PRECHECK \
    --keep-server-log "logs/p0a4/v${version}_q4_k_m_memory_precheck_cache_off_server.log"
}

student_adapter_memory() {
  local version="${1:-2}"
  require_active_student_version "$version"
  local manifest="${P0A4_ADAPTER_ROUTER_MANIFEST:-$ADAPTER_MANIFEST}"
  P0A4_ADAPTER_ROUTER_MANIFEST="$manifest"
  export P0A4_ADAPTER_ROUTER_MANIFEST
  require_adapter_router_ready
  require_audit_passed "reports/audit/gate_p0a4_edge_student_v${version}_adapter_selection170_retention.json"
  local paths
  paths="$($PYTHON_BIN - "$manifest" <<'PY'
import json,sys
from pathlib import Path
root=Path.cwd(); d=json.loads(Path(sys.argv[1]).read_text())
print(",".join(str(root/d["task_adapters"][t]["path"]) for t in ("gsm8k","humaneval","cmmlu")))
PY
)"
  local task index=0
  for task in gsm8k humaneval cmmlu; do
    local request
    request="$($PYTHON_BIN - "$manifest" "$task" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text()); t=sys.argv[2]
print(json.dumps({"lora":d["task_adapters"][t]["request_lora"]},separators=(",",":")))
PY
)"
    "$PYTHON_BIN" scripts/verify_gate_g3_gguf.py \
      --gguf "$EDGE_GGUF" --llama-cpp-dir "$LLAMA_CPP_DIR" \
      --teacher-trace data/distill/teacher_decision_trace.jsonl \
      --audit "reports/audit/gate_p0a4_edge_student_v${version}_adapter_memory_${task}.json" \
      --memory-trace-csv "reports/audit/gate_p0a4_edge_student_v${version}_adapter_memory_${task}.samples.csv" \
      --warmup-requests 20 --measure-requests 100 --max-tokens 128 --ctx-size 512 \
      --batch-size 32 --ubatch-size 16 --cache-type-k q8_0 --cache-type-v q8_0 \
      --host-prompt-cache-mib 0 --no-cache-idle-slots \
      --max-total-memory-mb 1400 --sample-interval-ms 50 --port "$((18090+index))" \
      --quantization-label "${EDGE_QUANT_TYPE}_Q8KV_ADAPTER_TOP1" \
      --server-extra-arg=--lora --server-extra-arg="$paths" \
      --server-extra-arg=--lora-init-without-apply --request-extra-json "$request"
    index=$((index+1))
  done
  "$PYTHON_BIN" - "$version" <<'PY'
import json
import sys
from pathlib import Path
version=sys.argv[1]
tasks=("gsm8k","humaneval","cmmlu")
reports={t:json.loads(Path(f"reports/audit/gate_p0a4_edge_student_v{version}_adapter_memory_{t}.json").read_text()) for t in tasks}
out={
 "gate":"P0-A4-ADAPTER-MEMORY", "status":"passed" if all(d.get("status")=="passed" for d in reports.values()) else "failed",
 "student_version":version,
 "all_adapters_resident":True, "top1_active_per_run":True,
 "peak_total_memory_mb_decimal":max(float(d.get("peak_total_memory_mb_decimal",0)) for d in reports.values()),
 "task_audits":{t:f"reports/audit/gate_p0a4_edge_student_v{version}_adapter_memory_{t}.json" for t in tasks},
}
p=Path(f"reports/audit/gate_p0a4_edge_student_v{version}_adapter_memory_dev.json")
p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"Adapter memory status={out['status']} peak={out['peak_total_memory_mb_decimal']}")
raise SystemExit(0 if out["status"]=="passed" else 1)
PY
}

run_official_shards() {
  local url="$1"
  local model_dir="$2"
  local prefix="$3"
  shift 3
  mkdir -p reports/sealed/p0a4/shards
  local pids=()
  local index
  for index in 0 1 2 3; do
    "$PYTHON_BIN" scripts/evaluate_chapter2_capability.py \
      --student-url "$url" --local-model-dir "$model_dir" \
      --use-frozen-final --split-dir "$FULL_SPLITS" --sample-limit-per-dataset 0 \
      --num-shards 4 --shard-index "$index" --prompt-style v15 \
      --max-new-tokens-map "$TOKEN_LIMITS" --humaneval-timeout-sec 10 --timeout-sec 240 \
      --output-trace "reports/sealed/p0a4/shards/${prefix}_${index}.jsonl" \
      --audit "reports/audit/${prefix}_${index}.json" "$@" &
    pids+=("$!")
  done
  local failed=0
  for index in "${pids[@]}"; do wait "$index" || failed=1; done
  [[ "$failed" -eq 0 ]] || return 1
}

baseline_full() {
  "$PYTHON_BIN" scripts/p0a4_trials.py reserve --phase baseline_full --version baseline14b-awq --resume-reserved
  run_official_shards "$P0A4_BASELINE_URL" "$BASELINE_MODEL" p0a4_baseline_full_shard
  "$PYTHON_BIN" scripts/p0a4_merge_shards.py \
    --role baseline --model-name Baseline-14B-AWQ --split-dir "$FULL_SPLITS" \
    --input reports/sealed/p0a4/shards/p0a4_baseline_full_shard_0.jsonl \
    --input reports/sealed/p0a4/shards/p0a4_baseline_full_shard_1.jsonl \
    --input reports/sealed/p0a4/shards/p0a4_baseline_full_shard_2.jsonl \
    --input reports/sealed/p0a4/shards/p0a4_baseline_full_shard_3.jsonl \
    --output "$BASELINE_FULL_TRACE" --audit "$BASELINE_FULL_AUDIT"
  "$PYTHON_BIN" scripts/p0a4_trials.py complete --phase baseline_full --version baseline14b-awq \
    --audit "$BASELINE_FULL_AUDIT" --seal-trace "$BASELINE_FULL_TRACE"
}

student_full() {
  local route="shared"
  local authorization="gated"
  local selection_audit="reports/audit/gate_p0a4_edge_student_v${P0A4_STUDENT_VERSION}_${EDGE_QUANT_TAG}_selection170_retention.json"
  local memory_audit="reports/audit/gate_p0a4_edge_student_v${P0A4_STUDENT_VERSION}_memory_dev.json"
  local shard_prefix="p0a4_student_${EDGE_QUANT_TAG}_full_shard"
  local trial_version="v${P0A4_STUDENT_VERSION}-${EDGE_QUANT_TAG}"
  local extra=()
  if [[ -n "$P0A4_ADAPTER_ROUTER_MANIFEST" ]]; then
    route="adapter-top1"
    selection_audit="reports/audit/gate_p0a4_edge_student_v${P0A4_STUDENT_VERSION}_adapter_selection170_retention.json"
    memory_audit="reports/audit/gate_p0a4_edge_student_v${P0A4_STUDENT_VERSION}_adapter_memory_dev.json"
    while IFS= read -r line; do extra+=(--request-extra-json-map "$line"); done < <(router_request_lines "$P0A4_ADAPTER_ROUTER_MANIFEST")
  fi
  if [[ "${P0A4_ALLOW_UNGATED_FULL:-0}" == "1" ]]; then
    authorization="explicit-user-override-skip-selection170-and-memory"
    require_quantized_student_ready
    require_edge_service_ready
    "$PYTHON_BIN" - "$selection_audit" "$memory_audit" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
selection,memory=map(Path,sys.argv[1:])
audit={
  "gate":"P0-A4-STUDENT-FULL-AUTHORIZATION", "check_version":"1.0", "status":"passed",
  "created_by":"scripts/run_p0a4.sh:student_full",
  "created_ts":datetime.now(timezone.utc).isoformat(),
  "authorization":"explicit_user_override",
  "waived_prerequisites":["selection170","memory_20_plus_100"],
  "selection170_audit":str(selection), "selection170_audit_exists":selection.is_file(),
  "memory_audit":str(memory), "memory_audit_exists":memory.is_file(),
  "warning":"This invocation consumes the single official-full Student attempt.",
}
audit["report_hash"]=hashlib.sha256(json.dumps(audit,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
path=Path("reports/audit/gate_p0a4_student_full_authorization.json")
path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(f"Ungated official-full authorization recorded: {path}")
PY
  else
    require_audit_passed "$selection_audit"
    require_audit_passed "$memory_audit"
  fi
  require_audit_passed "$BASELINE_FULL_AUDIT"
  "$PYTHON_BIN" scripts/p0a4_trials.py reserve --phase official_full_student \
    --version "$trial_version" --resume-reserved
  run_official_shards "$P0A4_EDGE_URL" "$EDGE_GGUF" "$shard_prefix" \
    --disable-thinking --kv-cache-type q8_0 "${extra[@]}"
  local shard_index
  for shard_index in 0 1 2 3; do
    require_student_eval_runtime "reports/audit/${shard_prefix}_${shard_index}.json"
  done
  "$PYTHON_BIN" scripts/p0a4_merge_shards.py \
    --role student --model-name "Edge-Student-v${P0A4_STUDENT_VERSION}-${route}-${EDGE_QUANT_TYPE}-Q8KV-${authorization}" \
    --split-dir "$FULL_SPLITS" \
    --input "reports/sealed/p0a4/shards/${shard_prefix}_0.jsonl" \
    --input "reports/sealed/p0a4/shards/${shard_prefix}_1.jsonl" \
    --input "reports/sealed/p0a4/shards/${shard_prefix}_2.jsonl" \
    --input "reports/sealed/p0a4/shards/${shard_prefix}_3.jsonl" \
    --output "$STUDENT_FULL_TRACE" --audit "$STUDENT_FULL_AUDIT"
  local result=0
  "$PYTHON_BIN" scripts/p0a4_retention_gate.py --config "$CONFIG" --gate official_full \
    --baseline-trace "$BASELINE_FULL_TRACE" --candidate-trace "$STUDENT_FULL_TRACE" \
    --candidate-name "edge-student-v${P0A4_STUDENT_VERSION}-${EDGE_QUANT_TAG//_/-}" \
    --output reports/audit/gate_p0a4_official_full_retention.json || result=$?
  "$PYTHON_BIN" scripts/p0a4_trials.py complete --phase official_full_student \
    --version "$trial_version" \
    --audit reports/audit/gate_p0a4_official_full_retention.json --seal-trace "$STUDENT_FULL_TRACE"
  return "$result"
}

student_full_direct() {
  P0A4_ALLOW_UNGATED_FULL=1 student_full
}

student_full_direct_gpu() {
  start_edge_server
  trap stop_edge_server EXIT INT TERM
  student_full_direct
}

checks() {
  "$PYTHON_BIN" -m unittest tests.test_p0a4_pipeline tests.test_p0a3_reselection tests.test_vllm_teacher_launcher
  "$PYTHON_BIN" scripts/p0a4_protocol.py preflight --config "$CONFIG" >/dev/null
  bash -n scripts/run_p0a4.sh
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p0a4.sh <command> [version]

Protocol and models:
  preflight                 Freeze/check full test and build disjoint train/96/170 data.
  build-llama-cuda          Rebuild llama.cpp for RTX 3090 CUDA compute capability 8.6.
  install-training-deps     Install only the P0-A4 LoRA/ZeRO-3 dependencies.
  download-teacher          Download Qwen2.5-14B BF16 for the distillation Teacher.

14B services and denominator:
  baseline-plan|baseline-serve
  baseline-dev              Build immutable 96/170 AWQ denominators.
  baseline-full             Run four official-full shards once, merge and seal.
  teacher-train [1..3]      Four-GPU ZeRO-3 BF16 LoRA training.
  teacher-plan|teacher-serve
  teacher-validate [1..3]   Evaluate on the separate Teacher validation set.
  teacher-select            Select the best complete Teacher validation result.
  teacher-distill [1..3]    Generate correctness-filtered train-only traces.

Student:
  student-v2-preflight      Freeze aggregate-only v2 sampling/Adapter plan and dry-run it.
  student-train [1..2]
  student-merge [1..2]
  student-check [1..2]      Verify training/merge/quantization provenance and show next stage.
  student-experts [1..2]    Optional task Adapter experts.
  prepare-adapters [1..2]   Optional llama.cpp Top-1 Adapter manifest.
  student-quantize [1..2]   Train-only imatrix plus versioned Q4_K_M; Q8 KV is fixed.
  edge-start|edge-start-adapters|edge-stop
  student-smoke96           Require every task and capped macro >=75%.
  student-170-check [1..2]  Verify Q4/96/memory/baseline/live service without using a trial.
  student-170 [1..2]        Aggregate-only feedback; require every task/macro >=80%.
  student-memory-precheck-q4 Q4 96-pass candidate, 5+20 requests, cache off, <=1400 MB.
  student-memory            20+100 requests and <=1400 MB Dev peak.
  student-adapter-smoke96   Optional Top-1 route on the same 96-row gate.
  student-adapter-memory-precheck [2]  All v2 Adapters resident, 5+20, <=1400 MB.
  student-v2-select-route   Require >=85% per task/macro and freeze shared vs Adapter.
  edge-start-v2-selected    Start the CUDA service for the frozen v2 route.
  student-v2-170            Consume the final 170-row slot with the frozen v2 route.
  student-v2-memory         Run the 20+100 memory gate for the frozen v2 route.
  student-adapter-170 [2]   Optional route consumes one of the two Student versions.
  student-adapter-memory    All adapters resident; each Top-1 expert active in turn.
  student-full              One official-full Student attempt, then permanent seal.
  student-full-direct       Explicitly skip 170/memory gates and consume the one full attempt.
  student-full-direct-gpu   Start managed CUDA edge service and resume the direct full attempt.
  checks

Start baseline-serve/teacher-serve/edge-start in a separate terminal as needed.
Formal test outputs are never accepted as training inputs. A failed full Student run
is final and cannot be reopened by this launcher.
EOF
}

case "${1:-}" in
  preflight) preflight ;;
  build-llama-cuda) build_llama_cuda ;;
  install-training-deps) install_training_deps ;;
  download-teacher) download_teacher ;;
  baseline-plan) baseline_plan ;;
  baseline-serve) baseline_serve ;;
  baseline-dev) baseline_dev ;;
  baseline-full) baseline_full ;;
  teacher-train) teacher_train "${2:-$P0A4_TEACHER_VERSION}" ;;
  teacher-plan) teacher_plan ;;
  teacher-serve) teacher_serve ;;
  teacher-validate) teacher_validate "${2:-$P0A4_TEACHER_VERSION}" ;;
  teacher-select) teacher_select ;;
  teacher-distill) teacher_distill "${2:-$P0A4_TEACHER_VERSION}" ;;
  student-train) student_train "${2:-$P0A4_STUDENT_VERSION}" ;;
  student-v2-preflight) student_v2_preflight ;;
  student-merge) student_merge "${2:-$P0A4_STUDENT_VERSION}" ;;
  student-check) student_check "${2:-$P0A4_STUDENT_VERSION}" ;;
  student-experts) student_experts "${2:-$P0A4_STUDENT_VERSION}" ;;
  prepare-adapters) prepare_adapters "${2:-$P0A4_STUDENT_VERSION}" ;;
  student-quantize) student_quantize "${2:-$P0A4_STUDENT_VERSION}" ;;
  edge-start) start_edge_server ;;
  edge-start-adapters) start_edge_server_adapters ;;
  edge-start-v2-selected) start_edge_server_v2_selected ;;
  edge-stop) stop_edge_server ;;
  student-smoke96) student_smoke96 ;;
  student-170-check) student_selection170_check "${2:-$P0A4_STUDENT_VERSION}" ;;
  student-170) student_selection170 "${2:-$P0A4_STUDENT_VERSION}" ;;
  student-memory-precheck-q4) student_memory_precheck_q4 ;;
  student-memory) student_memory ;;
  student-adapter-smoke96) student_adapter_smoke96 ;;
  student-adapter-memory-precheck) student_adapter_memory_precheck "${2:-$P0A4_STUDENT_VERSION}" ;;
  student-v2-select-route) student_v2_select_route ;;
  student-v2-170) student_v2_selection170 ;;
  student-v2-memory) student_v2_memory ;;
  student-adapter-170) student_adapter_selection170 "${2:-2}" ;;
  student-adapter-memory) student_adapter_memory "${2:-2}" ;;
  student-full) student_full ;;
  student-full-direct) student_full_direct ;;
  student-full-direct-gpu) student_full_direct_gpu ;;
  checks) checks ;;
  *) usage; exit 2 ;;
esac
