#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup_datasets.sh [--check-only] [--make-dirs]

Checks the local G-DATA dataset layout expected by this project.
This script does not download licensed datasets or accept dataset terms.

Options:
  --check-only  Check expected payload paths. This is the default.
  --make-dirs   Create the expected dataset directories before checking.
USAGE
}

MODE="check"
MAKE_DIRS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      MODE="check"
      shift
      ;;
    --make-dirs)
      MAKE_DIRS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

DATASETS_ROOT="${ROOT}/data/datasets"

DATASET_DIRS=(
  "gsm8k"
  "humaneval"
  "cmmlu"
  "opencodeinstruct"
  "coig_cqia"
  "mvtec_ad"
  "neu_det"
  "cityflow"
  "ua_detrac"
)

EXPECTED_PATHS=(
  "gsm8k/grade_school_math/data/train.jsonl"
  "gsm8k/grade_school_math/data/test.jsonl"
  "humaneval/data/HumanEval.jsonl.gz"
  "cmmlu/data/test"
  "opencodeinstruct/data/train-00000-of-00050.parquet"
  "opencodeinstruct/data/train-00017-of-00050.parquet"
  "opencodeinstruct/data/train-00033-of-00050.parquet"
  "coig_cqia/COIG-CQIA-full.jsonl"
  "mvtec_ad/mvtec_anomaly_detection"
  "neu_det/NEU-DET"
  "cityflow/AICity22_Track1_MTMC_Tracking"
  "ua_detrac/ua_detrac_kaggle_archive"
)

if [[ "${MAKE_DIRS}" -eq 1 ]]; then
  mkdir -p "${DATASETS_ROOT}"
  for dir in "${DATASET_DIRS[@]}"; do
    mkdir -p "${DATASETS_ROOT}/${dir}"
  done
fi

echo "Dataset root: ${DATASETS_ROOT}"

missing=0
for relative_path in "${EXPECTED_PATHS[@]}"; do
  full_path="${DATASETS_ROOT}/${relative_path}"
  if [[ -e "${full_path}" ]]; then
    echo "[OK] ${relative_path}"
  else
    echo "[MISSING] ${relative_path}"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Dataset setup check failed. Place or extract the missing payloads, then rerun this script." >&2
  exit 1
fi

echo "Dataset setup check passed."
