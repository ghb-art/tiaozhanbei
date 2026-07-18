#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-external/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
PY="${PY:-.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "Missing Python executable: $PY" >&2
  exit 2
fi

if ! command -v cmake >/dev/null 2>&1 && [ ! -x ".venv/bin/cmake" ]; then
  "$PY" -m pip install cmake
fi
if ! command -v ninja >/dev/null 2>&1 && [ ! -x ".venv/bin/ninja" ]; then
  "$PY" -m pip install ninja
fi

CMAKE_BIN="$(command -v cmake || true)"
NINJA_BIN="$(command -v ninja || true)"
CMAKE_BIN="${CMAKE_BIN:-$ROOT/.venv/bin/cmake}"
NINJA_BIN="${NINJA_BIN:-$ROOT/.venv/bin/ninja}"

mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi

git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_REF"
git -C "$LLAMA_CPP_DIR" checkout FETCH_HEAD

"$CMAKE_BIN" -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$NINJA_BIN" \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON

"$CMAKE_BIN" --build "$LLAMA_CPP_DIR/build" --config Release \
  --target llama-cli llama-server llama-quantize llama-imatrix

echo "llama.cpp ready: $LLAMA_CPP_DIR"
find "$LLAMA_CPP_DIR/build" -type f \( -name 'llama-cli' -o -name 'llama-server' -o -name 'llama-quantize' -o -name 'llama-imatrix' \) -print
