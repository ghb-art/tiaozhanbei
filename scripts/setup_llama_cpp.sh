#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-external/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-2d973636e292ee6f75fadcf08d29cb33511f509f}"
LLAMA_CPP_CUDA="${LLAMA_CPP_CUDA:-ON}"
LLAMA_CPP_CUDA_ARCHITECTURES="${LLAMA_CPP_CUDA_ARCHITECTURES:-86}"
PY="${PY:-.venv/bin/python}"
LLAMA_CPP_UTF8_PATCH="${LLAMA_CPP_UTF8_PATCH:-$ROOT/patches/llama_cpp_chat_utf8_sanitize.patch}"

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

if git -C "$LLAMA_CPP_DIR" apply --reverse --check "$LLAMA_CPP_UTF8_PATCH" >/dev/null 2>&1; then
  echo "llama.cpp UTF-8 parser patch already applied"
elif git -C "$LLAMA_CPP_DIR" apply --check "$LLAMA_CPP_UTF8_PATCH"; then
  git -C "$LLAMA_CPP_DIR" apply "$LLAMA_CPP_UTF8_PATCH"
  echo "Applied llama.cpp UTF-8 parser patch"
else
  echo "llama.cpp UTF-8 parser patch is incompatible with $LLAMA_CPP_REF" >&2
  exit 1
fi

cmake_args=(
  -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"
  -G Ninja
  -DCMAKE_MAKE_PROGRAM="$NINJA_BIN"
  -DGGML_CUDA="$LLAMA_CPP_CUDA"
  -DGGML_NATIVE=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=ON
)
if [[ "$LLAMA_CPP_CUDA" == "ON" ]]; then
  cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$LLAMA_CPP_CUDA_ARCHITECTURES")
fi
"$CMAKE_BIN" "${cmake_args[@]}"

"$CMAKE_BIN" --build "$LLAMA_CPP_DIR/build" --config Release \
  --target llama-cli llama-server llama-quantize llama-imatrix

echo "llama.cpp ready: $LLAMA_CPP_DIR"
find "$LLAMA_CPP_DIR/build" -type f \( -name 'llama-cli' -o -name 'llama-server' -o -name 'llama-quantize' -o -name 'llama-imatrix' \) -print
