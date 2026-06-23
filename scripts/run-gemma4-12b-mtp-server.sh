#!/usr/bin/env bash
# Gemma 4 12B dense target + gemma4_assistant MTP draft (dense tied LM head).
# Gemma 4 12B uses Gemma4UnifiedAssistantForCausalLM.
# Override MAIN_GGUF to your converted 12B target GGUF.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${ROOT}/.scratch/gemma-4-12b/gemma-4-12B-it-Q4_K_M.gguf}"
_DRAFT_Q4="${ROOT}/.scratch/gemma-4-12b-assistant-mtp-Q4_K_M.gguf"
_DRAFT_F16="${ROOT}/.scratch/gemma-4-12b-assistant-mtp.gguf"
if [[ -n "${DRAFT_GGUF:-}" ]]; then
  DRAFT="$DRAFT_GGUF"
elif [[ -f "$_DRAFT_Q4" ]]; then
  DRAFT="$_DRAFT_Q4"
else
  DRAFT="$_DRAFT_F16"
fi

VERIFY_ASSISTANT_GGUF="${VERIFY_ASSISTANT_GGUF:-1}"

CTX="${CTX:-8192}"
NGL="${NGL:-99}"
NGL_DRAFT="${NGL_DRAFT:-99}"
# Sweet spot: turbo4 fuer Keys (empfindlicher), turbo3 fuer Values (robuster)
CTK="${CTK:-turbo4}"
CTV="${CTV:-turbo3}"
CTKD="${CTKD:-turbo4}"
CTVD="${CTVD:-turbo3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
FA="${FA:-on}"
SPEC="${SPEC:-mtp}"
TEMP="${TEMP:-}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
ENABLE_SLOTS="${ENABLE_SLOTS:-1}"
LOG_TIMESTAMPS="${LOG_TIMESTAMPS:-1}"
LOG_PREFIX="${LOG_PREFIX:-1}"
NO_WARMUP="${NO_WARMUP:-0}"
PARALLEL="${PARALLEL:-1}"
DRAFT_BLOCK_SIZE="${DRAFT_BLOCK_SIZE:-2}"
DRAFT_MAX="${DRAFT_MAX:-6}"

if [[ ! -f "$SERVER" ]]; then
  echo "error: missing ${SERVER} (build with: cmake --build build --target llama-server)" >&2
  exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: ${MAIN}" >&2
  echo "hint: convert google/gemma-4-12B-it into ${MAIN} or set MAIN_GGUF" >&2
  exit 1
fi
if [[ "$SPEC" == "mtp" ]]; then
  if [[ ! -f "$DRAFT" ]]; then
    echo "error: draft (assistant) GGUF not found: ${DRAFT}" >&2
    echo "hint: hf download google/gemma-4-12B-it-assistant --local-dir ${ROOT}/.scratch/gemma-4-12B-it-assistant" >&2
    echo "hint: PYTHONPATH=${ROOT}/gguf-py python3 ${ROOT}/convert_hf_to_gguf.py ${ROOT}/.scratch/gemma-4-12B-it-assistant --outfile ${_DRAFT_F16} --outtype f16" >&2
    exit 1
  fi
  if [[ "$VERIFY_ASSISTANT_GGUF" != "0" ]]; then
    if ! python3 "${ROOT}/scripts/verify-gemma4-assistant-gguf.py" "$DRAFT"; then
      echo "error: assistant GGUF verification failed" >&2
      exit 1
    fi
  fi
fi

ARGS=(
  -m "$MAIN"
  -c "$CTX"
  -ngl "$NGL"
  -ngld "$NGL_DRAFT"
  -ctk "$CTK"
  -ctv "$CTV"
  -ctkd "$CTKD"
  -ctvd "$CTVD"
  -fa "$FA"
  --host "$HOST"
  --port "$PORT"
  --parallel "$PARALLEL"
  -np "$PARALLEL"
  --cont-batching
)

if [[ "$SPEC" == "mtp" ]]; then
  ARGS+=(
    --mtp-head "$DRAFT"
    --spec-type mtp
    --draft-block-size "${DRAFT_BLOCK_SIZE}"
    --draft-max "${DRAFT_MAX}"
    --draft-min "${DRAFT_MIN:-0}"
  )
else
  echo "info: speculative decoding disabled (SPEC=${SPEC}); running baseline" >&2
fi

if [[ -n "$TEMP" ]]; then
  ARGS+=(--temp "$TEMP")
fi

[[ "$ENABLE_METRICS"  != "0" ]] && ARGS+=(--metrics)
[[ "$ENABLE_SLOTS"    != "0" ]] && ARGS+=(--slots)
[[ "$LOG_TIMESTAMPS"  != "0" ]] && ARGS+=(--log-timestamps)
[[ "$LOG_PREFIX"      != "0" ]] && ARGS+=(--log-prefix)
[[ "$NO_WARMUP"       != "0" ]] && ARGS+=(--no-warmup)

echo "info: SPEC=${SPEC} DRAFT_BLOCK_SIZE=${DRAFT_BLOCK_SIZE} DRAFT_MAX=${DRAFT_MAX}" >&2
echo "info: CTX=${CTX} NGL=${NGL} FA=${FA} CTK=${CTK} CTKD=${CTKD}" >&2
echo "info: MAIN=${MAIN}" >&2
echo "info: DRAFT=${DRAFT}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
