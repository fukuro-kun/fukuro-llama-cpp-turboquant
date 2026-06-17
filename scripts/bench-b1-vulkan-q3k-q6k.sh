#!/usr/bin/env bash
# B1 Benchmark: Vulkan Q3_K/Q6_K Block-Load Performance
# Vergleicht tg128 fuer Gemma 4 Modelle in Q3_K und Q6_K auf AMD-Vulkan.
#
# Nutzung:
#   MODEL_DIR=/pfad/zu/modellen ./scripts/bench-b1-vulkan-q3k-q6k.sh
#   MODEL_DIR=/jade/models/unsloth NGL=99 ./scripts/bench-b1-vulkan-q3k-q6k.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${LLAMA_BENCH:-${ROOT}/build/bin/llama-bench}"

MODEL_DIR="${MODEL_DIR:-}"
NGL="${NGL:-99}"
N_PROMPT="${N_PROMPT:-512}"
N_GEN="${N_GEN:-128}"
REPS="${REPS:-3}"
FA="${FA:-on}"

# Modelle fuer B1-Test
MODELS=(
  "gemma-4-12b-it-Q3_K_M.gguf"
  "gemma-4-12b-it-Q6_K.gguf"
  "gemma-4-31B-it-Q3_K_M.gguf"
  "gemma-4-31B-it-Q6_K.gguf"
  "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
  "gemma-4-26B-A4B-it-UD-Q6_K.gguf"
)

OUTPUT="${OUTPUT:-${ROOT}/bench-b1-vulkan-$(date +%Y%m%d-%H%M%S).csv}"

if [[ -z "$MODEL_DIR" ]]; then
  echo "error: MODEL_DIR nicht gesetzt" >&2
  echo "usage: MODEL_DIR=/pfad/zu/modellen $0" >&2
  exit 1
fi

if [[ ! -f "$BENCH" ]]; then
  echo "error: ${BENCH} nicht gefunden (build mit: cmake --build build --target llama-bench)" >&2
  exit 1
fi

# Pruefe ob llama-bench Vulkan unterstuetzt
if ! "$BENCH" --help 2>&1 | grep -q 'vulkan'; then
  echo "warn: llama-bench ohne Vulkan-Flag kompiliert — GGML_VULKAN=ON noetig" >&2
fi

# GPU-Info (AMD)
echo "info: GPU-Info:" >&2
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverVersion' | head -4 >&2 || true
elif command -v rocminfo >/dev/null 2>&1; then
  rocminfo 2>/dev/null | grep -E 'Marketing Name' | head -2 >&2 || true
else
  echo "  (keine GPU-Info verfuegbar)" >&2
fi

echo "info: B1 Benchmark startet — Ergebnisse in ${OUTPUT}" >&2
echo "info: NGL=${NGL} FA=${FA} REPS=${REPS}" >&2
echo "info: n_prompt=${N_PROMPT} n_gen=${N_GEN}" >&2

# CSV-Header
printf "model,quant,pp_ts,tg_ts,status,notes\n" > "$OUTPUT"

for MODEL_FILE in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "warn: Modell nicht gefunden: ${MODEL_PATH} — ueberspringe" >&2
    printf "%s,%s,,,missing,Modell nicht gefunden\n" "$MODEL_FILE" "${MODEL_FILE##*.}" >> "$OUTPUT"
    continue
  fi

  QUANT="${MODEL_FILE##*.}"
  QUANT="${QUANT%.gguf}"

  echo "" >&2
  echo "=== Modell: ${MODEL_FILE} ===" >&2

  STATUS="ok"
  NOTES=""
  PP_TS=""
  TG_TS=""

  set +e
  BENCH_OUT=$("$BENCH" \
    -m "$MODEL_PATH" \
    -ngl "$NGL" \
    -fa "$FA" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -r "$REPS" \
    2>&1)
  BENCH_RC=$?
  set -e

  if [[ $BENCH_RC -ne 0 ]]; then
    STATUS="error"
    NOTES="exit code ${BENCH_RC}"
    echo "warn: Benchmark fehlgeschlagen — rc=${BENCH_RC}" >&2
    echo "  output: ${BENCH_OUT:0:200}" >&2
  else
    # t/s extrahieren (llama-bench Output-Format)
    PP_LINE=$(echo "$BENCH_OUT" | grep -E "^\\| pp[0-9]+" | head -1)
    TG_LINE=$(echo "$BENCH_OUT" | grep -E "^\\| tg[0-9]+" | head -1)

    if [[ -n "$PP_LINE" ]]; then
      PP_TS=$(echo "$PP_LINE" | awk -F'|' '{print $3}' | tr -d ' ')
    fi
    if [[ -n "$TG_LINE" ]]; then
      TG_TS=$(echo "$TG_LINE" | awk -F'|' '{print $3}' | tr -d ' ')
    fi

    echo "  pp ${N_PROMPT}: ${PP_TS:-N/A} t/s" >&2
    echo "  tg ${N_GEN}: ${TG_TS:-N/A} t/s" >&2
  fi

  printf "%s,%s,%s,%s,%s,%s\n" \
    "$MODEL_FILE" \
    "$QUANT" \
    "${PP_TS:-}" \
    "${TG_TS:-}" \
    "$STATUS" \
    "$NOTES" >> "$OUTPUT"

done

echo "" >&2
echo "=== Ergebnisse ===" >&2
cat "$OUTPUT" | column -t -s, >&2
echo "" >&2
echo "info: CSV gespeichert in ${OUTPUT}" >&2
