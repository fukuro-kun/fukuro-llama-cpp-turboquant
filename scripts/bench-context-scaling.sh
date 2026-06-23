#!/usr/bin/env bash
# Kontext-Scaling-Benchmark: Misst t/s und Speicherverbrauch fuer steigende Kontextlaengen.
# Verwendet llama-bench mit verschiedenen -c Werten.
#
# Nutzung:
#   MODEL=/pfad/zum/modell.gguf ./scripts/bench-context-scaling.sh
#   MODEL=/pfad/zum/modell.gguf CTX_LIST="1024 4096 8192" NGL=8 ./scripts/bench-context-scaling.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${LLAMA_BENCH:-${ROOT}/build/bin/llama-bench}"

MODEL="${MODEL:-}"
NGL="${NGL:-99}"
# Sweet spot: turbo4 fuer Keys (empfindlicher), turbo3 fuer Values (robuster)
CTK="${CTK:-turbo4}"
CTV="${CTV:-turbo3}"
N_PROMPT="${N_PROMPT:-512}"
N_GEN="${N_GEN:-128}"
FA="${FA:-on}"
REPS="${REPS:-3}"

# Kontextlaengen: logarithmisch von 1k bis 100k
CTX_LIST="${CTX_LIST:-1024 2048 4096 8192 16384 32768 65536 100000 131072}"

OUTPUT="${OUTPUT:-${ROOT}/bench-context-scaling-$(date +%Y%m%d-%H%M%S).csv}"

if [[ -z "$MODEL" ]]; then
  echo "error: MODEL nicht gesetzt" >&2
  echo "usage: MODEL=/pfad/zum/modell.gguf $0" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: Modell nicht gefunden: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "$BENCH" ]]; then
  echo "error: ${BENCH} nicht gefunden (build mit: cmake --build build --target llama-bench)" >&2
  exit 1
fi

echo "info: Benchmark startet — Ergebnisse in ${OUTPUT}" >&2
echo "info: Modell: ${MODEL}" >&2
echo "info: NGL=${NGL} CTK=${CTK} FA=${FA} REPS=${REPS}" >&2
echo "info: n_prompt=${N_PROMPT} n_gen=${N_GEN}" >&2

# CSV-Header
printf "ctx_len,pp_ts,tg_ts,vram_mb_used,ram_mb_used,status,notes\n" > "$OUTPUT"

for CTX in $CTX_LIST; do
  echo "" >&2
  echo "=== Kontext: ${CTX} ===" >&2

  # VRAM vor dem Benchmark (in MB)
  VRAM_BEFORE=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  fi

  # RAM vor dem Benchmark (in kB, /proc/self/status)
  RAM_BEFORE=0
  if [[ -f /proc/self/status ]]; then
    RAM_BEFORE=$(grep VmRSS /proc/self/status 2>/dev/null | awk '{print $2}' || echo 0)
  fi

  STATUS="ok"
  NOTES=""
  PP_TS=""
  TG_TS=""
  VRAM_USED=""
  RAM_USED=""

  # llama-bench ausfuehren
  set +e
  BENCH_OUT=$("$BENCH" \
    -m "$MODEL" \
    -c "$CTX" \
    -ngl "$NGL" \
    -ctk "$CTK" \
    -ctv "$CTV" \
    -fa "$FA" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -r "$REPS" \
    2>&1)
  BENCH_RC=$?
  set -e

  if [[ $BENCH_RC -ne 0 ]]; then
    STATUS="oom"
    NOTES="llama-bench exit code ${BENCH_RC}"
    echo "warn: Kontext ${CTX} fehlgeschlagen (OOM?) — rc=${BENCH_RC}" >&2
    echo "  output: ${BENCH_OUT:0:200}" >&2
  else
    # t/s extrahieren
    # llama-bench output: z.B. "| pp512     |  123.45 |  456.78 | ..."
    PP_LINE=$(echo "$BENCH_OUT" | grep -E "^\| pp[0-9]+" | head -1)
    TG_LINE=$(echo "$BENCH_OUT" | grep -E "^\| tg[0-9]+" | head -1)

    if [[ -n "$PP_LINE" ]]; then
      PP_TS=$(echo "$PP_LINE" | awk -F'|' '{print $3}' | tr -d ' ')
    fi
    if [[ -n "$TG_LINE" ]]; then
      TG_TS=$(echo "$TG_LINE" | awk -F'|' '{print $3}' | tr -d ' ')
    fi

    # VRAM nach dem Benchmark
    if command -v nvidia-smi >/dev/null 2>&1; then
      VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
      VRAM_USED=$((VRAM_AFTER - VRAM_BEFORE))
    fi

    # RAM nach dem Benchmark
    if [[ -f /proc/self/status ]]; then
      RAM_AFTER=$(grep VmRSS /proc/self/status 2>/dev/null | awk '{print $2}' || echo 0)
      RAM_USED=$((RAM_AFTER - RAM_BEFORE))
    fi

    echo "  pp ${N_PROMPT}: ${PP_TS:-N/A} t/s" >&2
    echo "  tg ${N_GEN}: ${TG_TS:-N/A} t/s" >&2
    echo "  VRAM delta: ${VRAM_USED:-N/A} MB" >&2
    echo "  RAM delta: ${RAM_USED:-N/A} kB" >&2
  fi

  # CSV-Zeile schreiben
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$CTX" \
    "${PP_TS:-}" \
    "${TG_TS:-}" \
    "${VRAM_USED:-}" \
    "${RAM_USED:-}" \
    "$STATUS" \
    "$NOTES" >> "$OUTPUT"

done

echo "" >&2
echo "=== Ergebnisse ===" >&2
cat "$OUTPUT" | column -t -s, >&2
echo "" >&2
echo "info: CSV gespeichert in ${OUTPUT}" >&2
