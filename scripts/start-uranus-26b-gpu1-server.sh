#!/usr/bin/env bash
# Uranus llama-server GPU 1 für InferenzQuelle Router.
# Gemma-4 26B-A4B QAT, turbo4/3 KV-Cache.
# 2. Instanz auf GPU 1, Port 18082.
#
# Rasch auf-/abbaubar:
#   Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-uranus-26b-gpu1-server.sh
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-gpu1-server.sh
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ KONFIGURATIONS-VARIANTEN (Benchmark 2026-08-15, Trilium ppMetN3aSi09) │
# └─────────────────────────────────────────────────────────────────────┘
#
# Zwei Varianten für 2× RTX 4060 Ti (je 16 GB VRAM), 8-Kern CPU:
#
#   A) moe0 / 96k (Default, produktiv seit 2026-08-15)
#      --n-cpu-moe 0  -c 196608  →  2 Slots × 96k
#      Alle MoE-Experten auf GPU → maximale Generierungsgeschwindigkeit.
#      82,5 t/s solo, 63,7 t/s parallel, 46,4 t/s bei 37k Prompt.
#      2,6-7,7× schneller als Variante B. 167 MiB VRAM-Reserve — knapp.
#      cache-reuse 1 (konservativ, KV-shift für nicht-prefix Chunks).
#
#   B) moe5 / 128k (Legacy, vor 2026-08-15)
#      --n-cpu-moe 5  -c 262144  →  2 Slots × 128k
#      5 MoE-Layer auf CPU → mehr VRAM für KV-Cache → größerer Kontext.
#      ~12-30 t/s (deutlich langsamer durch CPU-MoE-Roundtrips).
#      cache-reuse 1 (konservativ, RAG/Tool-Defs ohne Prefix-Overlap).
#      Override: MOE=5 CTX=262144 CACHE_RAM=16384
#
#   Default = A (moe0/96k), identisch mit GPU 0 (start-uranus-26b-server.sh).
#   Override via Env-Vars: MOE=5 CTX=262144 bash start-uranus-26b-gpu1-server.sh
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ CPU-KONKURRENZ BEI 2 INSTANZEN (erfahren 2026-07-18)                 │
# └─────────────────────────────────────────────────────────────────────┘
#
#   2 unabhängige llama-server Prozesse auf EINER CPU überlasten sich
#   gegenseitig, selbst wenn nur 1 Instanz aktiv generiert! Die idle
#   Instanz verbraucht CPU für MoE-Prefetch, Cache-Updates, Slot-Management.
#   - 2 Instanzen, 1 Request aktiv: 3.5 t/s (CPU 100%, load 15.5)
#   - 1 Instanz, 1 Request aktiv:   45 t/s (CPU 11%, load 1.0)
#
#   Mitigation in diesem Skript:
#   - GGML_SCHED_PREFETCH_EXPERTS=0 — kein MoE-Prefetch wenn idle
#   - GGML_SCHED_PREFETCH_SLOTS=0   — kein Slot-Prefetch wenn idle
#   Diese Env-Vars reduzieren die idle-CPU-Last der zweiten Instanz.
#   Bei aktiver Generierung auf BEIDEN Instanzen gleichzeitig bleibt
#   CPU-Konkurrenz trotzdem — das ist ein Hardware-Limit (8 Kerne).
#
# Architecture: 2. llama-server Instanz auf GPU 1 (physisch).
#   CUDA_VISIBLE_DEVICES=1 isoliert den Prozess auf GPU 1.
#   Port 18082, 2 Slots × 96k Kontext (Default), --n-cpu-moe 0.
#   Siehe start-uranus-26b-server.sh für GPU 0 (Port 18080).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-/media/fukuro/raid5/modelle/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18082}"
CTX="${CTX:-196608}"             # 192k total → 2×96k (Variante A, Default)
SLOTS="${SLOTS:-2}"
CACHE_RAM="${CACHE_RAM:-32768}"  # 32 GB CPU prompt-cache (gleich wie GPU 0)
MOE="${MOE:-0}"                  # 0 = alle Experten auf GPU (Variante A, Default)
CACHE_REUSE="${CACHE_REUSE:-1}"   # 1 = konservativ (KV-shift für nicht-prefix Chunks)

# --- Modell-Check ---
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: $MAIN" >&2; exit 1
fi

# --- Port-Check ---
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# --- GPU-Isolation ---
# Physische GPU 1 isolieren → kein CUDA-Context auf GPU 0.
export CUDA_VISIBLE_DEVICES=1

# --- CPU-Konkurrenz-Mitigation ---
# MoE-Prefetch und Slot-Prefetch DEAKTIVIERT für 2. Instanz.
# Reduziert idle-CPU-Last wenn diese Instanz nicht aktiv generiert.
# Die erste Instanz (GPU 0) behält GGML_SCHED_PREFETCH_EXPERTS=1.
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-0}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-0}"
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"

cd "$ROOT"

# --- Instanz: GPU 1, Port 18082 ---
echo "Starte llama-server (GPU 1, Port $PORT, $SLOTS Slots × $((CTX/SLOTS/1024))k, --n-cpu-moe $MOE, cache-reuse $CACHE_REUSE)..."
setsid "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT" \
  -dev CUDA0 \
  -c "$CTX" -ngl 999 --n-cpu-moe "$MOE" \
  -ctk turbo4 -ctv turbo3 -fa on \
  --parallel "$SLOTS" -np "$SLOTS" --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram "$CACHE_RAM" \
  --cache-reuse "$CACHE_REUSE" \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > /tmp/uranus-26b-gpu1.log 2>&1 &
PID=$!
echo "  PID: $PID, Log: /tmp/uranus-26b-gpu1.log"

# --- Health-Check (warten bis ready) ---
echo
echo "Warte auf Health-Check (max 120s)..."
READY=0
for i in $(seq 1 120); do
  H=$(curl -s --connect-timeout 2 http://127.0.0.1:$PORT/health 2>/dev/null || echo "")
  if [[ "$H" == *"ok"* ]]; then
    READY=1
    echo "  Healthy nach ${i}s"
    break
  fi
  sleep 1
done

if [[ "$READY" == "0" ]]; then
  echo "WARNUNG: Health-Check nicht bestanden nach 120s"
  echo "  ($PORT): ${H:-keine Antwort}"
  echo "  Log: /tmp/uranus-26b-gpu1.log"
  echo "  PID: $PID (noch laufend)"
  exit 1
fi

# --- PID speichern für Stop-Skript ---
echo "$PID" > /tmp/uranus-26b-gpu1.pid

echo
echo "=== Uranus 26B-A4B Server GPU 1 gestartet ==="
echo "  http://0.0.0.0:$PORT  (GPU 1, $SLOTS Slots × $((CTX/SLOTS/1024))k, --n-cpu-moe $MOE)  PID $PID"
echo
echo "Stop: bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-gpu1-server.sh"
echo "Log:  tail -f /tmp/uranus-26b-gpu1.log"
