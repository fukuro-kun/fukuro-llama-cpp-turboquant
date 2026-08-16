#!/usr/bin/env bash
# Uranus llama-server für InferenzQuelle Router.
# Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive IQ4_XS.
# 1 Instanz auf GPU 0. GPU 1 frei für VLM/FLUX.
#
# Zwei Profile (umschaltbar via THINKING env var):
#   THINKING=0 (Default): 2 Slots × 64k, Non-Thinking, max. Durchsatz
#   THINKING=1:           1 Slot  × 128k, Thinking-Modus, max. Qualität
#
# MTP (optional, via MTP env var):
#   MTP=0 (Default): kein MTP (sicher, ungetestet auf dieser Hardware)
#   MTP=1: nativer MTP mit --draft-max 4 (spekulativer Speedup)
#
# Rasch auf-/abbaubar:
#   Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-uranus-qwen35b-server.sh
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh  (gleicher Port, gleiches PID-File)
#
# Architektur: Qwen3.6-35B-A3B ist ein MoE mit 35B total, nur ~3B aktiv pro Token.
#   256 Experten, 8 routed + 1 shared pro Token, 40 Layer, 262K nativer Kontext.
#   Hybrid-Attention 3:1: 30 Layer Gated DeltaNet (rekurrent, kein KV-Cache)
#   + 10 Layer GQA (klassischer KV-Cache). Dadurch wächst KV nur auf ¼ der Layer.
#   IQ4_XS = 18.7 GB, ~4.5 GB größer als Gemma-4 26B-A4B QAT (14.2 GB).
#
# WICHTIG — KV-Cache-Typen bei Qwen3 MoE:
#   turbo3/turbo2 sind bei Qwen3 MoE KILLED (NaN-Divergenz, Commit 4456735).
#   Ursache: MoE-Roundoff akkumuliert über rekurrente DeltaNet-State-Updates.
#   turbo4 ist sicher, q8_0 ist der konservative Default.
#   Wir nutzen q8_0/q8_0 (sicher, +0.01% PPL vs f16).
#   NICHT turbo3/turbo4 wie bei Gemma-4!
#
# Sampling — Qwen3.6 nutzt presence_penalty (NICHT repetition_penalty):
#   Non-Thinking: temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5
#   Thinking:     temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5
#
# Cache-Konfiguration (wie 26B, angepasst an 128 GB RAM + 16 GB VRAM):
#   --cache-ram 32768        32 GB CPU-RAM für serialisierte KV-States
#   --cache-reuse 256        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key automatisch)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-/media/fukuro/raid5/modelle/Qwen/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
CACHE_RAM="${CACHE_RAM:-32768}"  # 32 GB CPU prompt-cache (128 GB RAM available)

# --- Profile ---
THINKING="${THINKING:-0}"        # 0=Non-Thinking (2×64k), 1=Thinking (1×128k)
MTP="${MTP:-0}"                  # 0=kein MTP (Default), 1=nativer MTP mit --draft-max 4

# --- Modell-Check ---
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: $MAIN" >&2
  echo "  Erwartet auf uranus: /media/fukuro/raid5/modelle/Qwen/Qwen3.6-35B-A3B/" >&2
  exit 1
fi

# --- Port-Check ---
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# --- VRAM-Check + adaptive Konfiguration ---
# GPU0 freier VRAM bestimmt MoE-Offload. Qwen IQ4_XS = 18.7 GB → mehr Offload nötig
# als Gemma-4 26B QAT (14.2 GB). 256 Experten auf 40 Layer → ~375 MB/Layer Expert-Gewichte.
#
# Profil Non-Thinking (THINKING=0): 2 Slots × 64k, CTX=131072
#   KV q8_0 pro Slot bei 64k: ~700 MB → 2×700 MB = 1.4 GB total
#   ab 15 GB → moe=10, 2×64k  (~15 GB GPU, ~1 GB Reserve — knapp)
#   ab 12 GB → moe=15, 2×64k  (~13 GB GPU, ~3 GB Reserve — sicher)
#   ab  9 GB → moe=20, 2×64k  (~11 GB GPU, ~5 GB Reserve)
#   ab  6 GB → moe=25, 1×64k  (VRAM-Knappheit: 1 Slot)
#   <   6 GB → Fehler
#
# Profil Thinking (THINKING=1): 1 Slot × 128k, CTX=131072
#   KV q8_0 bei 128k: ~2.8 GB (nur 10 Full-Attn-Layer!)
#   HauhauCS empfiehlt 128k Kontext minimum für Thinking-Qualität.
#   ab 15 GB → moe=15, 1×128k (~13 GB GPU + 2.8 GB KV = ~15.8 GB — sehr knapp)
#   ab 12 GB → moe=20, 1×128k (~11 GB GPU + 2.8 GB KV = ~13.8 GB — sicher)
#   ab  9 GB → moe=25, 1×128k (~9 GB GPU + 2.8 GB KV = ~11.8 GB)
#   <   9 GB → Fehler (Thinking braucht mehr VRAM)
ADAPTIVE=${ADAPTIVE:-1}
if [[ "$ADAPTIVE" == "1" ]]; then
  VRAM_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')
  if [[ -z "$VRAM_FREE_MIB" ]]; then
    echo "WARNUNG: nvidia-smi nicht verfügbar, überspringe VRAM-Check" >&2
    # Fallback-Defaults
    if [[ "$THINKING" == "1" ]]; then
      SLOTS=1; MOE=20; CTX=131072
    else
      SLOTS=2; MOE=15; CTX=131072
    fi
  else
    VRAM_FREE_GIB=$((VRAM_FREE_MIB / 1024))
    echo "VRAM-Check: GPU0 hat ${VRAM_FREE_GIB} GB frei (Profil: THINKING=$THINKING)"
    if [[ "$THINKING" == "1" ]]; then
      # Thinking-Profil: 1×128k
      if   (( VRAM_FREE_GIB >= 15 )); then
        SLOTS=1; MOE=15; CTX=131072    # 1×128k, aggressiv (sehr knapp)
      elif (( VRAM_FREE_GIB >= 12 )); then
        SLOTS=1; MOE=20; CTX=131072    # 1×128k, sicher
      elif (( VRAM_FREE_GIB >= 9  )); then
        SLOTS=1; MOE=25; CTX=131072    # 1×128k, max Offload
      else
        echo "error: nur ${VRAM_FREE_GIB} GB VRAM frei (Thinking braucht min. 9 GB)" >&2
        echo "  Belegung:" >&2
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
        echo "  Lösung: THINKING=0 verwenden oder GPU freigeben" >&2
        exit 2
      fi
    else
      # Non-Thinking-Profil: 2×64k
      if   (( VRAM_FREE_GIB >= 15 )); then
        SLOTS=2; MOE=10; CTX=131072    # 2×64k, aggressiv (knapp)
      elif (( VRAM_FREE_GIB >= 12 )); then
        SLOTS=2; MOE=15; CTX=131072    # 2×64k, sicher (Default)
      elif (( VRAM_FREE_GIB >= 9  )); then
        SLOTS=2; MOE=20; CTX=131072    # 2×64k, mehr Offload
      elif (( VRAM_FREE_GIB >= 6  )); then
        SLOTS=1; MOE=25; CTX=65536     # 1×64k, VRAM-Knappheit
      else
        echo "error: nur ${VRAM_FREE_GIB} GB VRAM frei (mindestens 6 GB nötig)" >&2
        echo "  Belegung:" >&2
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
        echo "  Lösung: Training/VLM/FLUX stoppen oder mehr MoE-Offload (MOE=25) manuell setzen" >&2
        exit 2
      fi
    fi
    echo "  → Adaptiv: SLOTS=$SLOTS, MOE=$MOE, CTX=$CTX ($((CTX/1024))k), THINKING=$THINKING"
  fi
else
  # Manuelle Defaults
  if [[ "$THINKING" == "1" ]]; then
    SLOTS="${SLOTS:-1}"; MOE="${MOE:-20}"; CTX="${CTX:-131072}"
  else
    SLOTS="${SLOTS:-2}"; MOE="${MOE:-15}"; CTX="${CTX:-131072}"
  fi
fi

# --- GPU-Isolation ---
export CUDA_VISIBLE_DEVICES=0

# --- thecodacus MoE-Optimierungen ---
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"

cd "$ROOT"

# --- Sampling + Reasoning-Flags nach Profil ---
if [[ "$THINKING" == "1" ]]; then
  # Thinking-Profil: temp 1.0, top_p 0.95, Thinking an
  SAMPLING_FLAGS=(
    --temp 1.0 --top-p 0.95 --top-k 20
    --presence-penalty 1.5 --repeat-penalty 1.0
  )
  REASONING_FLAGS=(
    --reasoning-budget -1
    --jinja
  )
  PROFILE_NAME="Thinking 1×$((CTX/1024))k"
else
  # Non-Thinking-Profil: temp 0.7, top_p 0.80, Thinking aus
  SAMPLING_FLAGS=(
    --temp 0.7 --top-p 0.80 --top-k 20
    --presence-penalty 1.5 --repeat-penalty 1.0
  )
  REASONING_FLAGS=(
    --reasoning-budget 0
    --jinja
    --chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}'
  )
  PROFILE_NAME="Non-Thinking ${SLOTS}×$((CTX/SLOTS/1024))k"
fi

# --- MTP-Flags ---
MTP_FLAGS=()
if [[ "$MTP" == "1" ]]; then
  # Nativer MTP mit --spec-draft-n-max 4 (Default 16 → OOM auf 16GB, Issue #1768)
  MTP_FLAGS=(--spec-draft-n-max 4)
  PROFILE_NAME="${PROFILE_NAME} +MTP"
fi

# --- Instanz: GPU 0, Port 18080 ---
echo "Starte llama-server (Qwen3.6-35B-A3B, GPU 0, Port $PORT, $PROFILE_NAME, --n-cpu-moe $MOE)..."
setsid "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT" \
  -dev CUDA0 \
  -c "$CTX" -ngl 999 --n-cpu-moe "$MOE" \
  -ctk q8_0 -ctv q8_0 -fa on \
  --parallel "$SLOTS" -np "$SLOTS" --cont-batching \
  "${SAMPLING_FLAGS[@]}" \
  "${REASONING_FLAGS[@]}" \
  "${MTP_FLAGS[@]}" \
  --no-context-shift \
  --no-mmap --mlock \
  --cache-ram "$CACHE_RAM" \
  --cache-reuse 256 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > /tmp/uranus-qwen35b-gpu0.log 2>&1 &
PID=$!
echo "  PID: $PID, Log: /tmp/uranus-qwen35b-gpu0.log"

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
  echo "  Log: /tmp/uranus-qwen35b-gpu0.log"
  echo "  PID: $PID (noch laufend)"
  exit 1
fi

# --- PID speichern für Stop-Skript (gleiches PID-File wie 26B) ---
echo "$PID" > /tmp/uranus-26b-gpu0.pid

echo
echo "=== Uranus Qwen3.6-35B-A3B Server gestartet ==="
echo "  http://0.0.0.0:$PORT  ($PROFILE_NAME, --n-cpu-moe $MOE)  PID $PID"
echo
echo "Stop: bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh"
echo "Log:  tail -f /tmp/uranus-qwen35b-gpu0.log"
