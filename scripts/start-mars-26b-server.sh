#!/usr/bin/env bash
# Mars/phobos llama-server für InferenzQuelle.
# Gemma-4 26B-A4B QAT, Vulkan, f16 KV-Cache, 2 Slots, +Vision (mmproj).
#
# KV-Cache: f16 (nicht turbo3/4) — TurboQuant Vulkan-Shader wurden revertiert
#   (Commits 603f47105, 9cbabbad8). f16-Fallback-Workaround in llama-context.cpp
#   wurde mit cb3b1d571 entfernt. Bis Shader wiederhergestellt sind: f16.
#
# -fit off: Verhindert dass fit_params ngl auf 0 reduziert bei --mmproj auf APU
#   (mmproj reserviert GPU-Speicher in der fit_params-Margin, auf unified-memory
#   APUs bleibt nichts mehr für das Hauptmodell → CPU-Fallback).
#
# Cache-Konfiguration (konservativ für unified memory APU):
#   --cache-ram 6144         6 GB CPU-RAM für serialisierte KV-States
#                            (phobos LXC hat 28 GB, aber unified memory:
#                             Modell 14.2 GB + KV-Cache 256k × 2 Slots im
#                             selben RAM-Pool. Konservativ um GTT-Overflow
#                             zu vermeiden — siehe 188k-Klippe RCA 2026-07-12)
#   --cache-reuse 256        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key)
#                            Bei 2 Slots besonders wertvoll für Cache-Reuse
#
# Vision (seit 2026-08-10):
#   --mmproj Q6_K            Vision-Encoder (SigLIP ~550M, Q6_K ~806MB)
#                            Läuft über GTT (shared RAM) bei Vulkan.
#                            Ermöglicht Bildverarbeitung via gemma-4-26B.
#
# Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-mars-26b-server.sh
# Stop:  systemctl --user stop llama-server.service
# Log:   journalctl --user -u llama-server.service -f

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-/jade/models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
PORT="${PORT:-18080}"
HOST="${HOST:-0.0.0.0}"
MMPROJ="${MMPROJ_GGUF:-/home/fukuro/modelle/gemma-4-26B-A4B-it/mmproj-Q6_K.gguf}"

# Modell-Check
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: $MAIN" >&2; exit 1
fi
if [[ ! -f "$MMPROJ" ]]; then
  echo "WARNUNG: mmproj nicht gefunden: $MMPROJ — Vision DEAKTIVIERT" >&2
fi

# Port frei?
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# thecodacus MoE-Optimierungen
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"

cd "$ROOT"
exec "$SERVER" \
  -m "$MAIN" \
  --mmproj "$MMPROJ" \
  --host "$HOST" --port "$PORT" \
  -c 262144 -ngl 99 \
  -ctk f16 -ctv f16 -fa on \
  -fit off \
  --parallel 2 -np 2 --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram 6144 \
  --cache-reuse 256 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --no-warmup \
  --metrics --slots \
  --log-timestamps --log-prefix \
  "$@"
