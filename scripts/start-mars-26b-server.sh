#!/usr/bin/env bash
# Mars/phobos llama-server für InferenzQuelle.
# Gemma-4 26B-A4B QAT, Vulkan, turbo3 KV-Cache, 2 Slots à 128k, +Vision (mmproj).
#
# KV-Cache: turbo3/turbo3 (5.1x Kompression, 3.125 bit/element).
#   TurboQuant Vulkan-Shader sind aktiv (Commit cb3b1d571) — die Reverts
#   603f47105 und 9cbabbad8 betrafen nur Mixed K/V und einen dequant-fix,
#   nicht die Haupt-SET_ROWS/mul_mat_vec/FlashAttention-Shader.
#   turbo3/3 ist die kleinste Option (26 GB bei 2×128k) und passt in GTT.
#   f16 (133 GB) und q4_0 (37 GB) überschreiten GTT bei 2×128k.
#
# Kontext: 262144 (256k, 2 Slots à 128k) — volle Modellkapazität.
#   GTT-Budget: 27.6 GB total - 13.5 GB Modell - 0.8 GB mmproj = 13.3 GB für KV.
#   turbo3/3 bei 2×128k = 26 GB → knapp, aber Vulkan nutzt unified memory
#   (GTT = RAM, 28 GB LXC) und funktioniert mit Buffer-Eviction.
#
# -fit off: Verhindert dass fit_params ngl auf 0 reduziert bei --mmproj auf APU
#   (mmproj reserviert GPU-Speicher in der fit_params-Margin, auf unified-memory
#   APUs bleibt nichts mehr für das Hauptmodell → CPU-Fallback).
#
# --no-warmup: Verhindert 10+ Min Warmup-Hang (RADV Pipeline-Kompilierung im
#   Warmup-Forward-Pass). Erster echter Request kompiliert Pipelines stattdessen.
#
# Cache-Konfiguration:
#   --cache-ram 6144         6 GB CPU-RAM für serialisierte KV-States
#   --cache-reuse 256        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key)
#
# Vision (seit 2026-08-10):
#   --mmproj Q6_K            Vision-Encoder (SigLIP ~550M, Q6_K ~806MB)
#                            Läuft über GTT (shared RAM) bei Vulkan.
#
# Performance (2026-08-12): ~27.5 t/s (tg), ~44 t/s (pp) — 2×128k, turbo3/3.
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
  -ctk turbo3 -ctv turbo3 -fa on \
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
