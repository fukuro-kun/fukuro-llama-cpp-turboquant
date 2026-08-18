#!/usr/bin/env bash
# Venus llama-server für InferenzQuelle.
# Gemma-4 26B-A4B QAT, Vulkan, f16 KV-Cache, 2 Slots à 128k, +Vision (mmproj).
#
# Venus hat eine AMD Vega iGPU (GCN/Renoir). Auf GCN ist turbo4/3 bei PP
# 35-54% langsamer als f16 (scalar FA fallback, Dequant-Overhead).
# Daher: f16 KV statt turbo4/3. 62 GB RAM bieten genug Platz für f16 bei 256k.
#
# -fit off: Verhindert dass fit_params ngl auf 0 reduziert bei --mmproj auf APU
#   (gleicher Bug wie auf Mars — fit_params reserviert GPU-Speicher für mmproj).
#
# --no-warmup: Verhindert langen Warmup-Hang (RADV Pipeline-Kompilierung).
#
# Cache-Konfiguration (großzügig — 40 GB RAM available nach Modell-Laden):
#   --cache-ram 16384        16 GB CPU-RAM für serialisierte KV-States
#                            (62 GB RAM — Modell 13.3 GB + f16 KV-Cache 256k×2
#                             = ~22 GB benutzt, 40 GB available)
#   --cache-reuse 1        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key)
#                            Bei 2 Slots besonders wertvoll für Cache-Reuse
#
# Vision (seit 2026-08-10):
#   --mmproj Q6_K            Vision-Encoder (SigLIP ~550M, Q6_K ~806MB)
#
# Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-venus-26b-server.sh
# Stop:  sudo systemctl stop llama-server-venus.service
# Log:   journalctl -u llama-server-venus.service -f
#
# Suspend-Policy: Venus schläft 08:00–13:00 Uhr (Meditationszeit).
# Außerhalb: Server läuft, danach `sudo venus-suspend`.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${HOME}/modelle/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
PORT="${PORT:-18080}"
HOST="${HOST:-0.0.0.0}"
MMPROJ="${MMPROJ_GGUF:-${HOME}/modelle/gemma-4-26B-A4B-it/mmproj-Q6_K.gguf}"

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
  -fit off \
  --parallel 2 -np 2 --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram 16384 \
  --cache-reuse 1 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --no-warmup \
  --metrics --slots \
  --log-timestamps --log-prefix \
  "$@"
