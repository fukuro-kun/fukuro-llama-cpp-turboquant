#!/usr/bin/env bash
# Styx llama-server für InferenzQuelle + pandora-voice-service.
# Gemma-4 26B-A4B QAT, MoE-Offload, turbo4/3 KV-Cache.
#
# Kontext: 196608 (196k) statt 224k — Stabilität (2026-07-19):
#   Bei 224k vollem Kontext würde RSS 31.4 GB > 31 GB RAM → Swap → CPU-I/O
#   → verstärkt MoE-Bottleneck → 503 Service Unavailable (beobachtet 13:58).
#   196k: RSS ~28.4 GB, 2.6 GB Reserve, Swap-Risiko MITTEL statt HOCH.
#   tg/pp kaum beeinflusst (CPU-MoE dominiert, SWA liest nur Window).
#   Reserve gewinnt wenn Kontext meist <100k belegt ist (Router-Chat, Eval).
#
# Cache-Konfiguration (wie Uranus, angepasst an 32 GB RAM):
#   --cache-ram 16384        16 GB CPU-RAM für serialisierte KV-States
#                            (27 GB available — llama 9.5 GB + Whisper 2.2 GB = 11.7 GB)
#   --cache-reuse 256        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key automatisch)
#
# Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-styx-26b-server.sh
# Stop:  systemctl --user stop llama-server-styx.service
# Log:   journalctl --user -u llama-server-styx.service -f

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-/data/modelle/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
PORT="${PORT:-18080}"
HOST="${HOST:-0.0.0.0}"

# Modell-Check
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: $MAIN" >&2; exit 1
fi

# Port frei?
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# thecodacus MoE-Optimierungen
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"

cd "$ROOT"
exec "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT" \
  -c 196608 -ngl 999 --n-cpu-moe 20 \
  -ctk turbo4 -ctv turbo3 -fa on \
  --parallel 1 -np 1 --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram 16384 \
  --cache-reuse 256 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --metrics --slots \
  --log-timestamps --log-prefix \
  "$@"
