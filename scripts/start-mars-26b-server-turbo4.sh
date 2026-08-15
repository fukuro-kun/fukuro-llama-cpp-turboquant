#!/usr/bin/env bash
# Mars/phobos llama-server für InferenzQuelle — turbo3/4 VARIANTE.
# Gemma-4 26B-A4B QAT, Vulkan, turbo3/4 KV-Cache, 2 Slots à 128k, +Vision (mmproj).
#
# ⚠️ EXPERIMENTELL — nicht als Production-Default nutzen!
#   turbo3/4 + mmproj + 262144 hat zwei bekannte Probleme:
#   1. OOM: turbo3/4 braucht 1.8 GB mehr KV-Speicher als turbo3/3.
#      Mit --cache-ram 6144 (6GB) übersteigt das 28GB RAM → OOM-Kill.
#      Fix: --cache-ram 0 (kein Prompt-Cache bis Pipeline-Cache vollständig).
#   2. ACO Pipeline-Kompilierung pathologisch langsam: Jede turbo4 V FA-
#      Pipeline-Variante braucht ~3min zu kompilieren. Bei 262144+mmproj
#      gibt es viele Varianten → Startup kann 30-60min dauern.
#      Fix: Einmaliger Precompile (precompile-vulkan-shaders.sh mit turbo4 V),
#      danach Pipeline-Cache-Hit beim Restart.
#   Siehe docs/fork/2026-08-15_TURBO4_V_INVESTIGATION.md
#
# Unterschiede zu start-mars-26b-server.sh (turbo3/3 Production):
#   -ctv turbo4           (statt turbo3) — 4.25 bit V-Cache, höhere Präzision
#   --cache-ram 0         (statt 6144) — OOM-Vermeidung
#   Separater Cache-Pfad  — turbo4 V Pipelines getrennt von turbo3/3
#
# KV-Cache: turbo3/turbo4 (K=turbo3 3.125bit, V=turbo4 4.25bit).
#   V-Cache ist präziser (4.25 vs 3.125 bit), braucht aber 1.8 GB mehr.
#   Echte KV-Buffer-Größen (GQA-korrigiert):
#     turbo3/3 bei 2x128k = 1.0 GB
#     turbo3/4 bei 2x128k = 1.3 GB (braucht 2 GiB Suballocation)
#   Gesamt-RAM: turbo3/4 = 25.6 GB (Modell 13.7 + KV 11.9) vs turbo3/3 = 23.8 GB
#
# RADV_PERFTEST=nircache,nogttspill — gleiche Flags wie turbo3/3 Production.
#   nogttspill fixt die GTT-Spill-Heuristik auf UMA-APUs.
#
# Periodischer Pipeline-Cache-Flush (Commit d795404be):
#   Der VK Pipeline-Cache wird nach JEDER Pipeline-Kompilierung gespeichert,
#   nicht nur beim Shutdown. Schützt vor Cache-Verlust bei OOM-Kill/SIGKILL.
#   Kritisch für turbo4 V da der Precompile 30-60min dauert.
#
# Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-mars-26b-server-turbo4.sh
# Stop:  kill <PID> (SIGTERM für sauberen Shutdown + finalen Cache-Flush)
# Log:   journalctl --user -u llama-server.service -f (wenn als service)

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

# turbo3/4 KV-Cache braucht 2 GiB Suballocation-Block (Default: 1 GiB).
export GGML_VK_SUBALLOCATION_BLOCK_SIZE="${GGML_VK_SUBALLOCATION_BLOCK_SIZE:-2147483648}"

# Separater Vulkan Pipeline-Cache für turbo4 V — verhindert Cache-Konflikte
# mit turbo3/3 Pipelines. Beim ersten Start leer → alle Pipelines werden neu
# kompiliert (30-60min). Beim zweiten Start → Cache-Hit → schnell.
export GGML_VK_CACHE_DIR="${GGML_VK_CACHE_DIR:-/home/fukuro/.cache/ggml-vk-pipeline-cache-turbo4}"

# Separater Mesa Shader-Cache für turbo4 V.
export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/home/fukuro/.cache/mesa-shader-cache-turbo4}"
export MESA_SHADER_CACHE_MAX_SIZE="${MESA_SHADER_CACHE_MAX_SIZE:-4G}"

# NIR-Cache + nogttspill — gleiche RADV-Flags wie turbo3/3 Production.
export RADV_PERFTEST="${RADV_PERFTEST:-nircache,nogttspill}"

cd "$ROOT"
exec "$SERVER" \
  -m "$MAIN" \
  --mmproj "$MMPROJ" \
  --host "$HOST" --port "$PORT" \
  -c 262144 -ngl 99 \
  -ctk turbo3 -ctv turbo4 -fa on \
  -fit off \
  --parallel 2 -np 2 --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram 0 \
  --no-warmup \
  --metrics --slots \
  --log-timestamps --log-prefix \
  "$@"
