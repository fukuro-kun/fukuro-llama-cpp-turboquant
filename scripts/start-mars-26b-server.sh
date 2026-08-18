#!/usr/bin/env bash
# Mars/phobos llama-server für InferenzQuelle.
# Gemma-4 26B-A4B QAT, Vulkan, turbo4/3 KV-Cache, 2 Slots à 128k.
# Vision (mmproj) DEAKTIVIERT — turbo4/3 ohne mmproj für maximale TG-Performance.
#
# KV-Cache: turbo3/turbo4 (K=turbo3 5.1x, V=turbo4 4.25x Kompression).
#   V-Cache höhere Präzision als turbo3/3 (4.25 vs 3.125 bit).
#   turbo4/3 ohne mmproj funktioniert zuverlässig mit 262144 Kontext.
#   turbo4/3 + mmproj + 262144 → ❌ BLOCKED (RADV VkPipelineCache defekt, §5.17).
#
# Echte KV-Buffer-Größen (GQA-korrigiert, gemessen auf Phobos):
#   turbo3/3 bei 2x128k = 1.0 GB (passt in 1 GiB Default-Suballocation)
#   turbo4/3 bei 2x128k = 1.3 GB  (braucht 2 GiB Suballocation)
# Alle passen problemlos in GTT (27.6 GB). GTT ist nicht limitierend.
#
# ⚠️ turbo4 V + mmproj + 262144 Wartungsfenster-Test (2026-08-15, §5.17):
#   RADV VkPipelineCache produziert keine Cache-Hits für turbo4 V FA-Shader.
#   Precompile nutzlos — jeder Restart erfordert 5+min pro Pipeline-Variante.
#   turbo4/3 ohne mmproj funktioniert: Pipeline-Cache wird effektiv genutzt.
#   Siehe Trilium SWumEN7WOXBI §5.17 und docs/fork/2026-08-15_TURBO4_V_INVESTIGATION.md.
#
# GGML_VK_SUBALLOCATION_BLOCK_SIZE=2147483648 (2 GiB):
#   Default ist 1 GiB (ggml-vulkan.cpp, Fragmentierungs-Limit, NICHT BIOS-Carveout).
#   turbo4/3 non-SWA KV-Cache = 1180 MiB > 1 GiB Default.
#   Ohne Erhoehung: KV-Cache-Buffer-Allocation schlaegt fehl -> CPU-Fallback.
#   Buffer > 1 GiB landen automatisch im GTT (System-RAM) — Normalfall auf APU.
#
# Kontext: 262144 (256k, 2 Slots à 131072 = 128k) — volle Modellkapazität.
#
# -fit off: Verhindert dass fit_params ngl auf 0 reduziert bei mmproj auf APU.
#   Auch ohne mmproj gesetzt als Sicherheits-Puffer.
#
# --no-warmup: Verhindert 10+ Min Warmup-Hang (RADV Pipeline-Kompilierung im
#   Warmup-Forward-Pass). Erster echter Request kompiliert Pipelines stattdessen.
#
# Cache-Konfiguration:
#   --cache-ram 12288        12 GB CPU-RAM für serialisierte KV-States
#                            (freies RAM aus fehlendem mmproj genutzt)
#   --cache-reuse 1        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key)
#
# Vision: DEAKTIVIERT. Vision-Requests werden an venus geroutet.
#   turbo4/3 + mmproj + 262144 → ❌ BLOCKED (RADV Pipeline-Cache defekt).
#   turbo3/3 + mmproj + 262144 + nogttspill funktionierte, aber TG langsamer.
#
# Performance (2026-08-15): ~16.8 t/s (tg parallel) — 2×128k, turbo4/3, kein mmproj.
#   Erster Request mit Pipeline-Cache: 2.9s (Cache funktioniert für turbo4/3 ohne mmproj).
#   Startup: ~290s (Pipeline-Kompilierung, nur beim ersten Start nach Cache-Reset).
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
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"

# turbo4/3 KV-Cache braucht 2 GiB Suballocation-Block (Default: 1 GiB).
# turbo3/3 braucht es nicht zwingend (KV-Buffer 1.0 GB <= 1 GiB Default),
# aber als Sicherheits-Puffer gegen Fragmentierung gesetzt.
export GGML_VK_SUBALLOCATION_BLOCK_SIZE="${GGML_VK_SUBALLOCATION_BLOCK_SIZE:-2147483648}"

# Vulkan Pipeline-Cache auf Disk — vermeidet Re-Kompilierung bei Restart.
# Wird beim sauberen Shutdown (SIGTERM) geschrieben. Bei SIGKILL geht der Cache verloren.
# Cache wird validiert gegen pipelineCacheUUID (driver/GPU-Wechsel → automatischer Reset).
# Hinweis: Für turbo4/3 ohne mmproj ist der Cache effektiv (erster Request 2.9s mit Cache).
#          Für turbo4 V + mmproj + 262144 ist der Cache ineffective (RADV recompiliert
#          turbo4 V FA-Shader trotz Pipeline-Cache — siehe §5.17).
export GGML_VK_CACHE_DIR="${GGML_VK_CACHE_DIR:-/home/fukuro/.cache/ggml-vk-pipeline-cache}"

# Mesa/RADV Shader-Cache — speichert kompilierte Shader-ISA (NIR→ISA).
# Wird async von Mesa's Background-Thread geschrieben, bei vkDestroyDevice geflusht.
# MESA_SHADER_CACHE_DIR ist der Basis-Pfad; Mesa legt Subdirs (mesa_shader_cache/,
# radv_builtin_shaders/) darunter an. Default-Backend: mesa_shader_cache_db (Mesa-DB).
# Backup: /home/fukuro/.cache/vulkan-cache-backups/ (siehe precompile-vulkan-shaders.sh)
export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/home/fukuro/.cache/mesa-shader-cache}"
export MESA_SHADER_CACHE_MAX_SIZE="${MESA_SHADER_CACHE_MAX_SIZE:-2G}"
# NIR-Cache beschleunigt Replay bei Graphics Pipeline Libraries.
# nogttspill: Deaktiviert RADV GTT-Spill-Heuristik — fixt 262144+mmproj Pathologie
# auf gfx1103 (UMA-APU, GTT == System-RAM, Spilling ist kostenlos).
# Siehe Trilium SWumEN7WOXBI §5.15.
export RADV_PERFTEST="${RADV_PERFTEST:-nircache,nogttspill}"

cd "$ROOT"
exec "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT" \
  -c 262144 -ngl 99 \
  -ctk turbo4 -ctv turbo3 -fa on \
  -fit off \
  --parallel 2 -np 2 --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram 12288 \
  --cache-reuse 1 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --no-warmup \
  --metrics --slots \
  --log-timestamps --log-prefix \
  "$@"
