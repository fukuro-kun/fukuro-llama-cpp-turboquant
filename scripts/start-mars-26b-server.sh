#!/usr/bin/env bash
# Mars/phobos llama-server für InferenzQuelle.
# Gemma-4 26B-A4B QAT, Vulkan, turbo3/3 KV-Cache, 2 Slots à 128k.
# Vision (mmproj) DEAKTIVIERT seit 13.08.2026 — siehe §5.11 in Trilium SWumEN7WOXBI.
#
# KV-Cache: turbo3/turbo3 (K+V=turbo3, 5.1x Kompression).
#   Production-Default. turbo3/3 funktioniert zuverlässig mit 262144 (ohne mmproj).
#
#   turbo3/4 (V=turbo4, 3.8x, höhere Präzision) ist EXPERIMENTELL — siehe unten.
#
# Echte KV-Buffer-Größen (GQA-korrigiert, gemessen auf Phobos):
#   turbo3/3 bei 2x128k = 1.0 GB  (passt in 1 GiB Default-Suballocation)
#   turbo3/4 bei 2x128k = 1.3 GB  (braucht 2 GiB Suballocation)
#   turbo4/4 bei 2x128k = 1.4 GB
# Alle passen problemlos in GTT (27.6 GB). GTT ist nicht limitierend.
#
# ⚠️ turbo3/4 + mmproj + 262144 = NICHT PRODUKTIONS-TAUGLICH (2026-08-13):
#   RADV (Mesa 25.0.7, PHOENIX) kompiliert turbo4 V FlashAttention-Shader
#   extrem langsam (~14 Min pro Pipeline-Variante). Jede unterschiedliche
#   Batch-Size erzeugt eine neue Variante. Der ggml Vulkan-Pipeline-Cache
#   (GGML_VK_CACHE_DIR) speichert VkPipelineCache-Objekte, aber RADV's
#   interne Shader-Kompilierung wird nicht effektiv gecacht — der Mesa-
#   Shader-Cache (~/.cache/mesa_shader_cache_db/) wird via mmap geladen aber
#   nicht zurückgeschrieben (index seit Mai 2025 unverändert, 934 KB).
#   turbo3/4 ohne mmproj oder mit 131072 Kontext funktioniert (27.3 t/s).
#   Nur die Kombination turbo3/4 + mmproj + 262144 ist betroffen.
#   Für turbo3/4-Experimente: CTV=turbo4 setzen + GGML_VK_SUBALLOCATION_BLOCK_SIZE=2GiB.
#
# GGML_VK_SUBALLOCATION_BLOCK_SIZE=2147483648 (2 GiB):
#   Default ist 1 GiB (ggml-vulkan.cpp, Fragmentierungs-Limit, NICHT BIOS-Carveout).
#   turbo3/4 non-SWA KV-Cache = 1180 MiB > 1 GiB Default.
#   Ohne Erhoehung: KV-Cache-Buffer-Allocation schlaegt fehl -> CPU-Fallback.
#   Buffer > 1 GiB landen automatisch im GTT (System-RAM) — Normalfall auf APU.
#   Für turbo3/3 nicht streng nötig (1.0 GB <= 1 GiB), aber als Sicherheits-
#   Puffer gesetzt — verhindert Edge-Case-Fallback bei Fragmentierung.
#
# Kontext: 262144 (256k, 2 Slots à 128k) — volle Modellkapazität.
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
# Vision (seit 2026-08-10, DEAKTIVIERT 2026-08-13):
#   --mmproj Q6_K            Vision-Encoder (SigLIP ~550M, Q6_K ~806MB)
#   ⚠️ mmproj + 262144 + turbo3/3 = 0.01 t/s durch RADV-Kompilierung (§5.11 SWumEN7WOXBI)
#   Übergangsweise deaktiviert. Vision-Requests → venus/uranus.
#   Reaktivierung nach Mesa-Upgrade ≥25.1.x oder Precompile-Strategie.
#
# Performance (2026-08-12): ~27.3 t/s (tg), ~49.6 t/s (pp) — 2×128k, turbo3/3.
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

# turbo3/4 KV-Cache braucht 2 GiB Suballocation-Block (Default: 1 GiB).
# turbo3/3 braucht es nicht zwingend (KV-Buffer 1.0 GB <= 1 GiB Default),
# aber als Sicherheits-Puffer gegen Fragmentierung gesetzt.
export GGML_VK_SUBALLOCATION_BLOCK_SIZE="${GGML_VK_SUBALLOCATION_BLOCK_SIZE:-2147483648}"

# Vulkan Pipeline-Cache auf Disk — vermeidet Re-Kompilierung bei Restart.
# Wird beim sauberen Shutdown (SIGTERM) geschrieben. Bei SIGKILL geht der Cache verloren.
# Cache wird validiert gegen pipelineCacheUUID (driver/GPU-Wechsel → automatischer Reset).
# Hinweis: Für turbo3/3 ist der Cache effektiv (pipelines werden wiederverwendet).
#          Für turbo3/4+mmproj+262144 ist der Cache ineffective (RADV recompiliert
#          turbo4 V FA-Shader trotz Pipeline-Cache — siehe Header-Kommentar).
export GGML_VK_CACHE_DIR="${GGML_VK_CACHE_DIR:-/home/fukuro/.cache/ggml-vk-pipeline-cache}"

cd "$ROOT"
exec "$SERVER" \
  -m "$MAIN" \
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
