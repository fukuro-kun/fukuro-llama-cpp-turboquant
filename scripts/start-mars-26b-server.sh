#!/usr/bin/env bash
# Mars/phobos llama-server für InferenzQuelle.
# Gemma-4 26B-A4B QAT, Vulkan, turbo3/3 KV-Cache, 2 Slots à ~79k.
# Vision (mmproj) AKTIVIERT seit 14.08.2026 — Kontext auf 161792 reduziert.
#
# KV-Cache: turbo3/turbo3 (K+V=turbo3, 5.1x Kompression).
#   Production-Default. turbo3/3 funktioniert zuverlässig.
#
#   turbo3/4 (V=turbo4, 3.8x, höhere Präzision) ist EXPERIMENTELL — siehe unten.
#
# Echte KV-Buffer-Größen (GQA-korrigiert, gemessen auf Phobos):
#   turbo3/3 bei 2x80k = 0.63 GB (passt in 1 GiB Default-Suballocation)
#   turbo3/3 bei 2x128k = 1.0 GB (passt in 1 GiB Default-Suballocation)
#   turbo3/4 bei 2x128k = 1.3 GB  (braucht 2 GiB Suballocation)
# Alle passen problemlos in GTT (27.6 GB). GTT ist nicht limitierend.
#
# ⚠️ Kontext-Limit bei mmproj + turbo3/3 + Vulkan/RADV (2026-08-14):
#   Bei n_ctx >= 163840 (160k) mit mmproj kompiliert RADV (Mesa 26.1.2, PHOENIX)
#   FlashAttention-Shader extrem langsam (~110s pro 8 Tokens, 0.07 t/s).
#   Bei n_ctx <= 161792 (158k) mit mmproj funktioniert es einwandfrei (32 t/s).
#   Der Schwellwert liegt exakt bei 163840 — vermutlich durch einen RADV-Compiler-
#   Pfad der bei großen Buffer-Größen eine andere Code-Generierung wählt.
#   Mesa-Upgrade (25.0.7→26.1.2) und Precompile-Cache lösten das Problem nicht.
#   Lösung: Kontext auf 161792 reduzieren (2 Slots à 80896 Tokens ≈ 79k).
#   Siehe Trilium SWumEN7WOXBI §5.11 für vollständige Diagnose.
#
# GGML_VK_SUBALLOCATION_BLOCK_SIZE=2147483648 (2 GiB):
#   Default ist 1 GiB (ggml-vulkan.cpp, Fragmentierungs-Limit, NICHT BIOS-Carveout).
#   turbo3/4 non-SWA KV-Cache = 1180 MiB > 1 GiB Default.
#   Ohne Erhoehung: KV-Cache-Buffer-Allocation schlaegt fehl -> CPU-Fallback.
#   Buffer > 1 GiB landen automatisch im GTT (System-RAM) — Normalfall auf APU.
#   Für turbo3/3 nicht streng nötig (1.0 GB <= 1 GiB), aber als Sicherheits-
#   Puffer gesetzt — verhindert Edge-Case-Fallback bei Fragmentierung.
#
# Kontext: 161792 (158k, 2 Slots à 80896 ≈ 79k) — mmproj-kompatibel.
#   Ohne mmproj wären 262144 (256k, 2×128k) möglich — volle Modellkapazität.
#   Trade-off: 79k Kontext pro Slot statt 128k, dafür Vision-Unterstützung.
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
# Vision (seit 2026-08-10, DEAKTIVIERT 13.08., REAKTIVIERT 14.08.2026):
#   --mmproj Q6_K            Vision-Encoder (SigLIP ~550M, Q6_K ~806MB)
#   Mit n_ctx=161792 funktioniert mmproj+turbo3/3+Vulkan zuverlässig (32 t/s).
#   n_ctx>=163840 mit mmproj → 0.07 t/s (RADV-Compiler-Pathologie).
#
# Performance (2026-08-14): ~32 t/s (tg), ~43 t/s (pp) — 2×79k, turbo3/3, mmproj.
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
  --mmproj "$MMPROJ" \
  --host "$HOST" --port "$PORT" \
  -c 161792 -ngl 99 \
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
