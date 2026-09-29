#!/usr/bin/env bash
# Mars/phobos llama-server für InferenzQuelle.
# Gemma-4 26B-A4B QAT, Vulkan, turbo4/4 KV-Cache, 2 Slots à 128k.
# Vision (mmproj-F16) AKTIV seit 2026-09-29 — §5.20 Retest: funktioniert.
#
# KV-Cache: turbo4/4 (K=turbo4 4.25x, V=turbo4 4.25x) seit 2026-09-29.
#   V-Cache höhere Präzision als turbo3 (4.25 vs 3.125 bit).
#   Funktioniert mit mmproj + 262144 seit §5.20-Retest.
#
# Echte KV-Buffer-Größen (GQA-korrigiert, gemessen auf Phobos):
#   turbo3/3 bei 2x128k = 1.0 GB (passt in 1 GiB Default-Suballocation)
#   turbo4/3 bei 2x128k = 1.3 GB  (braucht 2 GiB Suballocation)
# Alle passen problemlos in GTT (27.6 GB). GTT ist nicht limitierend.
#
# turbo4 V + mmproj + 262144 (2026-08-15, §5.17): damals BLOCKED
#   (keine Pipeline-Cache-Hits, 5+min Recompiles). RETEST 2026-09-29 (§5.20):
#   FUNKTIONIERT — Vision korrekt in 2.8s, ~23 t/s Decode.
#   Damalige Ursache war vermutlich TTM-Deckel ~15 GiB + nogttspill,
#   nicht (nur) der VkPipelineCache. Siehe Trilium SWumEN7WOXBI §5.19/§5.20.
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
#   --cache-reuse 1        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key)
#
# Vision: AKTIV seit 2026-09-29 (§5.20 Retest — historischer Blocker war
#   TTM-Deckel + nogttspill, nicht der VkPipelineCache).
#   mmproj-F16.gguf (GPU-Offload): Encode 4.1s @280 Tok, ~19s @1024 Tok.
#   --image-min/max-tokens steuert die Scan-Granularitaet (wie Qwen).
#   Gemma-4-Bild-Chunks nutzen non-causal Attention: -ub muss mindestens
#   die tatsaechliche Chunk-Tokenzahl aufnehmen (sonst GGML_ASSERT).
#   -ub 2048, image-max-tokens 1984: 64 Tokens (~3 %) Reserve fuer
#   Alignment/zusatzliche Bild-Tokens; bei neuen Bildern erneut pruefen.
#
# Performance (2026-09-29): ~18 t/s pro Slot parallel (~36 t/s Aggregat) —
#   2×128k, turbo4/4, mmproj-F16 aktiv. Vision-Encode: 4.1s @280 Tok,
#   ~19s @1024 Tok (GPU). Text im Bild wird ab ~1024 Tok gelesen (OCR).
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
IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-512}"
IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1984}"

if [[ ! "$IMAGE_MIN_TOKENS" =~ ^[0-9]+$ || ! "$IMAGE_MAX_TOKENS" =~ ^[0-9]+$ ]] ||
   (( IMAGE_MIN_TOKENS < 1 || IMAGE_MIN_TOKENS > IMAGE_MAX_TOKENS || IMAGE_MAX_TOKENS > 1984 )); then
  echo "error: image token range must satisfy 1 <= min <= max <= 1984 with -ub 2048" >&2
  exit 1
fi

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

# GPU-Exklusivitaets-Guard (2026-09-29, Host-OOM auf mars):
# Auf dieser APU (GTT == System-RAM, 30 GB Host) duerfen niemals zwei
# llama-server parallel allokieren. SIGTERM braucht Sekunden bis die
# GTT-Buffer freigegeben sind — ein neuer Server waehrend der alte noch
# freigibt => kombinierter Peak > RAM => Host-Kernel-OOM.
# Regel: stoppen -> Prozess weg + GTT gefallen -> erst dann starten.
deadline=$((SECONDS + 90))
while pgrep -x llama-server >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "error: other llama-server process still running after 90s" >&2
    echo "       (check: pgrep -af llama-server)" >&2; exit 1
  fi
  echo "waiting for other llama-server to exit (GTT release)..." >&2
  sleep 3
done
GTT_USED="/sys/bus/pci/devices/0000:65:00.0/mem_info_gtt_used"
if [[ -r "$GTT_USED" ]]; then
  used=$(cat "$GTT_USED")
  # >4 GiB Rest-GTT ohne laufenden Server = anderer Allokator oder Leak
  if (( used > 4294967296 )); then
    echo "warning: gtt_used=${used} bytes — GPU memory not fully released" >&2
  fi
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
# ⚠️ nogttspill (2026-09-29 RCA): verbietet RADV den GTT-Spill-Pfad.
#   War urspruenglich fuer die 262144+mmproj Pathologie auf gfx1103.
#   ABER: ohne Spill schlaegt die amdgpu CS-Validation bei grosser
#   Gesamt-Allokation fehl -> "Not enough memory for command submission"
#   -> vk::Queue::submit ErrorDeviceLost beim Context-Init.
#   Auf UMA-APUs (VRAM-Carveout ~1 GiB, GTT == System-RAM) ist das Flag
#   kontraproduktiv — nur noch nircache setzen.
#   Siehe Trilium SWumEN7WOXBI §5.15 + zYeLUsss9udM / o6jGT8Qwqm4y.
export RADV_PERFTEST="${RADV_PERFTEST:-nircache}"

cd "$ROOT"
exec "$SERVER" \
  -m "$MAIN" \
  --mmproj "${MMPROJ:-/jade/models/gemma-4-26B-A4B-it/mmproj-F16.gguf}" \
  --image-min-tokens "$IMAGE_MIN_TOKENS" \
  --image-max-tokens "$IMAGE_MAX_TOKENS" \
  --host "$HOST" --port "$PORT" \
  -c 262144 -ngl 99 \
  -ctk turbo4 -ctv turbo4 -fa on \
  -fit off \
  -ub 2048 \
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
