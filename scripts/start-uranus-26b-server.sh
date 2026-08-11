#!/usr/bin/env bash
# Uranus llama-server für InferenzQuelle Router.
# Gemma-4 26B-A4B QAT, MoE-Offload, turbo3/4 KV-Cache.
# 1 Instanz auf GPU 0, 2 Slots × 128k. GPU 1 frei für VLM/FLUX.
#
# Rasch auf-/abbaubar:
#   Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-uranus-26b-server.sh
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh
#
# Cache-Konfiguration (wie Styx, angepasst an 128 GB RAM + 16 GB VRAM):
#   --cache-ram 32768        32 GB CPU-RAM für serialisierte KV-States (128 GB available)
#   --cache-reuse 256        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key automatisch)
#
# MoE-Offload (--n-cpu-moe 5): 5 Experten auf CPU, Rest auf GPU.
#   Ohne MoE-Offload: 14.2 GB Modell → nur 1.8 GB für KV-Cache → zu wenig für 128k/Slot.
#   Mit 5 MoE-Offload: ~10.5 GB auf GPU → ~5.5 GB für KV-Cache → 2x128k möglich.
#   Benchmark 2026-07-21 (GPU-isoliert, Lasttest 122k Tokens): MoE 5 = 55.7 t/s (+31% vs MoE 10),
#   732 MiB Reserve bei 122k echten Tokens unter Volllast — stabil, kein OOM.
#   Override: MOE=4 (aggressiv, +40% tg, OOM-Risiko bei 128k) oder MOE=10 (konservativ).
#   Adaptive Logik greift nur bei VRAM-Knappheit (< 8 GB frei): reduziert Slots/MoE-Offload.
#
#   WICHTIG — CPU-Konkurrenz bei 2 Instanzen (erfahren 2026-07-18):
#   2 unabhängige llama-server Prozesse auf EINER CPU überlasten sich gegenseitig,
#   selbst wenn nur 1 Instanz aktiv generiert! Die idle Instanz verbraucht CPU für
#   MoE-Prefetch (GGML_SCHED_PREFETCH_EXPERTS=1), Cache-Updates, Slot-Management.
#   - 2 Instanzen, 1 Request aktiv: 3.5 t/s (CPU 100%, load 15.5)
#   - 1 Instanz, 1 Request aktiv:   45 t/s (CPU 11%, load 1.0)
#   - 1 Instanz, 2 Requests parallel: 30 t/s pro Slot (CPU 53%, load 4.3)
#   Das ist ein 13x Speedup! Nie wieder 2 Instanzen auf einer CPU mit MoE-Offload.
#   GPU 1 bleibt frei für VLM (GLM-4.6V-Flash) und FLUX-Server.
#
# Architecture: 1 llama-server Instanz auf GPU 0 (physisch).
#   CUDA_VISIBLE_DEVICES=0 isoliert den Prozess auf GPU 0 → kein CUDA-Context
#   auf GPU 1 (spart ~2GB VRAM für VLM/FLUX). -dev CUDA0 nach Remapping.
#   Port 18080, 2 Slots × 128k Kontext.
#   GPU 1: VLM (Port 18081) + FLUX (Port 18083), ungenutzt für 26B.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-/media/fukuro/raid5/modelle/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
CTX="${CTX:-262144}"             # 256k total, aufgeteilt auf SLOTS = je Slot
SLOTS="${SLOTS:-2}"
CACHE_RAM="${CACHE_RAM:-32768}"  # 32 GB CPU prompt-cache (128 GB RAM available)
MOE="${MOE:-5}"                  # 5 MoE-Layer auf CPU (Default seit 2026-07-21, Lasttest 122k Tokens stabil, 732 MiB Reserve)

# --- Modell-Check ---
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: $MAIN" >&2
  echo "  Download: ssh ganymed → rsync von /titan/topas/modelle/gemma-4-26B-A4B-it/" >&2
  exit 1
fi

# --- Port-Check ---
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# --- VRAM-Check + adaptive Konfiguration ---
# GPU0 freier VRAM bestimmt die Konfiguration. Wenn andere Prozesse (Training,
# VLM, FLUX) VRAM belegen, reduzieren wir Slots/MoE adaptiv statt zu crashen.
#
# WICHTIG: Diese Logik läuft NUR beim Start (einmaliger nvidia-smi-Check).
# Der Server passt sich zur Laufzeit NICHT an — wenn VRAM später frei wird
# (z.B. VLM/FLUX stoppen), läuft der Server weiter mit der Start-Config.
# Um von freigewordenem VRAM zu profitieren: Server neu starten.
#
# Tabelle (freier VRAM auf GPU0, ab dem Schwellwert gilt die jeweilige Config):
#   ab 12 GB → 2 Slots × 128k, MoE 5   (Default, volle Kapazität, +31% Speed)
#   ab  8 GB → 2 Slots × 128k, MoE 10  (2 Slots bleiben, mehr MoE-Offload für VRAM-Reserve)
#   ab  6 GB → 1 Slot  × 128k, MoE 15  (VRAM-Knappheit: 1 Slot, mehr MoE auf CPU)
#   ab  4 GB → 1 Slot  × 128k, MoE 20  (VRAM-Knappheit: max MoE-Offload)
#   <   4 GB → Fehler, nicht starten   (zu wenig für Modell + KV-Cache)
#
# Override: SLOTS/MOE/CTX als env vars setzen überspringt die Adaption.
ADAPTIVE=${ADAPTIVE:-1}
if [[ "$ADAPTIVE" == "1" ]]; then
  VRAM_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')
  if [[ -z "$VRAM_FREE_MIB" ]]; then
    echo "WARNUNG: nvidia-smi nicht verfügbar, überspringe VRAM-Check" >&2
  else
    VRAM_FREE_GIB=$((VRAM_FREE_MIB / 1024))
    echo "VRAM-Check: GPU0 hat ${VRAM_FREE_GIB} GB frei"
    if   (( VRAM_FREE_GIB >= 12 )); then
      SLOTS=2; MOE=5; CTX=262144    # 2×128k, Default (Lasttest 2026-07-21: 122k Tokens stabil, 732 MiB Reserve)
    elif (( VRAM_FREE_GIB >= 8  )); then
      SLOTS=2; MOE=10; CTX=262144   # 2×128k, mehr MoE-Offload für VRAM-Reserve bei paralleler GPU-Nutzung
    elif (( VRAM_FREE_GIB >= 6  )); then
      SLOTS=1; MOE=15; CTX=131072   # 1×128k, VRAM-Knappheit: mehr MoE auf CPU
    elif (( VRAM_FREE_GIB >= 4  )); then
      SLOTS=1; MOE=20; CTX=131072   # 1×128k, VRAM-Knappheit: max MoE-Offload
    else
      echo "error: nur ${VRAM_FREE_GIB} GB VRAM frei auf GPU0 (mindestens 4 GB nötig)" >&2
      echo "  Belegung:" >&2
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
      echo "  Lösung: Training/VLM/FLUX stoppen oder mehr MoE-Offload (MOE=20) manuell setzen" >&2
      exit 2
    fi
    echo "  → Adaptiv: SLOTS=$SLOTS, MOE=$MOE, CTX=$CTX ($((CTX/1024))k)"
  fi
fi

# --- GPU-Isolation ---
# Physische GPU 0 isolieren → kein CUDA-Context auf GPU 1 (spart ~2GB VRAM
# für VLM/FLUX). Nach Remapping ist -dev CUDA0 die einzige sichtbare GPU.
export CUDA_VISIBLE_DEVICES=0

# --- thecodacus MoE-Optimierungen ---
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"

cd "$ROOT"

# --- Instanz: GPU 0, Port 18080, 2 Slots × 128k ---
echo "Starte llama-server (GPU 0, Port $PORT, $SLOTS Slots × $((CTX/SLOTS/1024))k, --n-cpu-moe $MOE)..."
setsid "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT" \
  -dev CUDA0 \
  -c "$CTX" -ngl 999 --n-cpu-moe "$MOE" \
  -ctk turbo3 -ctv turbo4 -fa on \
  --parallel "$SLOTS" -np "$SLOTS" --cont-batching \
  --reasoning off \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram "$CACHE_RAM" \
  --cache-reuse 256 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > /tmp/uranus-26b-gpu0.log 2>&1 &
PID=$!
echo "  PID: $PID, Log: /tmp/uranus-26b-gpu0.log"

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
  echo "  Log: /tmp/uranus-26b-gpu0.log"
  echo "  PID: $PID (noch laufend)"
  exit 1
fi

# --- PID speichern für Stop-Skript ---
echo "$PID" > /tmp/uranus-26b-gpu0.pid

echo
echo "=== Uranus 26B-A4B Server gestartet ==="
echo "  http://0.0.0.0:$PORT  (GPU 0, $SLOTS Slots × $((CTX/SLOTS/1024))k, --n-cpu-moe $MOE)  PID $PID"
echo
echo "Stop: bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh"
echo "Log:  tail -f /tmp/uranus-26b-gpu0.log"
