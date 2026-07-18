#!/usr/bin/env bash
# Uranus llama-server für InferenzQuelle Router.
# Gemma-4 26B-A4B QAT, MoE-Offload, turbo3/4 KV-Cache.
# 2 Instanzen (eine pro GPU), adaptiv: VLM geladen → GPU1 nur 1 Slot.
#
# Rasch auf-/abbaubar:
#   Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-uranus-26b-server.sh
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh
#
# Cache-Konfiguration (wie Styx, angepasst an 128 GB RAM + 2x 16 GB VRAM):
#   --cache-ram 32768        32 GB CPU-RAM für serialisierte KV-States (128 GB available)
#   --cache-reuse 256        KV-shift für nicht-prefix Chunks (RAG, Tool-Defs)
#   --slot-cache-key-*       cache_key-Validierung (Router sendet cache_key automatisch)
#
# MoE-Offload (--n-cpu-moe 10): 10 Experten pro Instanz auf CPU.
#   Ohne MoE-Offload: 14.2 GB Modell → nur 1.8 GB für KV-Cache → zu wenig für 128k/Slot.
#   Mit 10 MoE-Offload: ~8 GB auf GPU → ~8 GB für KV-Cache → 2x128k möglich.
#
#   WICHTIG — CPU-MoE-Bottleneck (erfahren 2026-07-18):
#   Pro parallelem Request werden die CPU-MoE-Layer berechnet.
#   2 Instanzen × 2 Slots × 10 MoE = 40 MoE-Berechnungen pro Token-Schritt auf 8 CPU-Kernen.
#   Das ist der Flaschenhals, nicht die GPU!
#   - 1 Request aktiv:  1×10 = 10 MoE-Schritte → ~40 t/s (GPU-dominiert)
#   - 4 Requests parallel: 4×10 = 40 MoE-Schritte → ~2.7 t/s (CPU-dominiert)
#   - Vergleich styx (1 Slot, 20 MoE): 1×20 = 20 → 7.3 t/s
#   --n-cpu-moe 20 (alt) war schlimmer: 4×20 = 80 → 1.6 t/s.
#   Nie wieder --n-cpu-moe 20 bei 2 Instanzen mit mehreren Slots!
#
# Architecture: 2 unabhängige llama-server Instanzen (kein Tensor-Parallelism).
#   Instanz 1: GPU 0 (-dev CUDA0), Port 18080, 2 Slots × 128k
#   Instanz 2: GPU 1 (-dev CUDA1), Port 18082, 1-2 Slots × 128k (adaptiv je nach VLM)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-/media/fukuro/raid5/modelle/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT0="${PORT0:-18080}"
PORT1="${PORT1:-18082}"
CTX="${CTX:-262144}"          # 256k total, aufgeteilt auf Slots
CACHE_RAM="${CACHE_RAM:-32768}"  # 32 GB CPU prompt-cache (128 GB RAM available)

# --- Modell-Check ---
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: $MAIN" >&2
  echo "  Download: ssh ganymed → rsync von /titan/topas/modelle/gemma-4-26B-A4B-it/" >&2
  exit 1
fi

# --- Port-Checks ---
for port in "$PORT0" "$PORT1"; do
  if lsof -ti:"$port" >/dev/null 2>&1; then
    echo "error: port $port already in use (lsof -ti:$port)" >&2; exit 1
  fi
done

# --- VLM-Status auf GPU 1 prüfen (adaptiv) ---
# VLM (GLM-4.6V-Flash) läuft auf Port 18081, -dev CUDA1.
# Wenn VLM ein Modell geladen hat → GPU1 hat weniger VRAM → nur 1 Slot.
# Wenn VLM entladen/leer → GPU1 hat volle VRAM → 2 Slots.
VLM_LOADED=0
if curl -s --connect-timeout 2 http://127.0.0.1:18081/props >/dev/null 2>&1; then
  # VLM läuft — prüfe ob ein Slot aktiv ist
  VLM_SLOTS=$(curl -s --connect-timeout 2 http://127.0.0.1:18081/slots 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for s in d if s.get('is_processing',False)))" 2>/dev/null || echo "0")
  if [[ "$VLM_SLOTS" != "0" ]]; then
    VLM_LOADED=1
  fi
fi

if [[ "$VLM_LOADED" == "1" ]]; then
  SLOTS1=1
  CTX1=131072   # 1 Slot × 128k
  echo "VLM ist auf GPU 1 aktiv ($VLM_SLOTS Slots belegt) → Instanz 2: 1 Slot × 128k"
else
  SLOTS1=2
  CTX1=262144   # 2 Slots × 128k
  echo "VLM nicht aktiv → Instanz 2: 2 Slots × 128k"
fi

# --- thecodacus MoE-Optimierungen ---
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"

cd "$ROOT"

# --- Instanz 1: GPU 0, Port 18080, 2 Slots × 128k ---
echo "Starte Instanz 1 (GPU 0, Port $PORT0, 2 Slots × 128k)..."
setsid "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT0" \
  -dev CUDA0 \
  -c "$CTX" -ngl 999 --n-cpu-moe 10 \
  -ctk turbo3 -ctv turbo4 -fa on \
  --parallel 2 -np 2 --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram "$CACHE_RAM" \
  --cache-reuse 256 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > /tmp/uranus-26b-gpu0.log 2>&1 &
PID0=$!
echo "  PID: $PID0, Log: /tmp/uranus-26b-gpu0.log"

# --- Instanz 2: GPU 1, Port 18082, adaptiv Slots ---
echo "Starte Instanz 2 (GPU 1, Port $PORT1, $SLOTS1 Slot(s) × 128k)..."
setsid "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT1" \
  -dev CUDA1 \
  -c "$CTX1" -ngl 999 --n-cpu-moe 10 \
  -ctk turbo3 -ctv turbo4 -fa on \
  --parallel "$SLOTS1" -np "$SLOTS1" --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram "$CACHE_RAM" \
  --cache-reuse 256 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > /tmp/uranus-26b-gpu1.log 2>&1 &
PID1=$!
echo "  PID: $PID1, Log: /tmp/uranus-26b-gpu1.log"

# --- Health-Check (warten bis beide ready) ---
echo
echo "Warte auf Health-Check (max 120s)..."
READY=0
for i in $(seq 1 120); do
  H0=$(curl -s --connect-timeout 2 http://127.0.0.1:$PORT0/health 2>/dev/null || echo "")
  H1=$(curl -s --connect-timeout 2 http://127.0.0.1:$PORT1/health 2>/dev/null || echo "")
  if [[ "$H0" == *"ok"* ]] && [[ "$H1" == *"ok"* ]]; then
    READY=1
    echo "  Beide Instanzen healthy nach ${i}s"
    break
  fi
  sleep 1
done

if [[ "$READY" == "0" ]]; then
  echo "WARNUNG: Health-Check nicht bestanden nach 120s"
  echo "  GPU0 ($PORT0): ${H0:-keine Antwort}"
  echo "  GPU1 ($PORT1): ${H1:-keine Antwort}"
  echo "  Logs: /tmp/uranus-26b-gpu0.log, /tmp/uranus-26b-gpu1.log"
  echo "  PIDs: $PID0, $PID1 (noch laufend)"
  exit 1
fi

# --- PIDs speichern für Stop-Skript ---
echo "$PID0" > /tmp/uranus-26b-gpu0.pid
echo "$PID1" > /tmp/uranus-26b-gpu1.pid

echo
echo "=== Uranus 26B-A4B Server gestartet ==="
echo "  Instanz 1: http://0.0.0.0:$PORT0  (GPU 0, 2 Slots × 128k)  PID $PID0"
echo "  Instanz 2: http://0.0.0.0:$PORT1  (GPU 1, $SLOTS1 Slot(s) × 128k)  PID $PID1"
echo
echo "Stop: bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh"
echo "Logs: tail -f /tmp/uranus-26b-gpu0.log /tmp/uranus-26b-gpu1.log"
