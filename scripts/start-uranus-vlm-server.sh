#!/usr/bin/env bash
# Uranus VLM-Server (GLM-4.6V-Flash) für augen-check Skill.
# Vision-Modell für Bild-Analyse, OpenAI-kompatible /v1/chat/completions API.
#
# Rasch auf-/abbaubar:
#   Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-uranus-vlm-server.sh
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-vlm-server.sh
#
# Architektur: 1 llama-server Instanz auf GPU 1 (physisch).
#   CUDA_VISIBLE_DEVICES=1 isoliert den Prozess auf GPU 1 → kein CUDA-Context
#   auf GPU 0 (spart ~2GB VRAM auf dem Chat-Backend). -dev CUDA0 nach Remapping.
#   GPU 0 bleibt vollständig dem 26B-Chat-Server vorbehalten.
#
# Auto-Unload (--sleep-idle-seconds 120): Modell wird nach 2 min Inaktivität
# von der GPU entladen (~6GB frei). Prozess bleibt aktiv (~260MB CUDA-Kontext),
# beim nächsten Request wacht er in ~4s auf und lädt das Modell wieder.
# Verifiziert 2026-07-18: 8.5GB → 262MB nach 120s, Auto-Wake in 4s.
#
# 2 Slots: erlaubt parallele Bild-Analyse durch mehrere Agenten. Bei -np 1
# würden sich Requests serialisieren; 2 Slots geben Headroom ohne nennenswerten
# VRAM-Overhead (nur aktive Slots belegen KV-Cache).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MODEL="${VLM_GGUF:-/media/fukuro/raid5/modelle/GLM-4.6V-Flash/GLM-4.6V-Flash-Q4_K_M.gguf}"
MMPROJ="${VLM_MMPROJ:-/media/fukuro/raid5/modelle/GLM-4.6V-Flash/mmproj-GLM-4.6V-Flash-F16.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18081}"
CTX="${CTX:-8192}"
SLOTS="${SLOTS:-2}"
SLEEP_IDLE="${SLEEP_IDLE:-120}"

# --- Modell-Check ---
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: VLM model not found: $MODEL" >&2; exit 1
fi
if [[ ! -f "$MMPROJ" ]]; then
  echo "error: mmproj not found: $MMPROJ" >&2; exit 1
fi

# --- Port-Check ---
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# --- GPU-Isolation ---
# Physische GPU 1 isolieren → kein CUDA-Context auf GPU 0 (spart ~2GB VRAM).
# Nach Remapping ist -dev CUDA0 die physische GPU 1.
export CUDA_VISIBLE_DEVICES=1

cd "$ROOT"

# --- Instanz: GPU 1 (physisch), Port 18081, 2 Slots × 8k ---
echo "Starte VLM-Server (GPU 1, Port $PORT, $SLOTS Slots × $((CTX/1024))k, --sleep-idle-seconds $SLEEP_IDLE)..."
setsid "$SERVER" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --host "$HOST" --port "$PORT" \
  -dev CUDA0 \
  -c "$CTX" -ngl 999 \
  -fa on \
  --parallel "$SLOTS" -np "$SLOTS" \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --sleep-idle-seconds "$SLEEP_IDLE" \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > /tmp/vlm-server.log 2>&1 &
PID=$!
echo "  PID: $PID, Log: /tmp/vlm-server.log"

# --- Health-Check (warten bis ready) ---
echo
echo "Warte auf Health-Check (max 60s)..."
READY=0
for i in $(seq 1 60); do
  H=$(curl -s --connect-timeout 2 http://127.0.0.1:$PORT/health 2>/dev/null || echo "")
  if [[ "$H" == *"ok"* ]]; then
    READY=1
    echo "  Healthy nach ${i}s"
    break
  fi
  sleep 1
done

if [[ "$READY" == "0" ]]; then
  echo "WARNUNG: Health-Check nicht bestanden nach 60s"
  echo "  ($PORT): ${H:-keine Antwort}"
  echo "  Log: /tmp/vlm-server.log"
  echo "  PID: $PID (noch laufend)"
  exit 1
fi

# --- PID speichern für Stop-Skript ---
echo "$PID" > /tmp/uranus-vlm-gpu1.pid

echo
echo "=== Uranus VLM-Server (GLM-4.6V-Flash) gestartet ==="
echo "  http://0.0.0.0:$PORT  (GPU 1, $SLOTS Slots × $((CTX/1024))k, sleep-idle ${SLEEP_IDLE}s)  PID $PID"
echo
echo "Stop: bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-vlm-server.sh"
echo "Log:  tail -f /tmp/vlm-server.log"
