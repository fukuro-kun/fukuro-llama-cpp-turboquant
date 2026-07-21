#!/usr/bin/env bash
# Styx VLM-Server (GLM-4.6V-Flash IQ4_XS) — Fallback für augen-check Skill.
# Vision-Modell für Bild-Analyse, OpenAI-kompatible /v1/chat/completions API.
#
# Rasch auf-/abbaubar:
#   Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-styx-vlm-server.sh
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-styx-vlm-server.sh
#
# Architektur: 1 llama-server Instanz auf der einzigen GPU (GTX 1070, 8GB).
#   Styx hat nur 1 GPU → der 26B-Chat-Server (Port 18080) muss VOR dem
#   VLM-Start gestoppt werden (VRAM-Konflikt). Das Stop-Skript startet
#   den 26B-Server wieder.
#
# IQ4_XS-Quant (5.0GB): 7208 MiB VRAM bei -c 2048, 7290 MiB bei -c 4096.
#   Reserve nach Anfrage: ~580-820 MiB. Verifiziert 2026-07-20.
#   31.2 t/s Generation, 210 t/s Prompt-Processing (Pascal-Architektur).
#
# Kontext: -c 4096 (kompakt für Reserve). Bild-Analyse braucht wenig Kontext.
# 1 Slot: styx ist Fallback, nicht für parallele Last ausgelegt.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
MODEL="${VLM_GGUF:-/data/modelle/GLM-4.6V-Flash/GLM-4.6V-Flash-IQ4_XS.gguf}"
MMPROJ="${VLM_MMPROJ:-/data/modelle/GLM-4.6V-Flash/mmproj-GLM-4.6V-Flash-F16.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18081}"
CTX="${CTX:-4096}"
SLOTS="${SLOTS:-1}"
CHAT_SERVICE="${CHAT_SERVICE:-llama-server-styx}"
CHAT_PORT="${CHAT_PORT:-18080}"

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

# --- Port-Check (VLM) ---
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: VLM port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# --- 26B-Chat-Server stoppen (VRAM freigeben) ---
echo "Stoppe 26B-Chat-Server ($CHAT_SERVICE) für VRAM-Freigabe..."
if systemctl --user is-active "$CHAT_SERVICE" >/dev/null 2>&1; then
  systemctl --user stop "$CHAT_SERVICE"
  echo "  $CHAT_SERVICE gestoppt."
  # Warte bis VRAM frei (max 15s)
  for i in $(seq 1 15); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [[ "${USED:-9999}" -lt 200 ]]; then
      echo "  VRAM frei nach ${i}s (${USED} MiB)."
      break
    fi
    sleep 1
  done
else
  echo "  $CHAT_SERVICE läuft nicht — VRAM bereits frei."
fi

cd "$ROOT"

# --- VLM-Instanz starten ---
echo "Starte VLM-Server (Port $PORT, $SLOTS Slot × $((CTX/1024))k, IQ4_XS)..."
setsid "$SERVER" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" -ngl 999 \
  -fa on \
  --parallel "$SLOTS" -np "$SLOTS" \
  --temp 1.0 --top-p 0.95 --top-k 64 \
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
  echo "  Starte 26B-Chat-Server wieder (Fallback-Regression)..."
  systemctl --user start "$CHAT_SERVICE" || true
  exit 1
fi

# --- VRAM-Verifikation ---
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
echo "  VRAM: ${VRAM_USED} MiB used, ${VRAM_FREE} MiB frei"

# --- PID speichern für Stop-Skript ---
echo "$PID" > /tmp/styx-vlm.pid

echo
echo "=== Styx VLM-Server (GLM-4.6V-Flash IQ4_XS) gestartet ==="
echo "  http://0.0.0.0:$PORT  ($SLOTS Slot × $((CTX/1024))k, IQ4_XS)  PID $PID"
echo "  26B-Chat-Server ($CHAT_SERVICE) ist GESTOPPT — Stop-Skript startet ihn wieder."
echo
echo "Stop: bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-styx-vlm-server.sh"
echo "Log:  tail -f /tmp/vlm-server.log"
