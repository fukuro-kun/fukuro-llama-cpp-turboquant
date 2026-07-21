#!/usr/bin/env bash
# Stoppt den Styx VLM-Server (GLM-4.6V-Flash IQ4_XS) und startet den
# 26B-Chat-Server wieder. Sauberes Shutdown + VRAM-Rückgabe.
#
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-styx-vlm-server.sh

set -euo pipefail

PIDFILE="/tmp/styx-vlm.pid"
PORT="${PORT:-18081}"
CHAT_SERVICE="${CHAT_SERVICE:-llama-server-styx}"
CHAT_PORT="${CHAT_PORT:-18080}"

STOPPED=0

# --- VLM-Instanz stoppen ---
if [[ -f "$PIDFILE" ]]; then
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stoppe VLM-Server (PID $PID, Port $PORT)..."
    kill "$PID" 2>/dev/null || true
    # Warte max 10s auf clean shutdown
    for i in $(seq 1 10); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    # Falls noch da: SIGKILL
    if kill -0 "$PID" 2>/dev/null; then
      echo "  SIGKILL nach 10s"
      kill -9 "$PID" 2>/dev/null || true
    fi
    echo "  VLM gestoppt."
    STOPPED=1
  else
    echo "VLM-Server (PID $PID) läuft nicht mehr."
  fi
  rm -f "$PIDFILE"
else
  echo "Keine PID-Datei — suche nach Prozess auf Port $PORT..."
  PORTPID=$(lsof -ti:"$PORT" 2>/dev/null || true)
  if [[ -n "$PORTPID" ]]; then
    echo "  Beende PID $PORTPID auf Port $PORT"
    kill "$PORTPID" 2>/dev/null || true
    sleep 3
    kill -9 "$PORTPID" 2>/dev/null || true
    STOPPED=1
  fi
fi

# --- Warte bis VRAM frei (max 10s) ---
if [[ "$STOPPED" == "1" ]]; then
  for i in $(seq 1 10); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [[ "${USED:-9999}" -lt 200 ]]; then
      echo "  VRAM frei nach ${i}s (${USED} MiB)."
      break
    fi
    sleep 1
  done
fi

# --- 26B-Chat-Server wieder starten ---
echo
echo "Starte 26B-Chat-Server ($CHAT_SERVICE) wieder..."
if systemctl --user is-active "$CHAT_SERVICE" >/dev/null 2>&1; then
  echo "  $CHAT_SERVICE läuft bereits — überspringe Start."
else
  systemctl --user start "$CHAT_SERVICE"
  echo "  $CHAT_SERVICE gestartet."
  # Warte auf Health-Check (max 60s — 26B braucht länger zum Laden)
  echo "  Warte auf Health-Check (max 60s)..."
  READY=0
  for i in $(seq 1 60); do
    H=$(curl -s --connect-timeout 2 http://127.0.0.1:$CHAT_PORT/health 2>/dev/null || echo "")
    if [[ "$H" == *"ok"* ]]; then
      READY=1
      echo "  26B-Server healthy nach ${i}s"
      break
    fi
    sleep 1
  done
  if [[ "$READY" == "0" ]]; then
    echo "WARNUNG: 26B-Server nicht healthy nach 60s"
    echo "  ($CHAT_PORT): ${H:-keine Antwort}"
    echo "  Manueller Check: systemctl --user status $CHAT_SERVICE"
  fi
fi

# --- Verifikation ---
echo
if [[ "$STOPPED" == "1" ]]; then
  echo "=== Styx VLM-Server gestoppt, 26B-Chat-Server wieder aktiv ==="
else
  echo "=== Kein laufender VLM-Server gefunden, 26B-Server sichergestellt ==="
fi

# Health-Check verifizieren dass VLM-Port frei
H=$(curl -s --connect-timeout 1 http://127.0.0.1:$PORT/health 2>/dev/null || echo "frei")
echo "  VLM-Port $PORT: ${H:-frei}"
