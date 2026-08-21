#!/usr/bin/env bash
# Stoppt den Uranus 26B-A4B llama-server GPU 1 (2. Instanz, Port 18082).
# Rasch abbaubar — sauberes Shutdown.

set -euo pipefail

PIDFILE="/tmp/uranus-26b-gpu1.pid"
PORT="${PORT:-18082}"

STOPPED=0

# --- Instanz stoppen ---
if [[ -f "$PIDFILE" ]]; then
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stoppe llama-server GPU 1 (PID $PID, Port $PORT)..."
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
    echo "  Gestoppt."
    STOPPED=1
  else
    echo "llama-server GPU 1 (PID $PID) läuft nicht mehr."
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

# --- Verifikation ---
echo
if [[ "$STOPPED" == "1" ]]; then
  echo "=== Uranus 26B-A4B Server GPU 1 gestoppt ==="
else
  echo "=== Kein laufender 26B-Server auf GPU 1 gefunden ==="
fi

# Health-Check verifizieren dass Port frei
H=$(curl -s --connect-timeout 1 http://127.0.0.1:$PORT/health 2>/dev/null || echo "frei")
echo "  Port $PORT: ${H:-frei}"
