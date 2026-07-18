#!/usr/bin/env bash
# Stoppt die Uranus 26B-A4B llama-server Instanzen.
# Rasch abbaubar — sauberes Shutdown der 2 Instanzen.

set -euo pipefail

PIDFILE0="/tmp/uranus-26b-gpu0.pid"
PIDFILE1="/tmp/uranus-26b-gpu1.pid"
PORT0="${PORT0:-18080}"
PORT1="${PORT1:-18082}"

STOPPED=0

# --- Instanz 1 stoppen ---
if [[ -f "$PIDFILE0" ]]; then
  PID0=$(cat "$PIDFILE0")
  if kill -0 "$PID0" 2>/dev/null; then
    echo "Stoppe Instanz 1 (PID $PID0, Port $PORT0)..."
    kill "$PID0" 2>/dev/null || true
    # Warte max 10s auf clean shutdown
    for i in $(seq 1 10); do
      kill -0 "$PID0" 2>/dev/null || break
      sleep 1
    done
    # Falls noch da: SIGKILL
    if kill -0 "$PID0" 2>/dev/null; then
      echo "  SIGKILL nach 10s"
      kill -9 "$PID0" 2>/dev/null || true
    fi
    echo "  Gestoppt."
    STOPPED=1
  else
    echo "Instanz 1 (PID $PID0) läuft nicht mehr."
  fi
  rm -f "$PIDFILE0"
else
  echo "Keine PID-Datei für Instanz 1 — suche nach Prozess auf Port $PORT0..."
  PORTPID=$(lsof -ti:"$PORT0" 2>/dev/null || true)
  if [[ -n "$PORTPID" ]]; then
    echo "  Beende PID $PORTPID auf Port $PORT0"
    kill "$PORTPID" 2>/dev/null || true
    sleep 3
    kill -9 "$PORTPID" 2>/dev/null || true
    STOPPED=1
  fi
fi

# --- Instanz 2 stoppen ---
if [[ -f "$PIDFILE1" ]]; then
  PID1=$(cat "$PIDFILE1")
  if kill -0 "$PID1" 2>/dev/null; then
    echo "Stoppe Instanz 2 (PID $PID1, Port $PORT1)..."
    kill "$PID1" 2>/dev/null || true
    for i in $(seq 1 10); do
      kill -0 "$PID1" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PID1" 2>/dev/null; then
      echo "  SIGKILL nach 10s"
      kill -9 "$PID1" 2>/dev/null || true
    fi
    echo "  Gestoppt."
    STOPPED=1
  else
    echo "Instanz 2 (PID $PID1) läuft nicht mehr."
  fi
  rm -f "$PIDFILE1"
else
  echo "Keine PID-Datei für Instanz 2 — suche nach Prozess auf Port $PORT1..."
  PORTPID=$(lsof -ti:"$PORT1" 2>/dev/null || true)
  if [[ -n "$PORTPID" ]]; then
    echo "  Beende PID $PORTPID auf Port $PORT1"
    kill "$PORTPID" 2>/dev/null || true
    sleep 3
    kill -9 "$PORTPID" 2>/dev/null || true
    STOPPED=1
  fi
fi

# --- Verifikation ---
echo
if [[ "$STOPPED" == "1" ]]; then
  echo "=== Uranus 26B-A4B Server gestoppt ==="
else
  echo "=== Keine laufenden 26B-Instanzen gefunden ==="
fi

# Health-Check verifizieren dass Ports frei
H0=$(curl -s --connect-timeout 1 http://127.0.0.1:$PORT0/health 2>/dev/null || echo "frei")
H1=$(curl -s --connect-timeout 1 http://127.0.0.1:$PORT1/health 2>/dev/null || echo "frei")
echo "  Port $PORT0: ${H0:-frei}"
echo "  Port $PORT1: ${H1:-frei}"
