#!/usr/bin/env bash
# Uranus llama-server für InferenzQuelle Router.
# Qwen3.8-27B Swift-1.5 — zwei Varianten via VARIANT env var:
#   VARIANT=al  (Default): ukisai Swift-1.5 aligned   GSQ-RCO-IQ3_S-mtp  (12,1 GB)
#   VARIANT=unc            ajgazin Swift-1.5 uncensored UD-Q3_K_XL-MTP   (13,1 GB)
# 1 Instanz auf GPU 0, Port 18080. Beide Modelle sind dense (kein MoE-Offload)
# und passen komplett in 16 GiB VRAM — MTP-Head ist eingebettet.
#
# Evaluierte Produktions-Flags (docs/qwen38-eval/RANKING.md §8):
#   --spec-type draft-mtp --spec-draft-n-max 2   (embedded MTP, n=2 ist Optimum)
#   -ctk turbo4 -ctv turbo3 -fa on               (Haupt-KV quantisiert)
#   Draft-KV bleibt f16: -ctkd/-ctvd ist bei <=~130k netto ~+100 MiB SCHLECHTER
#     (Draft-KV −295 MiB, aber Draft-Compute-Workspace +396 MiB)
#   --parallel 1: par2 halbiert den Durchsatz auf 16-GB-Karten (decode-bound)
#   -c 131072 auf uranus verifiziert; 147456 geht auf eris (3080), uranus
#     nicht getestet → conservativ 131k, via CTX env überschreibbar
#
# Request-Sampler (Client-seitig, Router reicht durch):
#   seed=-1 temp=0.7 top_p=0.95 min_p=0.0 reasoning_effort=medium
#   Bei content="" retryen (Loop-Rate ~8%, seed×host-abhängig)
#
# Rasch auf-/abbaubar:
#   Start: bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-uranus-qwen38-server.sh
#          VARIANT=unc bash ~/git/fukuro-llama-cpp-turboquant/scripts/start-uranus-qwen38-server.sh
#   Stop:  bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh
#          (gleicher Port 18080, gleiches PID-File)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
CTX="${CTX:-131072}"
VARIANT="${VARIANT:-al}"

MODEL_DIR="/media/fukuro/raid5/modelle/Qwen/Qwen3.8-27B"
case "$VARIANT" in
  al)
    MAIN="${MAIN_GGUF:-${MODEL_DIR}/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}"
    PROFILE_NAME="swiftal (aligned)"
    ;;
  unc)
    MAIN="${MAIN_GGUF:-${MODEL_DIR}/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q3_K_XL.gguf}"
    PROFILE_NAME="swiftunc (uncensored)"
    ;;
  *)
    echo "error: VARIANT muss 'al' oder 'unc' sein (war: '$VARIANT')" >&2
    exit 2
    ;;
esac

# --- Modell-Check ---
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: GGUF nicht gefunden: $MAIN" >&2
  echo "  Erwartet auf uranus: $MODEL_DIR/" >&2
  exit 1
fi

# --- Port-Check ---
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use (lsof -ti:$PORT)" >&2; exit 1
fi

# --- VRAM-Warnung (kein adaptives Offload — Modell passt komplett) ---
VRAM_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ' || true)
if [[ -n "$VRAM_FREE_MIB" && "$VRAM_FREE_MIB" -lt 15000 ]]; then
  echo "WARNUNG: GPU0 nur ${VRAM_FREE_MIB} MiB frei — ${PROFILE_NAME} braucht ~13-15 GiB + KV" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
fi

# --- Instanz: GPU 0, Port 18080 ---
echo "Starte llama-server (Qwen3.8-27B ${PROFILE_NAME}, GPU 0, Port $PORT, ctx $CTX)..."
setsid "$SERVER" \
  -m "$MAIN" \
  --host "$HOST" --port "$PORT" \
  -dev CUDA0 \
  -c "$CTX" -ngl 99 -ub 512 -fa on \
  -ctk turbo4 -ctv turbo3 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --parallel 1 --cont-batching \
  --jinja \
  --temp 0.7 --top-p 0.95 --min-p 0.0 \
  --no-context-shift \
  --no-warmup -fit off \
  --cache-ram 16384 \
  --cache-reuse 1 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > "/tmp/uranus-qwen38-${VARIANT}-gpu0.log" 2>&1 &
PID=$!
echo "  PID: $PID, Log: /tmp/uranus-qwen38-${VARIANT}-gpu0.log"

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
  echo "  Log: /tmp/uranus-qwen38-${VARIANT}-gpu0.log"
  echo "  PID: $PID (noch laufend)"
  exit 1
fi

# --- PID speichern für Stop-Skript (gleiches PID-File wie 26B) ---
echo "$PID" > /tmp/uranus-26b-gpu0.pid

echo
echo "=== Uranus Qwen3.8-27B ${PROFILE_NAME} gestartet ==="
echo "  http://0.0.0.0:$PORT  (ctx $CTX, MTP n=2, turbo4/3)  PID $PID"
echo
echo "Stop: bash ~/git/fukuro-llama-cpp-turboquant/scripts/stop-uranus-26b-server.sh"
echo "Log:  tail -f /tmp/uranus-qwen38-${VARIANT}-gpu0.log"
