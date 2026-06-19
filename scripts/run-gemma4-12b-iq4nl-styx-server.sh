#!/usr/bin/env bash
# ============================================================================
# Gemma-4 12B IQ4_NL + IQ4_XS Draft MTP-Server auf styx
# ============================================================================
# OpenAI-kompatible API + Web-Browser parallel nutzbar
#
# Endpunkte:
#   OpenAI API:  http://styx:18080/v1/chat/completions
#   Web-Browser: http://styx:18080/
#   Health:      http://styx:18080/health
#
# Verwendung:
#   ./scripts/run-gemma4-12b-iq4nl-styx-server.sh
#
# Beispiel API-Request:
#   curl http://styx:18080/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"gemma-4","messages":[{"role":"user","content":"Hallo!"}],"max_tokens":100}'
# ============================================================================

set -euo pipefail

# Styx-spezifische Pfade
SERVER="/data/git/fukuro-llama-cpp-turboquant/build/bin/llama-server"
MAIN="/data/modelle/gemma-4-12b-it-IQ4_NL.gguf"
DRAFT="/data/modelle/drafts/gemma-4-12b-it-assistant.IQ4_XS.gguf"

# Konfiguration
CTX="${CTX:-196608}"
NGL="${NGL:-99}"
NGL_DRAFT="${NGL_DRAFT:-99}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo3}"
CTKD="${CTKD:-turbo3}"
CTVD="${CTVD:-turbo3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
FA="${FA:-on}"
TEMP="${TEMP:-0.7}"
PARALLEL="${PARALLEL:-2}"

# MTP Draft Parameter
DRAFT_BLOCK_SIZE="${DRAFT_BLOCK_SIZE:-3}"
DRAFT_MAX="${DRAFT_MAX:-8}"

echo "=== Gemma-4 12B MTP-Server auf styx ==="
echo "Target: $MAIN"
echo "Draft:  $DRAFT"
echo "ctx=${CTX}, ngl=${NGL}, cache=${CTK}/${CTV}"
echo "API:    http://${HOST}:${PORT}/v1/chat/completions"
echo "WebUI:  http://${HOST}:${PORT}/"
echo "========================================"

# Pruefe Binaries und Modelle
if [[ ! -f "$SERVER" ]]; then
    echo "error: llama-server nicht gefunden: ${SERVER}" >&2
    exit 1
fi
if [[ ! -f "$MAIN" ]]; then
    echo "error: Target-Modell nicht gefunden: ${MAIN}" >&2
    exit 1
fi
if [[ ! -f "$DRAFT" ]]; then
    echo "error: Draft-Modell nicht gefunden: ${DRAFT}" >&2
    exit 1
fi

# GPU leeren
killall -9 llama-server 2>/dev/null || true
sleep 2

# Server starten
exec "$SERVER" \
    -m "$MAIN" \
    --model-draft "$DRAFT" \
    --spec-type mtp \
    -c "$CTX" \
    -ngl "$NGL" \
    -ngld "$NGL_DRAFT" \
    -ctk "$CTK" \
    -ctv "$CTV" \
    -ctkd "$CTKD" \
    -ctvd "$CTVD" \
    -fa "$FA" \
    --host "$HOST" \
    --port "$PORT" \
    --parallel "$PARALLEL" \
    -np "$PARALLEL" \
    --cont-batching \
    --draft-block-size "$DRAFT_BLOCK_SIZE" \
    --draft-max "$DRAFT_MAX" \
    --draft-min 0 \
    --temp "$TEMP" \
    --metrics \
    --slots \
    --log-timestamps \
    --log-prefix \
    "$@"
