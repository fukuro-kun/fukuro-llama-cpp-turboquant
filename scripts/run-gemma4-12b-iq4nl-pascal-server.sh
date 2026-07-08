#!/usr/bin/env bash
# ============================================================================
# Gemma-4 12B IQ4_NL + IQ4_XS Draft MTP-Server auf Pascal-Host
# ============================================================================
# OpenAI-kompatible API + Web-Browser parallel nutzbar
#
# Endpunkte:
#   OpenAI API:  http://Pascal-Host:18080/v1/chat/completions
#   Web-Browser: http://Pascal-Host:18080/
#   Health:      http://Pascal-Host:18080/health
#
# Verwendung:
#   ./scripts/run-gemma4-12b-iq4nl-pascal-server.sh
#
# Beispiel API-Request:
#   curl http://Pascal-Host:18080/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"gemma-4","messages":[{"role":"user","content":"Hallo!"}],"max_tokens":100}'
# ============================================================================

set -euo pipefail

# Pascal-Host-spezifische Pfade
SERVER="/path/to/fukuro-llama-cpp-turboquant/build/bin/llama-server"
MAIN="/data/modelle/gemma-4-12b-it-IQ4_NL.gguf"
DRAFT="/data/modelle/drafts/gemma-4-12b-it-assistant.IQ4_XS.gguf"

# Konfiguration
CTX="${CTX:-163840}"
NGL="${NGL:-99}"
NGL_DRAFT="${NGL_DRAFT:-99}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo3}"
CTKD="${CTKD:-turbo3}"
CTVD="${CTVD:-turbo3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
FA="${FA:-on}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
PARALLEL="${PARALLEL:-2}"

# MTP Draft Parameter
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-8}"

echo "=== Gemma-4 12B MTP-Server auf Pascal-Host ==="
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
    --spec-type draft-mtp \
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
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
    --spec-draft-n-min 0 \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --metrics \
    --slots \
    --log-timestamps \
    --log-prefix \
    "$@"
