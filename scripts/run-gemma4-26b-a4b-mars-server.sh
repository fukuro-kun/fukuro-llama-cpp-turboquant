#!/usr/bin/env bash
# ============================================================================
# Gemma-4 26B-A4B QAT Server auf Mars (AMD Radeon 760M, Vulkan)
# ============================================================================
# PRODUKTIV-STANDARD seit 2026-07-12 (QAT + 256k Kontext, LXC phobos)
#   - Modell: QAT-UD-Q4_K_XL (14.2G, +10% pp / +16.6% tg vs IQ4_NL auf Mars)
#   - Kontext: 262144 (256k) — Modell-Maximum voll ausgenutzt (vorher 224k)
#   - Slots: 2 à 128k (parallel=2)
#   - KV-Cache: turbo3/turbo4 (mixed) — beste Vulkan-Konfiguration
#   - MTP: AUS (Q4_0 Draft bremst -2.4% auf Mars, siehe Benchmark 2026-07-10)
#   - FlashAttention: on (turbo4 FA aktiv, turbo3 scalar fallback)
#
# Endpunkte:
#   OpenAI API:  http://mars:18080/v1/chat/completions
#   Web-Browser: http://mars:18080/
#   Health:      http://mars:18080/health
# ============================================================================

set -euo pipefail

# Mars-spezifische Pfade
SERVER="/home/fukuro/git/fukuro-llama-cpp-turboquant/build/bin/llama-server"
MAIN="/jade/models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
DRAFT="/jade/models/gemma-4-26B-A4B-it/drafts/mtp-gemma-4-26B-A4B-it-Q4_0.gguf"

# Konfiguration — QAT Produktiv-Standard
CTX="${CTX:-262144}"
NGL="${NGL:-99}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
FA="${FA:-on}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo4}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
PARALLEL="${PARALLEL:-2}"

# MTP (Speculative Decoding) — standardmäßig AUS
# Q4_0 Draft: 57% Acceptance aber -2.4% Speed (Draft-Overhead auf shared-memory GPU)
MTP="${MTP:-0}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"

echo "=== Gemma-4 26B-A4B QAT Server auf Mars ==="
echo "Target: $MAIN"
echo "ctx=${CTX} (256k max), ngl=${NGL}, cache=${CTK}/${CTV}, FA=${FA}, parallel=${PARALLEL}"
if [[ "$MTP" == "1" ]]; then
    echo "MTP: AN (n_max=${SPEC_DRAFT_N_MAX}) — Draft: $DRAFT"
    echo "WARNUNG: MTP Q4_0 Draft ist -2.4% langsamer auf Mars (Benchmark 2026-07-10)"
else
    echo "MTP: AUS (Default — Q4_0 Draft bremst -2.4%)"
fi
echo "API:    http://${HOST}:${PORT}/v1/chat/completions"
echo "WebUI:  http://${HOST}:${PORT}/"
echo "==============================================================="

# Pruefe Binaries und Modelle
if [[ ! -f "$SERVER" ]]; then
    echo "error: llama-server nicht gefunden: ${SERVER}" >&2
    exit 1
fi
if [[ ! -f "$MAIN" ]]; then
    echo "error: Target-Modell nicht gefunden: ${MAIN}" >&2
    exit 1
fi
if [[ "$MTP" == "1" ]] && [[ ! -f "$DRAFT" ]]; then
    echo "error: Draft-Modell nicht gefunden: ${DRAFT}" >&2
    exit 1
fi

# GPU leeren
killall -9 llama-server 2>/dev/null || true
sleep 2

# Server starten — MTP optional
if [[ "$MTP" == "1" ]]; then
    exec "$SERVER" \
        -m "$MAIN" \
        --model-draft "$DRAFT" \
        --spec-type draft-mtp \
        -c "$CTX" \
        -ngl "$NGL" \
        -ngld 99 \
        -ctk "$CTK" \
        -ctv "$CTV" \
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
else
    exec "$SERVER" \
        -m "$MAIN" \
        -c "$CTX" \
        -ngl "$NGL" \
        -ctk "$CTK" \
        -ctv "$CTV" \
        -fa "$FA" \
        --host "$HOST" \
        --port "$PORT" \
        --parallel "$PARALLEL" \
        -np "$PARALLEL" \
        --cont-batching \
        --temp "$TEMP" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --metrics \
        --slots \
        --log-timestamps \
        --log-prefix \
        "$@"
fi
