#!/usr/bin/env bash
# ============================================================================
# Gemma-4 26B-A4B QAT Server auf Styx (GTX 1070, 8GB VRAM)
# ============================================================================
# PRODUKTIV-STANDARD seit 2026-07-10 (QAT + 224k Kontext)
#   - Modell: QAT-UD-Q4_K_XL (14.2G, kleiner als IQ4_NL 14.7G)
#   - Kontext: 229376 (224k) — +64k vs IQ4_NL (160k)
#   - MoE-Offloading mit --n-cpu-moe 20 (20 Expert-Layer auf CPU)
#   - MTP: AUS (Q4_0 Draft bremst -14% auf Styx, siehe Benchmark 2026-07-10)
#   - Pinning/Prefetch: AN (GGML_CUDA_REGISTER_HOST=1)
#   - Prompt-Cache: --cache-ram 16384 (16 GB), --cache-reuse 1,
#     --slot-cache-key-similarity 0.5, --slot-cache-key-min-prefix 64
#     (wie Uranus, angepasst an 32 GB RAM mit Whisper-Service)
#
# Endpunkte:
#   OpenAI API:  http://styx:18080/v1/chat/completions
#   Web-Browser: http://styx:18080/
#   Health:      http://styx:18080/health
# ============================================================================

set -euo pipefail

# thecodacus MoE-Optimierungen — aktiviert
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
# 2 Prefetch-Slots: Sweet-Spot aus Benchmark 2026-07-13
# 1 Slot: +9.8% tg aber -11.9% pp (kein Overlap möglich)
# 2 Slots: +28.9% pp, +2.1% tg (reiner Win)
# 3 Slots (default): identisch zu 2 Slots bei pp, minimal besser bei tg
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"

# Styx-spezifische Pfade
SERVER="/data/git/fukuro-llama-cpp-turboquant/build/bin/llama-server"
MAIN="/data/modelle/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
DRAFT="/data/modelle/gemma-4-26B-A4B-it/drafts/mtp-gemma-4-26B-A4B-it-Q4_0.gguf"

# Konfiguration — QAT Produktiv-Standard
CTX="${CTX:-229376}"
NGL="${NGL:-999}"
N_CPU_MOE="${N_CPU_MOE:-20}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
FA="${FA:-on}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo4}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
PARALLEL="${PARALLEL:-1}"

# Prompt-Cache (wie Uranus, angepasst für 32 GB RAM + Whisper-Service)
# 16 GB cache-ram: sicher bei 27 GB available (llama 9.5 GB + Whisper 2.2 GB = 11.7 GB)
CACHE_RAM="${CACHE_RAM:-16384}"
CACHE_REUSE="${CACHE_REUSE:-1}"
CACHE_KEY_SIMILARITY="${CACHE_KEY_SIMILARITY:-0.5}"
CACHE_KEY_MIN_PREFIX="${CACHE_KEY_MIN_PREFIX:-64}"

# MTP (Speculative Decoding) — standardmäßig AUS
# Q4_0 Draft: ~50% Acceptance aber -14% Speed (Draft konkurriert mit Expert-Prefetch)
MTP="${MTP:-0}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"

echo "=== Gemma-4 26B-A4B QAT Server auf Styx ==="
echo "Target: $MAIN"
echo "ctx=${CTX} (224k), ngl=${NGL}, n-cpu-moe=${N_CPU_MOE}, cache=${CTK}/${CTV}"
echo "Prompt-Cache: ram=${CACHE_RAM}MB, reuse=${CACHE_REUSE}, key-sim=${CACHE_KEY_SIMILARITY}, min-prefix=${CACHE_KEY_MIN_PREFIX}"
echo "Pinning: GGML_CUDA_REGISTER_HOST=${GGML_CUDA_REGISTER_HOST}"
echo "Prefetch: GGML_SCHED_PREFETCH_EXPERTS=${GGML_SCHED_PREFETCH_EXPERTS} (slots=${GGML_SCHED_PREFETCH_SLOTS})"
if [[ "$MTP" == "1" ]]; then
    echo "MTP: AN (n_max=${SPEC_DRAFT_N_MAX}) — Draft: $DRAFT"
    echo "WARNUNG: MTP Q4_0 Draft ist -14% langsamer auf Styx (Benchmark 2026-07-10)"
else
    echo "MTP: AUS (Default — Q4_0 Draft bremst -14%)"
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
        -ngld 999 \
        --n-cpu-moe "$N_CPU_MOE" \
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
        --cache-ram "$CACHE_RAM" \
        --cache-reuse "$CACHE_REUSE" \
        --slot-cache-key-similarity "$CACHE_KEY_SIMILARITY" \
        --slot-cache-key-min-prefix "$CACHE_KEY_MIN_PREFIX" \
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
        --n-cpu-moe "$N_CPU_MOE" \
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
        --cache-ram "$CACHE_RAM" \
        --cache-reuse "$CACHE_REUSE" \
        --slot-cache-key-similarity "$CACHE_KEY_SIMILARITY" \
        --slot-cache-key-min-prefix "$CACHE_KEY_MIN_PREFIX" \
        --metrics \
        --slots \
        --log-timestamps \
        --log-prefix \
        "$@"
fi
