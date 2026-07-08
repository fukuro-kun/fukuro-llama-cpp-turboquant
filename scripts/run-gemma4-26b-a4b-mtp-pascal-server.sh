#!/usr/bin/env bash
# ============================================================================
# Gemma-4 26B-A4B IQ4_NL + Q4_K_M Draft MTP-Server auf Pascal-Host
# ============================================================================
# MoE-Offloading mit thecodacus Pinning + Prefetch Optimierungen
#
# Endpunkte:
#   OpenAI API:  http://Pascal-Host:18080/v1/chat/completions
#   Web-Browser: http://Pascal-Host:18080/
#   Health:      http://Pascal-Host:18080/health
#
# Verwendung:
#   ./scripts/run-gemma4-26b-a4b-mtp-Pascal-Host-server.sh
#
# Env-Vars (optional, mit Defaults):
#   GGML_CUDA_REGISTER_HOST=1    — Memory Pinning (cudaHostRegister)
#   GGML_SCHED_PREFETCH_EXPERTS=1 — Async Expert Prefetch (3 slots)
#
# Benchmark-Ergebnisse (GTX 1070, 8GB VRAM):
#   pp512:  538 t/s, tg128: 21.3 t/s (ohne MTP)
#   pp512:  538 t/s, tg128: 31.9 t/s (mit MTP, 100% Akzeptanz)
#
# Beispiel API-Request:
#   curl http://Pascal-Host:18080/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"gemma-4","messages":[{"role":"user","content":"Hallo!"}],"max_tokens":100}'
# ============================================================================

set -euo pipefail

# thecodacus MoE-Optimierungen (Pinning + Prefetch)
export GGML_CUDA_REGISTER_HOST="${GGML_CUDA_REGISTER_HOST:-1}"
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"

# Pascal-Host-spezifische Pfade
SERVER="/path/to/fukuro-llama-cpp-turboquant/build/bin/llama-server"
MAIN="/data/modelle/gemma-4-26B-A4B-it/google_gemma-4-26B-A4B-it-IQ4_NL.gguf"
DRAFT="/data/modelle/gemma-4-26B-A4B-it/drafts/gemma-4-26b-a4b-it-assistant.Q4_K_M.gguf"

# Konfiguration
CTX="${CTX:-8192}"
NGL="${NGL:-999}"
NGL_DRAFT="${NGL_DRAFT:-999}"
N_CPU_MOE="${N_CPU_MOE:-20}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
FA="${FA:-on}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
PARALLEL="${PARALLEL:-1}"

# MTP Draft Parameter
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"

echo "=== Gemma-4 26B-A4B MTP-Server auf Pascal-Host (Pinning+Prefetch) ==="
echo "Target: $MAIN"
echo "Draft:  $DRAFT"
echo "ctx=${CTX}, ngl=${NGL}, n-cpu-moe=${N_CPU_MOE}"
echo "Pinning: GGML_CUDA_REGISTER_HOST=${GGML_CUDA_REGISTER_HOST}"
echo "Prefetch: GGML_SCHED_PREFETCH_EXPERTS=${GGML_SCHED_PREFETCH_EXPERTS}"
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
    --n-cpu-moe "$N_CPU_MOE" \
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
