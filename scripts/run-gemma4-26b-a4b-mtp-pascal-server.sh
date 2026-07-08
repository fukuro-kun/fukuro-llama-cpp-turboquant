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
#   ./scripts/run-gemma4-26b-a4b-mtp-pascal-server.sh
#
# Env-Vars (optional, mit Defaults):
#   GGML_CUDA_REGISTER_HOST=1    — Memory Pinning (cudaHostRegister)
#   GGML_SCHED_PREFETCH_EXPERTS=1 — Async Expert Prefetch (3 slots)
#
# KV-Cache: K=turbo3 (stärker komprimiert, ~5.1x), V=turbo4 (schonender, ~3.8x)
# Values sind empfindlicher (tragen Attention-Output), Keys robuster (nur QK^T-Scores)
# Default ctx=163840 (160k) — maximaler Kontext bei 8GB VRAM (8110 MiB belegt)
#
# MTP (Speculative Decoding) ist standardmäßig AUS.
# Tests zeigen: MTP ist bei JEDEM Kontext langsamer als no-MTP für realistische
# Generierung (Essay, Chat, Analyse) auf der GTX 1070. Der Speedup bei trivialen
# Prompts ("Count 1-100": 31.9 t/s) entsteht durch ~100% Acceptance Rate — bei
# kreativen Texten werden Drafts meist abgelehnt, der Draft-Overhead dominiert.
# Zusätzlich crasht ctx=65536 (2^16) reproduzierbar mit MTP (Integer-Overflow-Bug).
# MTP aktivieren nur für triviale Tasks mit ctx ≤ 32k: MTP=1 CTX=32768 ./...
#
# Benchmark-Ergebnisse (GTX 1070, 8GB VRAM, Essay-Prompt 1200 tok):
#   ctx=128k, MTP aus: 16.03 t/s, 7498 MiB VRAM
#   ctx=128k, MTP an:  15.68 t/s, 7794 MiB VRAM (-2%)
#   ctx=32k,  MTP aus: 16.49 t/s, 6702 MiB VRAM
#   ctx=32k,  MTP an:  14.49 t/s, 7210 MiB VRAM (-14%)
#   ctx=160k, MTP aus: ~16 t/s, 8110 MiB VRAM (max. Kontext)
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
CTX="${CTX:-163840}"
NGL="${NGL:-999}"
NGL_DRAFT="${NGL_DRAFT:-999}"
N_CPU_MOE="${N_CPU_MOE:-20}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
FA="${FA:-on}"
# KV-Cache: K=turbo3 (stärker komprimiert, robuster), V=turbo4 (schonender, empfindlicher)
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo4}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
PARALLEL="${PARALLEL:-1}"

# MTP (Speculative Decoding) — standardmäßig AUS (siehe Header-Kommentar)
# MTP=1 aktiviert MTP, empfohlen nur für triviale Tasks mit ctx ≤ 32k
MTP="${MTP:-0}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"

echo "=== Gemma-4 26B-A4B Server auf Pascal-Host (Pinning+Prefetch) ==="
echo "Target: $MAIN"
echo "ctx=${CTX}, ngl=${NGL}, n-cpu-moe=${N_CPU_MOE}, cache=${CTK}/${CTV}"
echo "Pinning: GGML_CUDA_REGISTER_HOST=${GGML_CUDA_REGISTER_HOST}"
echo "Prefetch: GGML_SCHED_PREFETCH_EXPERTS=${GGML_SCHED_PREFETCH_EXPERTS}"
if [[ "$MTP" == "1" ]]; then
    echo "MTP: AN (n_max=${SPEC_DRAFT_N_MAX}) — Draft: $DRAFT"
else
    echo "MTP: AUS (Default — MTP ist bei realistischer Generierung langsamer)"
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
        -ngld "$NGL_DRAFT" \
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
        --metrics \
        --slots \
        --log-timestamps \
        --log-prefix \
        "$@"
fi
