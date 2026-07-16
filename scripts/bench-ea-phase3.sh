#!/bin/bash
# EA Phase 3 Benchmark — Vergleich EA disabled vs enabled bei verschiedenen Kontextlängen
# Backend: CUDA (Pascal GTX 1070), turbo3/turbo4 KV, -fa on
#
# Usage:
#   MODEL=/path/to/model.gguf EA_RATIO=0.3 ./scripts/bench-ea-phase3.sh
#
# Required env vars:
#   MODEL       — path to GGUF model file
# Optional env vars:
#   EA_RATIO    — pruning ratio (default: 0.3)
#   BIN         — path to llama-bench (default: ./build/bin/llama-bench)
#   NGL         — GPU layers (default: 999)
#   NCPU_MOE    — CPU MoE experts (default: 20)
#   CTK, CTV    — KV cache types (default: turbo3/turbo4)
#   REPS        — repetitions per data point (default: 1)
#   THREADS     — CPU threads (default: 4)
#
# Output: Markdown-Tabelle auf stdout, metadata on stderr

set -e

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL env var required (path to GGUF)" >&2
    exit 1
fi

EA_RATIO="${EA_RATIO:-0.3}"
BIN="${BIN:-./build/bin/llama-bench}"
NGL="${NGL:-999}"
NCPU_MOE="${NCPU_MOE:-20}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-turbo4}"
REPS="${REPS:-1}"
THREADS="${THREADS:-4}"

# Context lengths to test (pp,tg pairs)
PG_ARGS="-pg 4096,128 -pg 16384,128 -pg 65536,128 -pg 131072,128"

echo "# EA Phase 3 Benchmark — $(basename $MODEL)" >&2
echo "# GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')" >&2
echo "# KV: $CTK/$CTV | -ngl $NGL | reps=$REPS" >&2
echo "" >&2

# --- Baseline (EA disabled) ---
echo "## Baseline (EA disabled)" >&2
echo "" >&2
$BIN -m "$MODEL" -ngl $NGL --n-cpu-moe $NCPU_MOE -ctk $CTK -ctv $CTV -fa on \
    $PG_ARGS -r $REPS -t $THREADS -o md 2>/dev/null

echo "" >&2
echo "" >&2

# --- EA enabled ---
echo "## EA enabled (ratio=$EA_RATIO)" >&2
echo "" >&2
LLAMA_ARG_EA_RATIO=$EA_RATIO \
$BIN -m "$MODEL" -ngl $NGL --n-cpu-moe $NCPU_MOE -ctk $CTK -ctv $CTV -fa on \
    $PG_ARGS -r $REPS -t $THREADS -o md 2>/dev/null

echo "" >&2
echo "# Benchmark complete — $(date)" >&2
