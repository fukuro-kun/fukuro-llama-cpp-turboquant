#!/bin/sh
# S1/S2 A/B: Expert-Prefetch-Arme fuer Flash-Next Dual-GPU-Offload.
# Synthetischer pp8192: llama-bench, -b/-ub 8192, q8_0/q8_0 KV,
# layer-split, MoE-Cache aus (wie die B-Benchmarks).
#
# Aufruf:
#   FLASHNEXT_MODEL=/pfad/modell.gguf s1-prefetch-ab.sh <build-bin-dir> <arm> [reps]
#
# Arme:
#   noprefetch  - komplett aus (Kontrolle)
#   base        - PREFETCH_EXPERTS=1 (3 Slots, Depth-1)
#   all3        - +PREFETCH_ALL (3 Slots)
#   all8        - +PREFETCH_ALL + 8 Slots   <- Bestkonfiguration ~508 t/s
#   allonly     - PREFETCH_ALL ohne Slot-Ring (generische Volluploads)
#   look8       - all8 + PREFETCH_LOOKAHEAD (Cross-Split-Staging)
#   spread3     - OFFLOAD_SPREAD + LOOKAHEAD + 3 Slots (VRAM-Limit)
#   ncmoe<N>    - wie all8, aber -ncmoe N statt 99 (S2-Residenz-Sweep)
#
# Env-Overrides: REHOST=0|1 (Default 1), MOE_CACHE (Default 0),
#   CHUNK_MB (Default: Build-Default), EXTRA_NCMOE.

set -u
BIN="${1:?usage: s1-prefetch-ab.sh <build-bin-dir> <arm> [reps]}"
ARM="${2:?arm missing}"
REPS="${3:-3}"
MODEL="${FLASHNEXT_MODEL:?set FLASHNEXT_MODEL=/pfad/zum/modell.gguf}"
OUT="${FLASHNEXT_OUT:-./s1-ab}-${ARM}-$(date +%Y%m%d-%H%M%S).json"
LOG="${OUT%.json}.log"

REHOST="${REHOST:-1}"
MOE_CACHE="${MOE_CACHE:-0}"
NCMOE=99
EXTRA_SLOTS=""

unset GGML_SCHED_PREFETCH_EXPERTS GGML_SCHED_PREFETCH_SLOTS \
      GGML_SCHED_PREFETCH_ALL GGML_SCHED_PREFETCH_LOOKAHEAD \
      GGML_SCHED_OFFLOAD_SPREAD GGML_SCHED_PREFETCH_CHUNK_MB || true
export GGML_CUDA_MOE_CACHE="$MOE_CACHE"

case "$ARM" in
  noprefetch) ;;
  base)    export GGML_SCHED_PREFETCH_EXPERTS=1 ;;
  all3)    export GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_ALL=1 ;;
  all8)    export GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_ALL=1 GGML_SCHED_PREFETCH_SLOTS=8 ;;
  allonly) export GGML_SCHED_PREFETCH_ALL=1 ;;
  look8)   export GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_ALL=1 GGML_SCHED_PREFETCH_SLOTS=8 GGML_SCHED_PREFETCH_LOOKAHEAD=1 ;;
  spread3) export GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_ALL=1 GGML_SCHED_PREFETCH_SLOTS=3 GGML_SCHED_PREFETCH_LOOKAHEAD=1 GGML_SCHED_OFFLOAD_SPREAD=1 ;;
  ncmoe*)
    NCMOE="${ARM#ncmoe}"
    EXTRA_SLOTS="${EXTRA_SLOTS:-4}"
    export GGML_SCHED_PREFETCH_EXPERTS=1 GGML_SCHED_PREFETCH_ALL=1 GGML_SCHED_PREFETCH_SLOTS="$EXTRA_SLOTS"
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

[ "$REHOST" = 1 ] && export GGML_CUDA_REGISTER_HOST=1
[ -n "${CHUNK_MB:-}" ] && export GGML_SCHED_PREFETCH_CHUNK_MB="$CHUNK_MB"

echo "arm=$ARM ncmoe=$NCMOE reghost=$REHOST" | tee "$LOG"
env | grep -E 'GGML_(SCHED|CUDA)' | sort | tee -a "$LOG"

"$BIN/llama-bench" \
  -m "$MODEL" \
  -ngl 99 -ncmoe "$NCMOE" \
  -fa 1 -ctk q8_0 -ctv q8_0 \
  -dev CUDA0/CUDA1 --split-mode layer \
  -b 8192 -ub 8192 \
  -p 8192 -n 0 -r "$REPS" \
  -o json >> "$OUT" 2>>"$LOG"
rc=$?
echo "rc=$rc" | tee -a "$LOG"
grep -E 'avg_ts|samples_ts' "$OUT" 2>/dev/null | tee -a "$LOG"
exit $rc
