#!/usr/bin/env bash
# ============================================================================
# Vulkan KV-Cache Benchmark: turbo3/3 vs turbo4/3 vs turbo4/4 vs f16
# ============================================================================
# Multi-Prompt (-p 512, 2048, 4096, 8192, -n 64), verschiedene KV-Cache Typen
# Modell: Gemma-4 26B-A4B IQ4_NL (MoE, 4B aktive Parameter)
# Backend: Vulkan (AMD RDNA3 APU, UMA)
#
# Hypothese: K=turbo4 (mit FlashAttention) könnte schneller sein als
# K=turbo3 (ohne FA, scalar fallback) trotz geringerer Kompression.
# ERGEBNIS: Hypothese WIDERLEGT — turbo3/3 ist konsistent am schnellsten.
#
# WICHTIG: Vor jedem Test wird GTT freigegeben (killall + sleep).
#          Auf UMA-Systemen können parallele Prozesse OOM triggern.
#
# Usage: MODEL_PATH=/path/to/model.gguf bash bench-vulkan-kv-cache.sh
#    or: bash bench-vulkan-kv-cache.sh /path/to/model.gguf
# ============================================================================

set -euo pipefail

BENCH="${LLAMA_BENCH:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)/build/bin/llama-bench}"
MODEL="${1:-${MODEL_PATH:?MODEL_PATH env var oder Argument 1 required — Pfad zum 26B-A4B GGUF}}"
OUTPUT="${OUTPUT:-/tmp/bench-vulkan-kv-cache-results.csv}"
LOCK="/tmp/bench-vulkan-kv-cache.lock"

# Lock-File
if [ -f "$LOCK" ]; then
    echo "ERROR: Benchmark läuft bereits (Lock: $LOCK)" >&2
    exit 1
fi
trap "rm -f '$LOCK'" EXIT
touch "$LOCK"

# Prüfe Binary und Modell
if [ ! -f "$BENCH" ]; then
    echo "ERROR: llama-bench nicht gefunden: $BENCH" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Modell nicht gefunden: $MODEL" >&2
    exit 1
fi

# GTT komplett freigeben vor Start
killall -9 llama-bench llama-server llama-cli 2>/dev/null || true
sleep 10

# CSV Header
echo "timestamp,ctk,ctv,ctx,pp_tps,tg_tps" > "$OUTPUT"

# Konfiguration
KV_COMBOS="turbo3:turbo3 turbo3:turbo4 turbo4:turbo4 f16:f16"
PROMPT_SIZES="512 2048 4096 8192"
NGL=99
FA="on"
REPS=2
COOLDOWN=8  # Sekunden zwischen Tests (GTT freigeben)

echo "=== Vulkan KV-Cache Benchmark ==="
echo "Modell: $MODEL"
echo "Prompts: ${PROMPT_SIZES}, Gen: 64, Reps: $REPS"
echo "KV-Kombinationen: $KV_COMBOS"
echo "Output: $OUTPUT"
echo "=================================================="

for kv_combo in $KV_COMBOS; do
    CTK="${kv_combo%%:*}"
    CTV="${kv_combo##*:}"

    for ctx in $PROMPT_SIZES; do
        echo ""
        echo "--- Test: K=$CTK V=$CTV pp=$ctx ---"
        echo "  Start: $(date)"

        # GTT zwischen Tests freigeben
        killall -9 llama-bench 2>/dev/null || true
        sleep "$COOLDOWN"

        # Benchmark ausführen
        result=$("$BENCH" \
            -m "$MODEL" \
            -p "$ctx" \
            -n 64 \
            -ngl "$NGL" \
            -ctk "$CTK" \
            -ctv "$CTV" \
            -fa "$FA" \
            -r "$REPS" \
            -o csv 2>/dev/null) || {
            echo "  ERROR/OOM bei K=$CTK V=$CTV pp=$ctx" >&2
            echo "$(date -Iseconds),$CTK,$CTV,$ctx,ERROR,ERROR" >> "$OUTPUT"
            sync
            sleep 15
            continue
        }

        # CSV parsen: 2 Datenzeilen (Header überspringen)
        # Spalten (1-indexed): n_prompt=34, n_gen=35, avg_ts=40
        pp_tps=""
        tg_tps=""
        while IFS= read -r line; do
            if [[ "$line" == *"avg_ts"* ]]; then
                continue
            fi
            n_prompt=$(echo "$line" | awk -F',' '{print $34}')
            n_gen=$(echo "$line" | awk -F',' '{print $35}')
            avg_ts=$(echo "$line" | awk -F',' '{print $40}')

            if [[ "$n_gen" == "\"0\"" ]] || [[ "$n_gen" == "0" ]]; then
                pp_tps="$avg_ts"
            elif [[ "$n_prompt" == "\"0\"" ]] || [[ "$n_prompt" == "0" ]]; then
                tg_tps="$avg_ts"
            fi
        done <<< "$result"

        echo "  pp${ctx}: ${pp_tps} t/s | tg64: ${tg_tps} t/s"
        echo "  Ende: $(date)"

        echo "$(date -Iseconds),$CTK,$CTV,$ctx,$pp_tps,$tg_tps" >> "$OUTPUT"
        sync
    done
done

echo ""
echo "=== BENCHMARK COMPLETE ==="
echo "Results: $OUTPUT"
echo ""
cat "$OUTPUT"
