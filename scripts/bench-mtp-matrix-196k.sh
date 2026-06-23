#!/bin/bash
# ============================================================================
# bench-mtp-matrix-196k.sh — Vollstaendige MTP-Matrix @ ctx=196608
# ============================================================================
# Verbessert nach InferenzQuelle-Best Practices:
# - GPU-Bereinigung vor jedem Test (pkill + sleep)
# - Datei-Pruefung vor dem Test
# - Output in Variable (sauberes Parsing)
# - sync nach jedem CSV-Eintrag
# - Exit 124 (Timeout) als OK behandeln
#
# Vergleichbar mit der 2048-Matrix in Trilium (sJFq491sG3GS):
# - 9 Targets: IQ4_XS, IQ4_NL, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K
# - 12 Drafts: IQ3_S, IQ3_M, IQ4_XS, IQ4_NL, Q3_K_S, Q3_K_M, Q3_K_L,
#              Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K
# - OHNE: F16 (zu gross fuer Draft), Q8_0 (nicht in 2048-Matrix)
#
# Verwendung:
#   ./scripts/bench-mtp-matrix-196k.sh
#
# Ausgabe:
#   /tmp/mtp_matrix_196608_<TIMESTAMP>.log
#   /tmp/mtp_matrix_196608_<TIMESTAMP>.csv
#
# Geschätzte Dauer: 9 Targets × 12 Drafts × ~5 Min = ~9 Stunden
# ============================================================================

set -euo pipefail

LLAMA_CLI="${LLAMA_CLI:-$(dirname "$0")/../build/bin/llama-cli}"
BASE_DIR="${BASE_DIR:-/pfad/zu/modellen/gemma-4-12b-it}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG="/tmp/mtp_matrix_196608_${TIMESTAMP}.log"
CSV="/tmp/mtp_matrix_196608_${TIMESTAMP}.csv"

CTX=196608
NG=20
TIMEOUT=1200

echo "=== MTP-Matrix Gemma-4 12B @ ctx=${CTX} ===" | tee -a "$LOG"
echo "Start: $(date -Iseconds)" | tee -a "$LOG"
echo "LLAMA: $LLAMA_CLI" | tee -a "$LOG"
echo "Base:  $BASE_DIR" | tee -a "$LOG"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "$LOG"
echo "VRAM:  $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MiB frei" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

# 9 Targets (wie 2048-Matrix)
TARGETS=(
  "$BASE_DIR/gemma-4-12b-it-IQ4_XS.gguf"
  "$BASE_DIR/gemma-4-12b-it-IQ4_NL.gguf"
  "$BASE_DIR/gemma-4-12b-it-Q3_K_M.gguf"
  "$BASE_DIR/gemma-4-12b-it-Q3_K_L.gguf"
  "$BASE_DIR/gemma-4-12b-it-Q4_K_S.gguf"
  "$BASE_DIR/gemma-4-12b-it-Q4_K_M.gguf"
  "$BASE_DIR/gemma-4-12b-it-Q5_K_S.gguf"
  "$BASE_DIR/gemma-4-12b-it-Q5_K_M.gguf"
  "$BASE_DIR/gemma-4-12b-it-Q6_K.gguf"
)

# 12 Drafts (wie 2048-Matrix, OHNE F16 und Q8_0)
DRAFTS=(
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.IQ3_S.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.IQ3_M.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.IQ4_XS.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.IQ4_NL.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q3_K_S.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q3_K_M.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q3_K_L.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q4_K_S.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q4_K_M.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q5_K_S.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q5_K_M.gguf"
  "$BASE_DIR/drafts/gemma-4-12b-it-assistant.Q6_K.gguf"
)

T_COUNT=${#TARGETS[@]}
D_COUNT=${#DRAFTS[@]}
TOTAL=$((T_COUNT * D_COUNT))

echo "Targets: $T_COUNT | Drafts: $D_COUNT | Kombinationen: $TOTAL" | tee -a "$LOG"
echo "Drafts:  IQ3_S, IQ3_M, IQ4_XS, IQ4_NL, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

# CSV Header
echo "target_quant,draft_quant,status,exit_code,gen_t_s,prompt_t_s,accept_rate,vram_before,vram_after,timestamp" > "$CSV"

CURRENT=0

for TARGET in "${TARGETS[@]}"; do
    TNAME=$(basename "$TARGET")
    TQ=$(echo "$TNAME" | sed 's/gemma-4-12b-it-//;s/\.gguf//')

    for DRAFT in "${DRAFTS[@]}"; do
        DNAME=$(basename "$DRAFT")
        DQ=$(echo "$DNAME" | sed 's/.*assistant\.\([^.]*\)\.gguf/\1/')

        CURRENT=$((CURRENT + 1))
        echo -n "[$CURRENT/$TOTAL] $TQ × $DQ ... " | tee -a "$LOG"

        # --- Datei-Pruefung ---
        if [ ! -f "$TARGET" ]; then
            echo "MISSING_TARGET" | tee -a "$LOG"
            echo "$TQ,$DQ,MISSING_TARGET,0,,,,,,$(date -Iseconds)" >> "$CSV"
            sync
            continue
        fi
        if [ ! -f "$DRAFT" ]; then
            echo "MISSING_DRAFT" | tee -a "$LOG"
            echo "$TQ,$DQ,MISSING_DRAFT,0,,,,,,$(date -Iseconds)" >> "$CSV"
            sync
            continue
        fi

        # --- GPU-Bereinigung vor jedem Test ---
        killall -9 llama-cli 2>/dev/null || true
        sleep 3

        VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

        # --- Benchmark mit Output in Variable (sauberes Parsing) ---
        set +e
        output=$(timeout "$TIMEOUT" "$LLAMA_CLI" \
            -m "$TARGET" \
            --model-draft "$DRAFT" \
            --spec-type mtp \
            --single-turn \
            -n "$NG" \
            --ctx-size "$CTX" \
            --cache-type-k turbo4 \
            --cache-type-v turbo3 \
            -ngl 99 \
            --no-display-prompt \
            --prompt "Hello world" 2>&1)
        EXIT=$?
        set -e

        VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

        GEN_T_S=""
        PROMPT_T_S=""
        ACCEPT=""
        STATUS=""

        if [ $EXIT -eq 0 ] || [ $EXIT -eq 124 ]; then
            # Extrahiere t/s: "[ Prompt: X t/s | Generation: Y t/s ]"
            timing=$(echo "$output" | grep -oP '\[ Prompt: [0-9,\.]+ t/s \| Generation: [0-9,\.]+ t/s \]' | tail -1 || echo "")

            if [ -n "$timing" ]; then
                PROMPT_T_S=$(echo "$timing" | grep -oP 'Prompt: \K[0-9,\.]+' | sed 's/,/./')
                GEN_T_S=$(echo "$timing" | grep -oP 'Generation: \K[0-9,\.]+' | sed 's/,/./')

                # draft: n_draft = 19, n_accept = 14, n_reject = 5, accept_rate = 73.68%
                draft_line=$(echo "$output" | grep -oP 'draft:.*' | tail -1 || echo "")
                ACCEPT=$(echo "$draft_line" | grep -oP 'accept_rate = \K[0-9,\.]+' | sed 's/,/./' || echo "")

                if [ $EXIT -eq 0 ]; then
                    STATUS="OK"
                    echo "OK | prompt=${PROMPT_T_S} t/s | gen=${GEN_T_S} t/s | accept=${ACCEPT}%" | tee -a "$LOG"
                else
                    STATUS="TIMEOUT"
                    echo "TIMEOUT | prompt=${PROMPT_T_S} t/s | gen=${GEN_T_S} t/s" | tee -a "$LOG"
                fi
            else
                STATUS="NO_TIMING"
                echo "NO_TIMING (keine Performance-Daten im Output)" | tee -a "$LOG"
            fi
        else
            STATUS="FAIL"
            echo "FAIL (exit=$EXIT)" | tee -a "$LOG"
        fi

        echo "$TQ,$DQ,$STATUS,$EXIT,\"$GEN_T_S\",\"$PROMPT_T_S\",\"$ACCEPT\",$VRAM_BEFORE,$VRAM_AFTER,$(date -Iseconds)" >> "$CSV"
        sync
        sleep 2
    done
    echo "--- $TQ komplett ---" | tee -a "$LOG"
done

echo "=== KOMPLETT ===" | tee -a "$LOG"
echo "Ende: $(date -Iseconds)" | tee -a "$LOG"
echo "Log:  $LOG" | tee -a "$LOG"
echo "CSV:  $CSV" | tee -a "$LOG"

OK_COUNT=$(grep -c ',OK,' "$CSV" || echo 0)
FAIL_COUNT=$(grep -c ',FAIL,' "$CSV" || echo 0)
TO_COUNT=$(grep -c ',TIMEOUT,' "$CSV" || echo 0)
NT_COUNT=$(grep -c ',NO_TIMING,' "$CSV" || echo 0)
echo "OK: $OK_COUNT | TIMEOUT: $TO_COUNT | NO_TIMING: $NT_COUNT | FAIL: $FAIL_COUNT | TOTAL: $TOTAL" | tee -a "$LOG"
