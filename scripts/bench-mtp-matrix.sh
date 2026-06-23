#!/bin/bash
# ============================================================================
# bench-mtp-matrix.sh — MTP-Matrix Benchmark fuer Gemma-4 Modelle
# ============================================================================
# Zweck: Systematisches Testen aller Draft-Quantisierungen gegen ein Target
#        bei definierbarer Kontextgroesse. Erzeugt eine Matrix von
#        Generation t/s und Akzeptanzraten.
#
# Verwendung:
#   ./scripts/bench-mtp-matrix.sh <TARGET_GGUF> <DRAFT_DIR> [CTX_SIZE] [N_TOKENS]
#
# Beispiele:
#   # 12B @ ctx=196608 (langsam, ~1-2h)
#   ./scripts/bench-mtp-matrix.sh \
#     ~/modelle/gemma-4-12b-it-IQ4_NL.gguf \
#     ~/modelle/gemma-4-12b-it/drafts/ \
#     196608 20
#
#   # 12B @ ctx=2048 (schnell, ~5min)
#   ./scripts/bench-mtp-matrix.sh \
#     ~/modelle/gemma-4-12b-it-IQ4_NL.gguf \
#     ~/modelle/gemma-4-12b-it/drafts/ \
#     2048 20
#
#   # 31B @ ctx=8192
#   ./scripts/bench-mtp-matrix.sh \
#     ~/modelle/gemma-4-31B-it-Q4_K_M.gguf \
#     ~/modelle/gemma-4-31B-it/drafts/ \
#     8192 20
#
# Ausgabe:
#   /tmp/mtp_matrix_<CTX>_<TIMESTAMP>.log  — Vollstaendiges Log
#   /tmp/mtp_matrix_<CTX>_<TIMESTAMP>.csv   — CSV fuer Import/Weiterverarbeitung
#
# Abhaengigkeiten:
#   - llama-cli (gebaut mit MTP-Support)
#   - nvidia-smi (fuer VRAM-Monitoring)
#   - timeout (coreutils)
#
# Autor: fukuro + KI-Agent
# Datum: 2026-06-17
# ============================================================================

set -euo pipefail

# --- Konfiguration ------------------------------------------------------------
TARGET="${1:-}"
DRAFT_DIR="${2:-}"
CTX="${3:-2048}"
NG="${4:-20}"
TIMEOUT_SEC="${5:-600}"  # 10 Minuten pro Draft (Prompt-Verarbeitung bei grossem Kontext)

LLAMA_CLI="${LLAMA_CLI:-$(dirname "$0")/../build/bin/llama-cli}"
PROMPT="${PROMPT:-Hello world}"
CACHE_K="${CACHE_K:-turbo4}"
CACHE_V="${CACHE_V:-turbo3}"
NGL="${NGL:-99}"

# --- Validierung -------------------------------------------------------------
if [ -z "$TARGET" ] || [ -z "$DRAFT_DIR" ]; then
    echo "Usage: $0 <TARGET_GGUF> <DRAFT_DIR> [CTX_SIZE] [N_TOKENS] [TIMEOUT_SEC]"
    echo ""
    echo "Beispiel (ctx=196608):"
    echo "  $0 ~/modelle/gemma-4-12b-it.gguf ~/modelle/gemma-4-12b-it/drafts/ 196608 20"
    exit 1
fi

if [ ! -f "$TARGET" ]; then
    echo "ERROR: Target nicht gefunden: $TARGET"
    exit 1
fi

if [ ! -d "$DRAFT_DIR" ]; then
    echo "ERROR: Draft-Verzeichnis nicht gefunden: $DRAFT_DIR"
    exit 1
fi

if [ ! -x "$LLAMA_CLI" ]; then
    echo "ERROR: llama-cli nicht ausfuehrbar: $LLAMA_CLI"
    echo "Setze LLAMA_CLI=/pfad/zu/llama-cli oder baue das Projekt."
    exit 1
fi

# --- Logging ------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG="/tmp/mtp_matrix_${CTX}_${TIMESTAMP}.log"
CSV="/tmp/mtp_matrix_${CTX}_${TIMESTAMP}.csv"

# CSV-Header
echo "draft_quant,status,exit_code,gen_toks,gen_t_s,prompt_t_s,accept_rate,vram_before,vram_after,host,gpu,timestamp" > "$CSV"

# --- Header -------------------------------------------------------------------
echo "=== MTP-Matrix Benchmark ===" | tee -a "$LOG"
echo "Target:  $(basename "$TARGET")" | tee -a "$LOG"
echo "Drafts:  $DRAFT_DIR" | tee -a "$LOG"
echo "Context: $CTX" | tee -a "$LOG"
echo "Tokens:  $NG" | tee -a "$LOG"
echo "LLAMA:   $LLAMA_CLI" | tee -a "$LOG"
echo "Cache:   K=$CACHE_K, V=$CACHE_V" | tee -a "$LOG"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "$LOG"
echo "VRAM:    $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MiB frei" | tee -a "$LOG"
echo "Host:    $(hostname)" | tee -a "$LOG"
echo "Datum:   $(date -Iseconds)" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

# --- Drafts finden ------------------------------------------------------------
DRAFTS=$(find "$DRAFT_DIR" -maxdepth 1 -name '*assistant*.gguf' -type f | sort)

if [ -z "$DRAFTS" ]; then
    echo "ERROR: Keine Drafts (*.gguf) in $DRAFT_DIR gefunden" | tee -a "$LOG"
    exit 1
fi

DRAFT_COUNT=$(echo "$DRAFTS" | wc -l)
echo "Gefundene Drafts: $DRAFT_COUNT" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

# --- Matrix durchlaufen -------------------------------------------------------
for DRAFT in $DRAFTS; do
    DNAME=$(basename "$DRAFT")
    # Quantisierung extrahieren: gemma-4-12b-it-assistant.IQ4_NL.gguf -> IQ4_NL
    Q=$(echo "$DNAME" | sed -n 's/.*assistant\.\([^.]*\)\.gguf/\1/p')
    if [ -z "$Q" ]; then
        Q="UNKNOWN"
    fi

    echo -n "[$Q] " | tee -a "$LOG"

    VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    # llama-cli ausfuehren mit Timeout
    set +e
    timeout "$TIMEOUT_SEC" "$LLAMA_CLI" \
        -m "$TARGET" \
        --model-draft "$DRAFT" \
        --spec-type mtp \
        --single-turn \
        -n "$NG" \
        --ctx-size "$CTX" \
        --cache-type-k "$CACHE_K" \
        --cache-type-v "$CACHE_V" \
        -ngl "$NGL" \
        --no-display-prompt \
        --prompt "$PROMPT" >> "$LOG" 2>&1
    EXIT=$?
    set -e

    VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    # Ergebnis parsen
    GEN_TOKS=""
    GEN_T_S=""
    PROMPT_T_S=""
    ACCEPT=""

    if [ $EXIT -eq 0 ]; then
        # generate: n_tokens = 20, t_generation = 0.50 s, t/s = 40.23
        GEN_LINE=$(grep -oP 'generate:.*' "$LOG" | tail -1)
        GEN_TOKS=$(echo "$GEN_LINE" | grep -oP 'n_tokens = \K[0-9]+' || echo "")
        GEN_T_S=$(echo "$GEN_LINE" | grep -oP 't/s = \K[0-9.]+' || echo "")

        # prompt: n_tokens = 3, t_prompt = 0.01 s, t/s = 300.00
        PROMPT_LINE=$(grep -oP 'prompt:.*' "$LOG" | tail -1)
        PROMPT_T_S=$(echo "$PROMPT_LINE" | grep -oP 't/s = \K[0-9.]+' || echo "")

        # draft: n_draft = 19, n_accept = 14, n_reject = 5, accept_rate = 73.68%
        DRAFT_LINE=$(grep -oP 'draft:.*' "$LOG" | tail -1)
        ACCEPT=$(echo "$DRAFT_LINE" | grep -oP 'accept_rate = \K[0-9.]+' || echo "")

        echo "OK | gen=${GEN_T_S} t/s | accept=${ACCEPT}% | VRAM=${VRAM_BEFORE}→${VRAM_AFTER}" | tee -a "$LOG"
    elif [ $EXIT -eq 124 ]; then
        echo "TIMEOUT (${TIMEOUT_SEC}s) | VRAM=${VRAM_BEFORE}→${VRAM_AFTER}" | tee -a "$LOG"
    else
        echo "FAIL (exit=$EXIT) | VRAM=${VRAM_BEFORE}→${VRAM_AFTER}" | tee -a "$LOG"
    fi

    # CSV-Zeile
    echo "$Q,$([ $EXIT -eq 0 ] && echo OK || ([ $EXIT -eq 124 ] && echo TIMEOUT || echo FAIL)),$EXIT,\"$GEN_TOKS\",\"$GEN_T_S\",\"$PROMPT_T_S\",\"$ACCEPT\",$VRAM_BEFORE,$VRAM_AFTER,$(hostname),$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1),$TIMESTAMP" >> "$CSV"

    echo "---" | tee -a "$LOG"
    sleep 2
done

# --- Fazit --------------------------------------------------------------------
echo "=== KOMPLETT ===" | tee -a "$LOG"
echo "Log:  $LOG" | tee -a "$LOG"
echo "CSV:  $CSV" | tee -a "$LOG"
echo "Drafts getestet: $DRAFT_COUNT" | tee -a "$LOG"

# Zusammenfassung anzeigen
echo ""
echo "=== ZUSAMMENFASSUNG ==="
echo ""
column -t -s, "$CSV" | head -20
