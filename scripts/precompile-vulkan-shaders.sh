#!/usr/bin/env bash
# Precompile aller Vulkan-Shader-Varianten für llama-server (turbo3/3 + mmproj).
#
# Kompiliert systematisch alle FlashAttention-Pipeline-Varianten durch:
#   - Generation (N=1 → FA_SCALAR, Br=1)
#   - GQA-Generation (N=2 → FA_COOPMAT1, Br=16)
#   - Prompt-Processing (N>8 → FA_COOPMAT1, Br=16, with mask)
#   - Large-Prompt-Processing (→ mask_opt pipeline)
#   - Vision-Request (→ mmproj-spezifische Pipelines)
#   - Verschiedene Prompt-Längen (→ aligned/unaligned Varianten)
#
# Cache-Persistenz:
#   - GGML_VK_CACHE_DIR: VkPipelineCache-Blob (geschrieben bei SIGTERM-Shutdown)
#   - MESA_SHADER_CACHE_DIR: Mesa Shader-ISA-Cache (async, flushed bei vkDestroyDevice)
#   - Beide Caches werden nach dem Run gesichert (tar-Backup).
#
# WICHTIG: Sauberer Shutdown mit SIGTERM + wait, niemals SIGKILL!
#          Bei SIGKILL geht der VkPipelineCache verloren.
#
# Usage:
#   bash scripts/precompile-vulkan-shaders.sh
#   BACKUP_DIR=/path/to/backup bash scripts/precompile-vulkan-shaders.sh
#
# Environment defaults (alle überschreibbar):
#   PORT=18099                  — Port für den Precompile-Server
#   BACKUP_DIR                  — Backup-Ziel für Cache-Snapshot
#   GGML_VK_CACHE_DIR           — Vulkan Pipeline-Cache-Verzeichnis
#   MESA_SHADER_CACHE_DIR       — Mesa Shader-Cache-Verzeichnis
#   MESA_SHADER_CACHE_MAX_SIZE  — Mesa Cache-Größenlimit

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-${ROOT}/build/bin/llama-server}"
# Modell- und Cache-Pfade müssen über Umgebungsvariablen gesetzt werden (keine hartkodierten Defaults)
MAIN="${MAIN_GGUF:?MAIN_GGUF must be set — path to main model GGUF}"
MMPROJ="${MMPROJ_GGUF:?MMPROJ_GGUF must be set — path to mmproj GGUF}"
PORT="${PORT:-18099}"
HOST="${HOST:-127.0.0.1}"

# Cache-Verzeichnisse (Defaults via XDG_CACHE_HOME oder ~/.cache)
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}"
export GGML_VK_CACHE_DIR="${GGML_VK_CACHE_DIR:-${CACHE_BASE}/ggml-vk-pipeline-cache}"
export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-${CACHE_BASE}/mesa-shader-cache}"
export MESA_SHADER_CACHE_MAX_SIZE="${MESA_SHADER_CACHE_MAX_SIZE:-2G}"
# NIR-Cache beschleunigt Replay bei Graphics Pipeline Libraries
export RADV_PERFTEST="${RADV_PERFTEST:-nircache}"
# Cache-Statistik beim Exit (verifiziert dass Cache geflusht wurde)
export MESA_SHADER_CACHE_SHOW_STATS="${MESA_SHADER_CACHE_SHOW_STATS:-true}"

BACKUP_DIR="${BACKUP_DIR:-${CACHE_BASE}/vulkan-cache-backups}"
LOCK_FILE="/tmp/precompile-vulkan-shaders.lock"

# MoE-Optimierungen (wie start-mars-26b-server.sh)
export GGML_SCHED_PREFETCH_EXPERTS="${GGML_SCHED_PREFETCH_EXPERTS:-1}"
export GGML_SCHED_PREFETCH_SLOTS="${GGML_SCHED_PREFETCH_SLOTS:-2}"
export GGML_VK_SUBALLOCATION_BLOCK_SIZE="${GGML_VK_SUBALLOCATION_BLOCK_SIZE:-2147483648}"

# --- Lock-File (verhindert parallele Ausführung) ---
if [[ -f "$LOCK_FILE" ]]; then
  OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "error: another precompile run is active (PID $OLD_PID)" >&2
    exit 1
  fi
  rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
# Trap: Lock-File aufräumen UND Server sauber beenden bei Signal/Exit
# (setsid-Prozess läuft sonst weiter und blockiert den Port)
trap 'kill -SIGTERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; rm -f "$LOCK_FILE"' EXIT INT TERM

# --- Checks ---
if [[ ! -f "$SERVER" ]]; then
  echo "error: missing $SERVER" >&2; exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: $MAIN" >&2; exit 1
fi
if [[ ! -f "$MMPROJ" ]]; then
  echo "error: mmproj not found: $MMPROJ" >&2; exit 1
fi
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use" >&2; exit 1
fi

mkdir -p "$GGML_VK_CACHE_DIR" "$MESA_SHADER_CACHE_DIR" "$BACKUP_DIR"

# --- Cache-Größe vor dem Run ---
CACHE_SIZE_BEFORE=$(du -sb "$GGML_VK_CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
MESA_SIZE_BEFORE=$(du -sb "$MESA_SHADER_CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
echo "=== Precompile Run ==="
echo "GGML_VK_CACHE_DIR:      $GGML_VK_CACHE_DIR ($(numfmt --to=iec $CACHE_SIZE_BEFORE 2>/dev/null || echo ${CACHE_SIZE_BEFORE}B))"
echo "MESA_SHADER_CACHE_DIR:  $MESA_SHADER_CACHE_DIR ($(numfmt --to=iec $MESA_SIZE_BEFORE 2>/dev/null || echo ${MESA_SIZE_BEFORE}B))"
echo "MESA_SHADER_CACHE_MAX:  $MESA_SHADER_CACHE_MAX_SIZE"
echo "RADV_PERFTEST:          $RADV_PERFTEST"
echo "PORT:                   $PORT"
echo ""

# --- Server starten (setsid: unabhängig von SSH-Session) ---
# --warmup: Führt einen leeren Decode mit BOS+EOS durch → kompiliert
#   deterministisch die N=1 FA_SCALAR Pipeline beim Startup.
#   Kein manueller Request nötig für die erste Pipeline-Variante.
#   Die restlichen Varianten werden durch die 6 Request-Phasen abgedeckt.
echo "[$(date '+%H:%M:%S')] Starting llama-server on port $PORT (with --warmup)..."
setsid "$SERVER" \
  -m "$MAIN" \
  --mmproj "$MMPROJ" \
  --host "$HOST" --port "$PORT" \
  -c 161792 -ngl 99 \
  -ctk turbo3 -ctv turbo3 -fa on \
  -fit off \
  --parallel 2 -np 2 --cont-batching \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-ram 6144 \
  --cache-reuse 256 \
  --slot-cache-key-similarity 0.5 \
  --slot-cache-key-min-prefix 64 \
  --warmup \
  --metrics --slots \
  --log-timestamps --log-prefix \
  > /tmp/precompile-server.log 2>&1 &

SERVER_PID=$!
echo "[$(date '+%H:%M:%S')] Server PID: $SERVER_PID"
echo "[$(date '+%H:%M:%S')] Log: /tmp/precompile-server.log"

# --- Warten bis Server ready ---
echo "[$(date '+%H:%M:%S')] Waiting for server to become healthy..."
TIMEOUT=600  # 10 Minuten max
ELAPSED=0
while ! curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"ok"'; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo "[$(date '+%H:%M:%S')] TIMEOUT: server not healthy after ${TIMEOUT}s" >&2
    kill -SIGTERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] ERROR: server process died" >&2
    tail -30 /tmp/precompile-server.log >&2
    exit 1
  fi
  printf '.'
done
echo ""
echo "[$(date '+%H:%M:%S')] Server healthy after ${ELAPSED}s"

# --- Hilfsfunktion: Request senden ---
send_request() {
  local label="$1"
  local content="$2"
  local max_tokens="${3:-64}"
  local images_json="${4:-}"

  echo "[$(date '+%H:%M:%S')] >>> $label"

  local body
  if [[ -n "$images_json" ]]; then
    body=$(cat <<JSON
{"messages":[{"role":"user","content":$images_json}],"max_tokens":$max_tokens,"temperature":1.0,"top_p":0.95,"top_k":64,"stream":false}
JSON
)
  else
    # content als JSON-String escapen
    local escaped
    escaped=$(printf '%s' "$content" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
    body=$(cat <<JSON
{"messages":[{"role":"user","content":$escaped}],"max_tokens":$max_tokens,"temperature":1.0,"top_p":0.95,"top_k":64,"stream":false}
JSON
)
  fi

  local start_time
  start_time=$(date +%s)

  local response
  local curl_rc=0
  response=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$body" 2>&1) || curl_rc=$?

  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))

  if [[ $curl_rc -ne 0 ]]; then
    echo "[$(date '+%H:%M:%S')]     WARNING: curl failed (rc=$curl_rc) for $label — pipelines may not be compiled" >&2
    echo "[$(date '+%H:%M:%S')]     response: ${response:0200}" >&2
    return 1
  fi

  # Tokens extrahieren (falls verfügbar)
  local prompt_tokens
  local completion_tokens
  prompt_tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null || echo "?")
  completion_tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens','?'))" 2>/dev/null || echo "?")

  echo "[$(date '+%H:%M:%S')]     ${duration}s, prompt=${prompt_tokens}, completion=${completion_tokens}"
}

# --- Phase 1: Generation (N=1 → FA_SCALAR, Br=1) ---
# Kurzer Prompt + 64 Tokens Generation → kompiliert scalar FA für HSK=512 und HSK=256
# 64 Tokens decken aligned (KV%64==0) und unaligned Varianten ab
echo ""
echo "=== Phase 1: Generation (FA_SCALAR, N=1) ==="
send_request "Short prompt + 64 tokens generation" "Hello, how are you today?" 64

# --- Phase 2: GQA-Generation (N=2 → FA_COOPMAT1, Br=16) ---
# Ein weiterer Request mit kurzem Prompt, der GQA-Generation auslöst
# (bei N<=8 mit GQA wird N=gqa_ratio=2 gesetzt)
echo ""
echo "=== Phase 2: GQA-Generation (FA_COOPMAT1, N=2) ==="
send_request "GQA generation trigger" "What is 2+2?" 32

# --- Phase 3: Prompt-Processing (N>8 → FA_COOPMAT1 with mask) ---
# Verschiedene Prompt-Längen triggern aligned/unaligned + mask/mask_opt Varianten
echo ""
echo "=== Phase 3: Prompt-Processing (FA_COOPMAT1, N>8, with mask) ==="

# Prompt-Längen die verschiedene KV%64 Werte abdecken
# 32 Tokens → KV%64=32 → unaligned
# 64 Tokens → KV%64=0 → aligned
# 128 Tokens → KV%64=0 → aligned, mask_opt trigger (nem1>=32, nem0*nem1>32768)
# 256 Tokens → KV%64=0 → aligned, mask_opt
# 100 Tokens → KV%64=36 → unaligned
# 200 Tokens → KV%64=8 → unaligned

for target_tokens in 32 64 100 128 200 256 512; do
  # Generiere Prompt mit ungefähr target_tokens Tokens
  # Grobe Schätzung: ~1 Token pro Wort, ~0.75 Token pro Zeichen
  # Wir verwenden Wiederholungen eines einfachen Worts
  content=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * ($target_tokens // 9 + 1))")
  send_request "Prompt ~${target_tokens} tokens" "$content" 8
done

# --- Phase 4: Large Prompt (mask_opt pipeline) ---
# mask_opt wird aktiviert wenn nem1 >= 32 && nem0*nem1 > 32768 && nem0 >= Bc*16
# Bei Bc=64: nem0 >= 1024. Bei nem1=512 (ubatch): 1024*512 > 32768 → mask_opt
echo ""
echo "=== Phase 4: Large Prompt (mask_opt pipeline) ==="
large_content=$(python3 -c "print('Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ' * 200)")
send_request "Large prompt ~2000+ tokens" "$large_content" 8

# --- Phase 5: Vision Request (mmproj-spezifische Pipelines) ---
echo ""
echo "=== Phase 5: Vision Request (mmproj Pipelines) ==="

# Kleines Test-Bild erstellen (1x1 rotes PNG)
python3 -c "
import base64
png = bytes([
    0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,  # PNG signature
    0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,  # IHDR chunk
    0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,  # 1x1
    0x08,0x02,0x00,0x00,0x00,0x90,0x77,0x53,0xDE,  # RGB
    0x00,0x00,0x00,0x0C,0x49,0x44,0x41,0x54,  # IDAT chunk
    0x08,0xD7,0x63,0xF8,0xCF,0xC0,0x00,0x00,0x00,0x03,0x00,0x01,
    0x5B,0x9E,0xE2,0xA0,
    0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,0x82,  # IEND
])
print(base64.b64encode(png).decode())
" > /tmp/test_image_b64.txt

IMAGE_B64=$(cat /tmp/test_image_b64.txt)
# Bild-Content als JSON-Array
IMAGES_JSON=$(python3 -c "
import json
b64 = open('/tmp/test_image_b64.txt').read().strip()
content = [
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}},
    {'type': 'text', 'text': 'What color is this image?'}
]
print(json.dumps(content))
")

send_request "Vision request (1x1 PNG)" "" 32 "$IMAGES_JSON"

# --- Phase 6: Weitere Generation für vollständige Cache-Abdeckung ---
# Nach den Prompt-Processing-Requests gibt es mehr KV-Cache → aligned/unaligned
# bei verschiedenen KV-Längen
echo ""
echo "=== Phase 6: Additional Generation (aligned/unaligned coverage) ==="
send_request "Final generation pass" "Please count from 1 to 20." 64

# --- Sauberer Shutdown (SIGTERM + wait) ---
echo ""
echo "[$(date '+%H:%M:%S')] Sending SIGTERM for clean shutdown (cache flush)..."
kill -SIGTERM "$SERVER_PID" 2>/dev/null || true

# Warten bis der Prozess beendet ist (Cache wird geflusht)
# SIGKILL nach Timeout zerstört den VkPipelineCache — Backup wird übersprungen
echo "[$(date '+%H:%M:%S')] Waiting for clean shutdown..."
WAIT_TIMEOUT=120
WAIT_ELAPSED=0
SIGKILLED=0
while kill -0 "$SERVER_PID" 2>/dev/null; do
  sleep 2
  WAIT_ELAPSED=$((WAIT_ELAPSED + 2))
  if [[ $WAIT_ELAPSED -ge $WAIT_TIMEOUT ]]; then
    echo "[$(date '+%H:%M:%S')] WARNING: server did not exit after ${WAIT_TIMEOUT}s, sending SIGKILL" >&2
    echo "[$(date '+%H:%M:%S')] WARNING: VkPipelineCache geht verloren — Backup wird übersprungen!" >&2
    kill -SIGKILL "$SERVER_PID" 2>/dev/null || true
    SIGKILLED=1
    break
  fi
  printf '.'
done
echo ""
wait "$SERVER_PID" 2>/dev/null || true
if [[ $SIGKILLED -eq 1 ]]; then
  echo "[$(date '+%H:%M:%S')] Server killed (NOT clean — cache may be incomplete)"
else
  echo "[$(date '+%H:%M:%S')] Server stopped cleanly"
fi

# --- Cache-Größe nach dem Run ---
CACHE_SIZE_AFTER=$(du -sb "$GGML_VK_CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
MESA_SIZE_AFTER=$(du -sb "$MESA_SHADER_CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
CACHE_DELTA=$((CACHE_SIZE_AFTER - CACHE_SIZE_BEFORE))
MESA_DELTA=$((MESA_SIZE_AFTER - MESA_SIZE_BEFORE))

echo ""
echo "=== Cache Statistics ==="
echo "GGML VK Pipeline Cache:  $(numfmt --to=iec $CACHE_SIZE_AFTER 2>/dev/null || echo ${CACHE_SIZE_AFTER}B) (delta: +$(numfmt --to=iec $CACHE_DELTA 2>/dev/null || echo ${CACHE_DELTA}B))"
echo "Mesa Shader Cache:       $(numfmt --to=iec $MESA_SIZE_AFTER 2>/dev/null || echo ${MESA_SIZE_AFTER}B) (delta: +$(numfmt --to=iec $MESA_DELTA 2>/dev/null || echo ${MESA_DELTA}B))"
echo ""
echo "Pipeline cache files:"
ls -lh "$GGML_VK_CACHE_DIR/" 2>&1

# --- Backup ---
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_TAR="${BACKUP_DIR}/vulkan-cache-${TIMESTAMP}.tar.gz"

if [[ $SIGKILLED -eq 1 ]]; then
  echo ""
  echo "[$(date '+%H:%M:%S')] SKIPPING backup — server was SIGKILLed, VkPipelineCache is incomplete/missing"
  echo "[$(date '+%H:%M:%S')] Re-run precompile after fixing the shutdown issue"
else
  echo ""
  echo "[$(date '+%H:%M:%S')] Creating backup: $BACKUP_TAR"
  tar -czf "$BACKUP_TAR" \
    -C "$(dirname "$GGML_VK_CACHE_DIR")" "$(basename "$GGML_VK_CACHE_DIR")" \
    -C "$(dirname "$MESA_SHADER_CACHE_DIR")" "$(basename "$MESA_SHADER_CACHE_DIR")" \
    2>&1

  BACKUP_SIZE=$(du -sh "$BACKUP_TAR" 2>/dev/null | cut -f1)
  echo "[$(date '+%H:%M:%S')] Backup created: $BACKUP_TAR ($BACKUP_SIZE)"
fi

# Alte Backups aufräumen (behalte nur das neueste — Cache verändert sich nicht nach vollständiger Kompilierung)
# Nur ausführen wenn ein neues Backup erstellt wurde
if [[ $SIGKILLED -eq 0 ]]; then
  find "$BACKUP_DIR" -maxdepth 1 -name "vulkan-cache-*.tar.gz" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | tail -n +2 | cut -d' ' -f2- | while IFS= read -r old; do
    echo "[$(date '+%H:%M:%S')] Removing old backup: $old"
    rm -f "$old"
  done
fi

echo ""
echo "=== Precompile Complete ==="
if [[ $SIGKILLED -eq 1 ]]; then
  echo "WARNING: No backup created (server was SIGKILLed)"
else
  echo "Backup:  $BACKUP_TAR"
  echo "Restore: tar -xzf $BACKUP_TAR -C ${CACHE_BASE}/"
fi
echo ""
echo "Mesa cache stats (from server log):"
grep -i "cache\|hit\|miss" /tmp/precompile-server.log 2>/dev/null | tail -5 || echo "(no stats in log)"
