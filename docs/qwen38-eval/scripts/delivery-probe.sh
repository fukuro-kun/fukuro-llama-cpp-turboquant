#!/usr/bin/env bash
# delivery-probe.sh — Delivery-Rate-Messung: Seed-Serie auf einem GPU-Slot.
# Usage: delivery-probe.sh <label> <model_path> <port> <gpu_id|-> <ledger.jsonl>
# Pro Seed: task1 + task2 (interleaved). Failure = content_chars==0 oder Client-Error.
# Early-Stop: >=5 Fails auf einem Task -> Rest dieses Tasks ueberspringen.
# Server: identische Flags wie die Hauptkampagne, KEIN mmproj/Vision.
set -u
LABEL="$1"; MODEL="$2"; PORT="$3"; GPU="${4:-}"; LEDGER="$5"
SEEDS="1 7 42 99 777 1337 2024 3141 4242 5555 8888 12345"
BIN=${BIN:-$HOME/git/fukuro-llama-cpp-turboquant/build/bin/llama-server}
CLIENT=${CLIENT:-/tmp/delivery-client.py}
OUTDIR=${OUTDIR:-/tmp/q38-delivery}
SRVLOG=/tmp/q38-dlv-$LABEL-srv.log
RUNLOG=/tmp/q38-dlv-$LABEL-run.log
mkdir -p "$OUTDIR/runs" "$OUTDIR"

echo "=== PROBE $LABEL $(date -Is) ===" >> "$RUNLOG"
if [ -n "$GPU" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi

RESTARTS=0
start_server() {
  "$BIN" -m "$MODEL" \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -c 98304 -ngl 99 -ub 512 -fa on -ctk turbo4 -ctv turbo3 \
    --parallel 1 --cache-ram 0 --no-warmup -fit off --jinja \
    --host 127.0.0.1 --port "$PORT" >> "$SRVLOG" 2>&1 &
  SRV=$!
  echo "$(date -Is) srv pid=$SRV gpu=${GPU:-all} port=$PORT" >> "$RUNLOG"
  local ok=0
  for i in $(seq 1 360); do
    kill -0 $SRV 2>/dev/null || break
    curl -sf -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  if [ "$ok" != 1 ]; then
    echo "LOAD-FAIL $LABEL" >> "$RUNLOG"; tail -15 "$SRVLOG" >> "$RUNLOG"; return 1
  fi
  echo "$(date -Is) healthy" >> "$RUNLOG"; return 0
}
start_server || exit 1

T1F=0; T2F=0
for s in $SEEDS; do
  for t in 1 2; do
    if [ "$t" = 1 ]; then curfails=$T1F; else curfails=$T2F; fi
    [ "$curfails" -ge 5 ] && continue
    out="$OUTDIR/runs/dlv-$LABEL-task$t-seed$s.json"
    [ -s "$out" ] && { echo "skip $out" >> "$RUNLOG"; continue; }
    # Server-Ueberlebens-Check: Crash -> Restart (max 2x), dann Abbruch
    if ! kill -0 $SRV 2>/dev/null || \
       ! curl -sf -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "$(date -Is) SERVER-DOWN, restart" >> "$RUNLOG"
      RESTARTS=$((RESTARTS+1))
      if [ "$RESTARTS" -gt 2 ]; then
        echo "ABORT $LABEL: >2 Restarts" >> "$RUNLOG"; exit 2
      fi
      sleep 10
      start_server || exit 2
      # nach Restart diesen Run erneut probieren
    fi
    echo "$(date -Is) run task$t seed$s" >> "$RUNLOG"
    line=$(python3 "$CLIENT" "$PORT" "/tmp/qwen38-task$t.txt" "$out" \
           "dlv-$LABEL-task$t-seed$s" "$s" medium 7200 2>>"$RUNLOG")
    rc=$?
    echo "  -> $line" >> "$RUNLOG"
    # Mid-Run-Servercrash: kein Modell-Fail -> Restart + Run wiederholen
    if [ "$rc" != 0 ] && ! kill -0 $SRV 2>/dev/null; then
      echo "$(date -Is) MIDRUN-CRASH bei task$t seed$s -> Restart+Retry" >> "$RUNLOG"
      RESTARTS=$((RESTARTS+1))
      if [ "$RESTARTS" -gt 2 ]; then
        echo "ABORT $LABEL: >2 Restarts" >> "$RUNLOG"; exit 2
      fi
      sleep 10
      start_server || exit 2
      line=$(python3 "$CLIENT" "$PORT" "/tmp/qwen38-task$t.txt" "$out" \
             "dlv-$LABEL-task$t-seed$s" "$s" medium 7200 2>>"$RUNLOG")
      rc=$?
      echo "  retry-> $line" >> "$RUNLOG"
    fi
    cc=$(printf '%s' "$line" | python3 -c \
         "import json,sys;print(json.load(sys.stdin).get('content_chars') or 0)" \
         2>/dev/null || echo 0)
    if [ "$rc" != 0 ] || [ "${cc:-0}" -le 0 ]; then
      if [ "$t" = 1 ]; then T1F=$((T1F+1)); else T2F=$((T2F+1)); fi
    fi
    # Ledger-Zeile anreichern + anhaengen
    printf '%s' "$line" | python3 -c "
import json,sys
try: r=json.load(sys.stdin)
except Exception: sys.exit(0)
r.update({'run_id':'dlv-$LABEL-task$t-seed$s','host':'$LABEL',
  'model_file':'$(basename "$MODEL")','task':'task$t-seed$s',
  'reasoning_effort':'medium','max_tokens':98304,'seed':$s,
  'run_file':'runs/dlv-$LABEL-task$t-seed$s.json'})
print(json.dumps(r,ensure_ascii=False))" >> "$LEDGER" 2>/dev/null || true
  done
done

kill -INT $SRV 2>/dev/null; sleep 5
kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "=== DONE $LABEL $(date -Is) task1_fails=$T1F task2_fails=$T2F ===" >> "$RUNLOG"
