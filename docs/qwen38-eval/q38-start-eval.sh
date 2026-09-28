#!/usr/bin/env bash
set -euo pipefail
model=$1
port=$2
label=$3
server="$HOME/git/fukuro-llama-cpp-turboquant/build/bin/llama-server"
log="/tmp/qwen38-${label}.log"
if [[ ! -x "$server" || ! -f "$model" ]]; then
  printf 'missing server or model\n' >&2
  exit 1
fi
if lsof -ti:"$port" >/dev/null 2>&1; then
  printf 'port %s already in use\n' "$port" >&2
  exit 2
fi
setsid "$server" -m "$model" --spec-type draft-mtp --spec-draft-n-max 2 \
  -c 98304 -ngl 99 -ub 512 -fa on -ctk turbo4 -ctv turbo3 \
  --parallel 1 --cache-ram 0 --no-warmup -fit off --jinja \
  --host 127.0.0.1 --port "$port" >"$log" 2>&1 </dev/null &
printf '%s\n' "$!"
