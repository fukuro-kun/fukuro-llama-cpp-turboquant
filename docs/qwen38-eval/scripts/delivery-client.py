#!/usr/bin/env python3
# delivery-client.py — Ein Request mit vollstaendig explizitem Sampler-Block.
# Usage: delivery-client.py <port> <prompt_file> <out.json> <label> <seed> [effort] [timeout_s]
# Druckt eine kompakte Ergebniszeile auf stdout (fuer Ledger-Aufbau).
import json
import sys
import time
import urllib.request

port, prompt_path, out, label, seed = sys.argv[1:6]
effort = sys.argv[6] if len(sys.argv) > 6 else "medium"
timeout = int(sys.argv[7]) if len(sys.argv) > 7 else 7200
prompt = open(prompt_path, encoding="utf-8").read()
body = {
    "model": "q38",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7,
    "top_p": 0.95,
    "min_p": 0.0,
    "seed": int(seed),
    "max_tokens": 98304,
}
if effort != "none":
    body["chat_template_kwargs"] = {"reasoning_effort": effort}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
t0 = time.monotonic()
try:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
except Exception as e:
    print(json.dumps({
        "label": label, "error": str(e)[:200], "content_chars": 0,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }, ensure_ascii=False))
    sys.exit(3)
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
m = data["choices"][0]["message"]
u = data.get("usage", {})
print(json.dumps({
    "label": label,
    "finish_reason": data["choices"][0].get("finish_reason"),
    "prompt_tokens": u.get("prompt_tokens"),
    "completion_tokens": u.get("completion_tokens"),
    "reasoning_chars": len(m.get("reasoning_content") or ""),
    "content_chars": len(m.get("content") or ""),
    "elapsed_seconds": round(time.monotonic() - t0, 1),
}, ensure_ascii=False))
