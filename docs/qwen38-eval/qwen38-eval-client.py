import json
import sys
import time
import urllib.request

port, prompt_path, output_path, label = sys.argv[1:5]
prompt = open(prompt_path, encoding="utf-8").read()
body = {
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7,
    "top_p": 0.95,
    "min_p": 0.0,
    "seed": 4242,
    "max_tokens": 98304,
    "chat_template_kwargs": {"reasoning_effort": "medium"},
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
start = time.monotonic()
with urllib.request.urlopen(req, timeout=10800) as response:
    data = json.loads(response.read())
with open(output_path, "w", encoding="utf-8") as out:
    json.dump(data, out, ensure_ascii=False)
message = data["choices"][0]["message"]
usage = data.get("usage", {})
print(json.dumps({
    "label": label,
    "finish_reason": data["choices"][0].get("finish_reason"),
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "reasoning_chars": len(message.get("reasoning_content") or ""),
    "content_chars": len(message.get("content") or ""),
    "elapsed_seconds": round(time.monotonic() - start, 2),
}, ensure_ascii=False))
