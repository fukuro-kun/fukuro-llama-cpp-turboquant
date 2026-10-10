#!/usr/bin/env python3
"""Bench driver: POST prompt files to a llama-server-compatible /completion endpoint.
Writes one JSONL line per run: engine, label, file, run, prompt_n, prompt_ms,
prompt_per_second, predicted_n, predicted_ms, predicted_per_second, ttft_ms (if stream).
Usage: bench_measure.py --url http://127.0.0.1:PORT --engine TAG --label LABEL \
        --files f1.txt f2.txt --runs 3 --n-predict 128 --out results.jsonl [--openai]
"""
import argparse, json, time, sys, urllib.request

def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0

def post_openai_stream(url, payload, timeout=1800):
    """Return (ttft_ms, total_s, usage, chunks)."""
    payload = dict(payload); payload["stream"] = True; payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; nch = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"): continue
            d = line[5:].strip()
            if d == b"[DONE]": break
            try: j = json.loads(d)
            except Exception: continue
            nch += 1
            if ttft is None and j.get("choices"):
                d0 = j["choices"][0].get("delta", {})
                if d0.get("content") or d0.get("reasoning_content"):
                    ttft = (time.time() - t0) * 1000
            if j.get("usage"): usage = j["usage"]
    return ttft, time.time() - t0, usage, nch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument("--openai", action="store_true", help="use /v1/chat/completions with stream for TTFT")
    p.add_argument("--max-prompt", type=int, default=0, help="truncate prompt to N chars (0=no)")
    p.add_argument("--vary", action="store_true", help="prepend unique marker per run (defeat prompt caches)")
    a = p.parse_args()

    for f in a.files:
        prompt = open(f).read()
        if a.max_prompt: prompt = prompt[:a.max_prompt]
        for run in range(1, a.runs + 1):
            rec = {"engine": a.engine, "label": a.label, "file": f, "run": run,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                ptxt = prompt
                if a.vary:
                    ptxt = f"### bench-marker run={run} file={f} seed={time.time_ns()%10**9} ###\n" + prompt
                if a.openai:
                    pl = {"model": "bench", "messages": [{"role": "user", "content": ptxt}],
                          "max_tokens": a.n_predict, "temperature": 0.0}
                    ttft, tot, usage, nch = post_openai_stream(a.url + "/v1/chat/completions", pl)
                    rec.update(ttft_ms=ttft, total_s=tot, chunks=nch,
                               prompt_n=usage.get("prompt_tokens") if usage else None,
                               predicted_n=usage.get("completion_tokens") if usage else None)
                    if usage and usage.get("completion_tokens") and tot and ttft is not None:
                        dec_s = tot - ttft/1000
                        rec["decode_per_second"] = (usage["completion_tokens"]-1)/dec_s if dec_s > 0 else None
                else:
                    pl = {"prompt": ptxt, "n_predict": a.n_predict, "temperature": 0.0,
                          "cache_prompt": False, "stream": False}
                    j, tot = post(a.url + "/completion", pl)
                    t = j.get("timings", {})
                    rec.update(total_s=tot, prompt_n=t.get("prompt_n"),
                               prompt_ms=t.get("prompt_ms"),
                               prompt_per_second=t.get("prompt_per_second"),
                               predicted_n=t.get("predicted_n"),
                               predicted_ms=t.get("predicted_ms"),
                               predicted_per_second=t.get("predicted_per_second"))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

if __name__ == "__main__":
    main()
