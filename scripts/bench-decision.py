#!/usr/bin/env python3
"""Benchmark: /v1/decision vs. /v1/chat/completions fuer dieselbe Klassifikationsaufgabe.

Misst auf einem laufenden llama-server (mit --decision-seqs):
  A) N Entscheidungs-Requests (Prefix-Cache wird ab dem 2. Request warm)
  B) N Chat-Completions die dasselbe Schema als JSON liefern sollen

Ausgabe: Tabelle mit Latenzen, Token-Zahlen und Schema-Konformitaet.

Beispiel:
  python3 scripts/bench-decision.py --url http://localhost:18080 --requests 5
"""

import argparse
import json
import statistics
import time
import urllib.request

SCHEMA = {
    "category": {
        "type": "enum",
        "choices": ["billing", "technical", "other"],
        "description": "Kategorie der Anfrage",
    },
    "urgent": {
        "type": "boolean",
        "description": "Erfordert sofortige Bearbeitung",
    },
}

CONTEXTS = [
    "Ich wurde doppelt belastet und brauche eine Rueckerstattung.",
    "Die Anwendung stuerzt beim Start mit einem Fehler ab.",
    "Koennt ihr mir sagen, wann die neue Version erscheint?",
    "Mein Account wurde gesperrt, ich kann mich nicht einloggen!",
    "Wie aendere ich meine hinterlegte Zahlungsmethode?",
]

INSTRUCTIONS = "Klassifiziere jede Support-Anfrage nach Kategorie und Dringlichkeit."


def post(url: str, path: str, body: dict, timeout: float = 600.0):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, (time.perf_counter() - t0) * 1000.0


def bench_decision(url: str, model: str | None, n: int, contexts: list[str]) -> dict:
    rows = []
    for i in range(n):
        ctxs = [contexts[(i + k) % len(contexts)] for k in range(2)]
        body = {
            "instructions": INSTRUCTIONS,
            "schema": SCHEMA,
            "contexts": ctxs,
            "mode": "tree",
        }
        if model:
            body["model"] = model
        try:
            res, wall = post(url, "/v1/decision", body)
            t = res.get("timings", {})
            rows.append({
                "wall_ms": wall,
                "total_ms": t.get("total_ms"),
                "scoring_ms": t.get("scoring_ms"),
                "prefill_ms": t.get("prefill_ms"),
                "cached_tokens": res.get("usage", {}).get("cached_tokens", 0),
                "n_results": len(res.get("results", [])),
                "ok": True,
            })
        except Exception as e:
            rows.append({"ok": False, "error": str(e), "wall_ms": None})
    return {"rows": rows}


def bench_chat(url: str, model: str | None, n: int, contexts: list[str]) -> dict:
    prompt_tpl = (
        INSTRUCTIONS
        + " Antworte ausschliesslich als JSON-Objekt {{\"category\": ..., \"urgent\": ...}}.\n\n"
        + "Anfrage: {ctx}"
    )
    rows = []
    for i in range(n):
        ctx = contexts[i % len(contexts)]
        body = {
            "messages": [{"role": "user", "content": prompt_tpl.format(ctx=ctx)}],
            "max_tokens": 700,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "stream": False,
        }
        if model:
            body["model"] = model
        try:
            res, wall = post(url, "/v1/chat/completions", body)
            ch = res["choices"][0]
            content = ch["message"].get("content") or ""
            valid = False
            try:
                obj = json.loads(content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                valid = obj.get("category") in SCHEMA["category"]["choices"] and isinstance(obj.get("urgent"), bool)
            except Exception:
                pass
            rows.append({
                "wall_ms": wall,
                "completion_tokens": res.get("usage", {}).get("completion_tokens", 0),
                "reasoning_chars": len(ch["message"].get("reasoning_content") or ""),
                "finish": ch.get("finish_reason"),
                "schema_valid": valid,
                "ok": True,
            })
        except Exception as e:
            rows.append({"ok": False, "error": str(e), "wall_ms": None})
    return {"rows": rows}


def summarize(name: str, rows: list[dict]):
    ok = [r for r in rows if r["ok"]]
    print(f"\n== {name} ({len(ok)}/{len(rows)} ok) ==")
    if not ok:
        for r in rows:
            print("  FEHLER:", r.get("error"))
        return
    walls = [r["wall_ms"] for r in ok]
    print(f"  wall_ms   : med={statistics.median(walls):8.0f}  min={min(walls):8.0f}  max={max(walls):8.0f}")
    if "scoring_ms" in ok[0]:
        sc = [r["scoring_ms"] or 0 for r in ok]
        pf = [r["prefill_ms"] or 0 for r in ok]
        ct = [r["cached_tokens"] for r in ok]
        print(f"  scoring_ms: med={statistics.median(sc):8.0f}")
        print(f"  prefill_ms: med={statistics.median(pf):8.0f}")
        print(f"  cached_tok: {ct}")
    if "completion_tokens" in ok[0]:
        toks = [r["completion_tokens"] for r in ok]
        val = sum(1 for r in ok if r["schema_valid"])
        print(f"  compl_toks: med={statistics.median(toks):8.0f}")
        print(f"  schema ok : {val}/{len(ok)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:18080")
    p.add_argument("--model", default=None)
    p.add_argument("--requests", type=int, default=5)
    p.add_argument("--skip-chat", action="store_true")
    p.add_argument("--skip-decision", action="store_true")
    args = p.parse_args()

    print(f"Ziel: {args.url}  requests={args.requests}")
    if not args.skip_decision:
        summarize("/v1/decision", bench_decision(args.url, args.model, args.requests, CONTEXTS)["rows"])
    if not args.skip_chat:
        summarize("/v1/chat/completions", bench_chat(args.url, args.model, args.requests, CONTEXTS)["rows"])


if __name__ == "__main__":
    main()
