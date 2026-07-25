# SPEC: Slot-Progress-Exposure im llama-server

**Projekt:** fukuro-llama-cpp-turboquant
**Datum:** 2026-07-25
**Status:** Draft — wartet auf Implementierung
**Abhängigkeiten:** janus InferenzQuelle Progress-Polling (Option D)

---

## Ziel

Der llama-server soll den Prefill-Fortschritt (`progress`) und die Gesamtzahl der
Task-Tokens (`n_tokens_total`) im `/slots` JSON-Endpoint exponieren. Dies ermöglicht
dem janus-Router, während des Prefills den Fortschritt zu pollen und zwischen
"Backend arbeitet (SWA cache invalidation)" und "Backend hängt" zu unterscheiden.

## Problem

Aktuell liefert `/slots` folgende Felder pro Slot:

```json
{
  "id": 0,
  "n_ctx": 131072,
  "is_processing": false,
  "id_task": 80908,
  "n_prompt_tokens": 24,
  "n_prompt_tokens_processed": 11,
  "n_prompt_tokens_cache": 0
}
```

**Fehlend:** `progress` (0.0–1.0) und `n_tokens_total` (Gesamtzahl der Prompt-Tokens
des Tasks). Der `progress`-Wert wird intern in `print_timings_pp()` berechnet:

```cpp
const double f_progress = (float) prompt.n_tokens() / task->n_tokens();
```

aber nur ins Log geschrieben (`SLT_INF`), nicht im `/slots` JSON ausgegeben.

## Änderung

### Datei: `tools/server/server-context.cpp`

In der `to_json()` Methode von `server_slot` (Zeile ~551), im `if (ptask)` Block:

**Hinzufügen:**
```cpp
res["n_tokens_total"] = (int32_t) ptask->n_tokens();
res["progress"] = ptask && ptask->n_tokens() > 0
    ? (double) prompt.n_tokens() / ptask->n_tokens()
    : 0.0;
```

**Position:** Nach `res["n_prompt_tokens_cache"]` (Zeile ~571), vor `res["params"]`.

### Ergebnis

`/slots` liefert danach:

```json
{
  "id": 0,
  "n_ctx": 131072,
  "is_processing": true,
  "id_task": 80908,
  "n_prompt_tokens": 24,
  "n_prompt_tokens_processed": 11,
  "n_prompt_tokens_cache": 0,
  "n_tokens_total": 2628,
  "progress": 0.42
}
```

## Kompatibilität

- **Backward-compatible:** Neue Felder werden hinzugefügt, keine bestehenden entfernt.
- **Clients die diese Felder nicht kennen** ignorieren sie (JSON ist tolerant).
- **janus** nutzt `progress` und `n_tokens_total` für Prefill-Progress-Polling.
- **Andere Clients** (bot, dashboard) sind nicht betroffen.

## Edge Cases

1. **`ptask` ist `nullptr`:** `to_json()` prüft bereits `if (ptask)` — die neuen
   Felder werden nur hinzugefügt wenn ein Task existiert.
2. **`ptask->n_tokens() == 0`:** Division durch Null — guarded durch
   `ptask->n_tokens() > 0` Check, liefert `0.0`.
3. **Slot ist idle (`is_processing: false`):** `ptask` ist `task_prev` (letzter Task).
   `progress` zeigt den Endstand des letzten Tasks (1.0 wenn fertig). Das ist okay —
   janus pollt nur während `is_processing: true`.
4. **Prompt wird nachgeladen (Streaming-Input):** `task->n_tokens()` kann sich
   ändern wenn neue Tokens zum Task hinzugefügt werden. `progress` kann >1.0 werden
   wenn `prompt.n_tokens() > task->n_tokens()`. Clamp auf 1.0:
   `std::min(1.0, (double) prompt.n_tokens() / ptask->n_tokens())`.

## Verifikation

1. **Build:** `cmake --build build --target llama-server` (oderäquivalent)
2. **Start:** `./llama-server -m <model> --port 18080`
3. **Test:** `curl -s http://localhost:18080/slots | python3 -m json.tool`
   → Felder `n_tokens_total` und `progress` müssen vorhanden sein
4. **Während Prefill:** Großen Prompt senden, `/slots` während des Prefills pollen:
   ```bash
   # In einem Terminal:
   curl -s http://localhost:18080/v1/chat/completions -d '{"messages":[{"role":"user","content":"$(python3 -c "print('x '*5000)")"}],"stream":true}' &
   # In einem anderen Terminal:
   watch -n 1 'curl -s http://localhost:18080/slots | python3 -c "import json,sys; d=json.load(sys.stdin); print([{\"progress\": s.get(\"progress\",0), \"processed\": s.get(\"n_prompt_tokens_processed\",0), \"total\": s.get(\"n_tokens_total\",0)} for s in d])"'
   ```
   → `progress` muss von 0.0 → 1.0 steigen, `n_prompt_tokens_processed` muss
   sich bewegen.
5. **Idle-Slot:** Wenn kein Task läuft: `progress` = 0.0 oder 1.0 (letzter Task),
   `n_tokens_total` = Token-Anzahl des letzten Tasks.
6. **Existierende Tests:** `cd tools/server/tests && pytest tests/unit/ -q` —
   alle müssen grün bleiben.

## Build-System

Keine Änderungen am Build-System nötig — nur `server-context.cpp` wird geändert.

## DOX-Update

Nach der Implementierung `tools/server/AGENTS.md` aktualisieren: `/slots` Endpoint
dokumentieren mit den neuen Feldern.
