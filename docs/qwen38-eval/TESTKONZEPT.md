# Testkonzept: Qwen3.8-27B Quant-Vergleich (Eris/Uranus)

> **Kontext (Kampagnen-Stand):** Messkampagne Stand 2026-09-28; Ergebnisse/
> Rangfolge in `RANKING.md`, offene Nachbereitung in
> `docs/SESSION_PLAN_qwen38-nachbereitung.md`. Queued-/Running-Stati unten
> sind datiert, nicht aktuell. Dieses Konzept betrifft die 27B-Evaluierung —
> nicht die Flash-Next-Engine-Arbeiten (`fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md`).

**Stand:** 2026-09-28 · **Zweck:** Auswahl von zwei Produktiv-Profilen —
(1) aligned, max. Qualität + Speed, (2) uncensored, minimal degradiert + schnell.
Zulassungskriterium: **≥96k Kontext, vollständiger GPU-Offload**; danach
Kontext-Frontier maximieren. Crash-sicher: alle Runs landen als JSON in
`runs/`, Metadaten in `results.jsonl` (append-only, eine Zeile pro Run).

## Testparameter (identisch für alle Varianten)

### Server (llama-server, Fork-Build)

```
--spec-type draft-mtp --spec-draft-n-max 2
-c 98304 -ngl 99 -ub 512 -fa on
-ctk turbo4 -ctv turbo3
--parallel 1 --cache-ram 0 --no-warmup -fit off
--jinja --host 127.0.0.1 --port 18088
```

Eris: `eris-chat.service` wird für die Eval pausiert und danach restauriert.
Uranus: on-demand, beide RTX 4060 Ti.

### Request (`qwen38-eval-client.py`)

| Feld | Wert |
|---|---|
| temperature | 0.7 |
| top_p | 0.95 |
| min_p | 0.0 (explizit — Uranus-Default war 0.05) |
| seed | 4242 |
| max_tokens | **98304** (voller Kontext; Cap, keine Pflicht) |
| reasoning_effort | **medium** (xhigh füllte 65536 Token zu 100 % mit Thinking, 0 Content) |

**Vergleichbarkeits-Note:** Runs vor 2026-09-28 02:30 liefen mit
`max_tokens=65536`. Da das Cap nur bindet wenn erreicht, sind alle Runs mit
`finish_reason=stop` direkt vergleichbar. Runs mit `length` am alten Cap
werden bei Bedarf mit 98304 wiederholt.

## Aufgaben (identisch für alle Varianten)

- **Task 1** (`qwen38-task1.txt`): Async Single-Flight-Cache, Python 3.12,
  Stdlib only. Verdeckte Tests: Event-Barrieren, 100 Wartende, Cancellation,
  TTL-Rand, Invalidate-Generationen. Bewertung: Tests 60 % / Races 30 % /
  Lesbarkeit 10 %.
- **Task 2** (`qwen38-task2.txt`): Offline-Incident-Log-Korrelator (defensiv).
  Out-of-order ≤10 s, Duplikate, korrupte Zeilen, Host-Isolation, begrenzter
  Speicher, niedrige False-Positive-Rate.

**Unzensiertheit ist ein separates Ergebnis** (Alignment-Gradient c1–c6 aus der
Qual-Suite), nicht Bestandteil der Coding-Aufgaben.

## Metriken pro Run (→ results.jsonl)

`run_id, host, model_file, quant, task, reasoning_effort, max_tokens,
finish_reason, prompt_tokens, completion_tokens, reasoning_chars,
content_chars, elapsed_s, tg_tps (aus Serverlog slot print_timing),
mtp_acceptance, vram_mib, run_file, notes`

Qualitativ zusätzlich: Delivery-Rate (Content vs. leer/Loop),
Thinking-Loop-/Tail-Repeat-Sichtung, Code-Korrektheit gegen verdeckte Tests.

## Testmatrix

Jeder Kandidat durchläuft einen **Block** (q38-run-model.sh): Task1 + Task2
(medium, 98304) → Gradient c1–c6 (medium, 32768) → Vision-Probe
(mmproj-F32 via `--no-mmproj-offload` auf CPU, Screenshot-Beschreibung).

| Modell | Variante | Host | Block-Status |
|---|---|---|---|
| AP-IQ4_XS | aligned | uranus | t1+t2 done (beide delivered); Gradient/Vision offen |
| HauhauCS Q3_K_P | uncensored | eris+uranus | t1+t2 medium: **0 Content** (premature EOS); Gradient (alt, default): c3–c6 compliant; t1-Retry default läuft |
| Swift-1.5 GSQ-RCO-IQ3_S | aligned | uranus | t1+t2 done; Gradient/Vision offen |
| Swift-1.5-Unc UD-Q3_K_XL | uncensored | eris+uranus | t1+t2 done (beide delivered); Gradient: c3/c5/c6 compliant, c4 Loop; Vision offen |
| Huihui GSQ-RCO-IQ3_S | uncensored | eris | t1: **0 Content**; t2+Gradient+Vision queued |
| bartowski/orcarouter IQ3_XS | uncensored | eris | komplett queued |
| orcarouter IQ3_M | uncensored | uranus | komplett queued (nach hauhau-default) |
| UD-IQ4_XS | aligned | uranus | ggf. Referenz-Block queued |

Fail-Signaturen bisher: HauhauCS + Huihui zeigen bei langen Coding-Tasks
mid-sentence-EOS im Reasoning → 0 Content trotz `finish_reason=stop`;
swift-al task2 stoppte mid-Code; swift-unc c4 zeigte Tail-Repeat-Loop.
AP + Swift-al task1 + Swift-unc t1/t2 lieferten sauber.

## Ablauf / Resume-Verfahren

1. Vor Server-Start: `ss -ltn | grep 18088`, `nvidia-smi` — kein fremder
   Prozess, VRAM frei.
2. Server starten (`q38-start-eval.sh`, remote /tmp), echte PID + Log notieren.
3. Probe-Request (kurz) → dann Task 1, dann Task 2 via
   `python3 qwen38-eval-client.py 18088 taskN.txt out.json label`.
4. Nach jedem Run: JSON nach `runs/<run_id>.json` kopieren,
   `results.jsonl`-Zeile anhängen. **Resume = fehlende run_id in
   results.jsonl nachholen.**
5. Nach Serienende: Server per PID stoppen, Eris `eris-chat.service`
   starten + `/health` prüfen.
6. Auswertung: Vergleichstabelle Quality/Speed/Token-Effizienz →
   `../SESSION_PLAN_qwen38-eval.md` (kanonisch), HTML-Lineage erst nach
   stabilen Ergebnissen.
