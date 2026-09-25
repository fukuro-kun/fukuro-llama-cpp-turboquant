# Parallele Schema-Entscheidungen (Jev-Ansatz)

Dieser Modus setzt den in Trilium als **Jev** beschriebenen Ansatz als `parallel-decision` um; er ist keine Jev-Produktimplementierung. Er ist weder eine neue Modellarchitektur noch ein neues Modell: Ein bestehendes GGUF-Modell bewertet endliche, vorgegebene Kandidatenmengen parallel, statt ein JSON-Objekt frei und Token für Token zu generieren.

Für jedes Schemafeld werden die zulässigen Werte in Tokenpfade zerlegt. Nach einem gemeinsamen Prompt-Präfix werden die Kandidatenzweige aus demselben KV-Kontext abgeleitet und gebündelt ausgewertet. Das Ergebnis wird anschließend aus den ausgewählten, typisierten Werten im Programm zusammengesetzt. Dadurch kann die Ausgabe keine zusätzlichen Felder oder Werte außerhalb des Schemas enthalten. **Inhaltliche Fehlentscheidungen verhindert das Verfahren nicht.** Die ausgegebenen Wahrscheinlichkeiten sind Scores innerhalb der Kandidatenentscheidung, keine kalibrierten Vertrauenswerte.

Die Implementierung enthält eine gemeinsame Engine (`decision-engine.*`), das Werkzeug `llama-parallel-decision` und die Server-Endpunkte `POST /decision` sowie `POST /v1/decision`.

## Bauen

Die Werkzeuge werden wie der übrige Fork gebaut. Die CMake-Konfiguration muss das gewünschte Backend aktivieren, zum Beispiel `-DGGML_CUDA=ON` oder `-DGGML_VULKAN=ON`.

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --target llama-server llama-parallel-decision -j$(nproc)
```

## Server starten

`--decision-seqs N` reserviert zusätzliche Sequenzen oberhalb der normalen Server-Slots und aktiviert den Endpunkt. Zulässig sind `0` (deaktiviert) oder mindestens `3`. Eine Sequenz hält das statische Präfix, weitere Sequenzen dienen den Kontextstämmen und Kandidatenzweigen. Der Server erweitert entsprechend `n_seq_max` und aktiviert automatisch den vereinheitlichten KV-Cache (`kv_unified`), damit Zweige die KV-Zellen ihres Stamms teilen können. Der Endpunkt funktioniert nur auf generativen Servern, nicht mit `--embedding` oder `--reranking`.

```bash
./build/bin/llama-server -m model.gguf -np 1 -c 32768 --decision-seqs 12
```

Die passende Sequenzzahl hängt von Modellarchitektur, Kontextlänge und Backend ab. Sliding-Window- und rekurrente/hybride Modelle können pro Sequenz zusätzlichen Zustand benötigen; beginne daher konservativ und messe Speicherbedarf sowie Durchsatz auf dem Zielsystem. Auf rekurrenten und hybriden Modellen werden kürzere Zweige rechts auf die Länge des längsten Zweigs in ihrer Gruppe aufgefüllt. Die Scores werden am letzten echten Token gelesen; das kausale Padding danach beeinflusst diese Scores nicht, verursacht aber zusätzliche Rechenarbeit.

**TurboQuant-Hinweis:** Mit `--decision-seqs` wird der vereinheitlichte KV-Cache erzwungen. Validiert sind CPU (Gemma-4-12B IQ4_NL, turbo4/turbo3) und CUDA auf drei Produktivsystemen: Gemma-4-26B-A4B-QAT (turbo4/turbo3, MoE-Offload via `--fit`) sowie das hybride Mamba-MoE Granite-4.0-H-Tiny (turbo4/turbo4, Rechtsauffüllung aktiv, geteilter 8-GB-VRAM). Weitere Modell-/Backend-Kombinationen benötigen eigene Laufzeittests.

**Vulkan-Bekannteinschränkung (2026-09):** Auf dem Vulkan-Backend (RADV, AMD Radeon 760M iGPU, Gemma-4-26B-A4B-QAT, turbo4/turbo3) bremst der vereinheitlichte KV-Cache den gesamten Inferenzpfad auf ~0,1 Token/s (statt ~38 t/s Prefill) — Decision-Ergebnisse sind korrekt, aber ~190 s statt ~100 ms langsam. Ursache vermutlich ein unoptimierter Vulkan-Pfad im Unified-KV-Layout; untersucht, bis dahin `--decision-seqs` auf Vulkan nicht einsetzen.

### Gemessene Latenzen (Fork, `scripts/bench-decision.py`, 2 Kontexte × enum+boolean)

| System | Modell | KV | Decision gesamt (med) | davon Scoring | Chat-Completion (med) |
|---|---|---|---:|---:|---:|
| CUDA, MoE-Offload | Gemma-4-26B-A4B QAT | turbo4/3 unified | ~3850 ms | ~540 ms | ~70650 ms, 0/5 im Enum-Raster |
| CUDA | Gemma-4-26B-A4B QAT | turbo4/3 unified | ~85 ms | ~28 ms | ~5940 ms, 0/5 im Enum-Raster |
| CUDA (VRAM-geteilt, ngl 20) | Granite-4.0-H-Tiny | turbo4/4 unified | ~764 ms | ~296 ms | ~949 ms, 0/5 im Enum-Raster |

Breiteres Schema (5 Felder inkl. `integer` 0–90, Arena-Preset des Playgrounds): Eris ~435 ms, MoE-Offload-Hydra ~14,2 s pro Entscheidung — die Kostentreiber sind Kandidatenzahl und Prefill, nicht die Feldanzahl.

Der Chat-Vergleich verwendet dieselbe Aufgabe ohne Schema-Zwangsliste: Die Modelle antworten zwar häufig mit JSON, erfinden aber eigene Kategorie-Bezeichnungen oder verpacken die Antwort in Markdown-Fences — im freien Chat ist die endliche Wertemenge nicht zuverlässig einzuhalten. Die Scoring-Zeit wächst annähernd unabhängig von der Modell-Generationsgeschwindigkeit; der Aufwand pro Entscheidung liegt im Wesentlichen im Kontext-Prefill, der durch den Prefix-Cache (`cached_tokens`) minimiert wird.

## `POST /v1/decision`

`contexts` enthält 1 bis 256 nichtleere Texte. Sie werden mit demselben Schema und denselben Anweisungen ausgewertet; die Ergebnisse stehen in derselben Reihenfolge. Bei vielen Kontexten oder Kandidaten kann die Engine mehrere Gruppen beziehungsweise Decode-Runden benötigen. Chat-Templates werden mit deaktiviertem Thinking gerendert.

Im Routerbetrieb kann `model` wie bei den übrigen Server-Endpunkten zur Modellauswahl mitgegeben werden.

```bash
curl http://localhost:8080/v1/decision \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-12b",
    "instructions": "Ordne die Anfrage ein und bewerte ihre Dringlichkeit.",
    "schema": {
      "category": {
        "type": "enum",
        "choices": ["billing", "technical", "cancellation", "other"],
        "description": "Welche Art von Support-Anfrage ist es?"
      },
      "urgent": {
        "type": "boolean",
        "description": "Muss die Anfrage dringend bearbeitet werden?"
      },
      "priority": {
        "type": "enum",
        "choices": ["low", "medium", "high", "critical"],
        "description": "Wie hoch ist die Priorität?"
      }
    },
    "contexts": ["Mir wurde der Betrag doppelt berechnet; bitte heute korrigieren."]
  }'
```

Die Antwort enthält eine schema-konforme Entscheidung, Feld-Scores, Nutzungsdaten und Laufzeiten:

```json
{
  "object": "decision",
  "results": [
    {
      "decision": {"category": "billing", "urgent": true, "priority": "high"},
      "fields": {
        "category": {"value": "billing", "probability": 0.91, "scored_nodes": 1, "tree": true},
        "urgent": {"value": true, "probability": 0.82, "scored_nodes": 1, "tree": true},
        "priority": {"value": "high", "probability": 0.74, "scored_nodes": 1, "tree": true}
      },
      "usage": {"context_tokens": 21, "scored_rows": 14}
    }
  ],
  "usage": {"prompt_tokens": 137, "cached_tokens": 116, "context_tokens": 21, "scored_rows": 14},
  "timings": {"prefill_ms": 50.7, "scoring_ms": 50.0, "total_ms": 100.7, "rounds": 1, "per_decision_ms": 100.7}
}
```

Die Zahlen im Beispiel veranschaulichen nur das Antwortformat und sind kein Benchmark dieses Forks. `probability` ist im `tree`-Modus auf die zulässigen Kandidaten beschränkt; im `greedy`-Modus ist es der Score des gewählten Pfads. Keiner der Werte ist als kalibrierte Sicherheit zu interpretieren.

## Schema

Unterstützt werden kompakte Felddefinitionen sowie ein JSON-Schema-Objekt mit `properties`. Kompakte Definitionen benötigen eine `description`; im JSON-Schema ist sie optional.

| Typ | Parameter | Einschränkung |
|---|---|---|
| `enum` | `choices` oder `enum` | Zeichenketten, 1–255 Werte |
| `boolean` | – | `true` oder `false` |
| `integer` | `minimum`, `maximum` | inklusive Grenzen, höchstens 255 Werte |
| `number` | `minimum`, `maximum`, `step` (JSON-Schema: `multipleOf`) | Raster muss beide Grenzen enthalten, höchstens 255 Werte |

Ein Schema darf 1 bis 32 Felder enthalten. Kandidaten, die nach der Tokenisierung kollidieren, werden zurückgewiesen. Felder werden unabhängig voneinander bewertet; ein Feld kann daher nicht vom Ergebnis eines anderen Feldes abhängen.

Numerische Felder akzeptieren `aggregate`: `mode` (Standard), `median` oder `mean`. `median` und `mean` verwenden die vollständige Kandidatenverteilung und sind deshalb nur im `tree`-Modus wirksam; die Antwort kann außerdem ein `interval_p10_p90` enthalten.

## Optionen

| Feld | Standard | Bedeutung |
|---|---:|---|
| `instructions` | `""` | zusätzliche Anweisungen zum erzeugten Feldkatalog; Teil des cachebaren Präfixes |
| `mode` | `auto` | `tree` bewertet alle Verzweigungsknoten; `greedy` folgt jeweils dem besten Token; `auto` wählt je Feld abhängig von `tree_max` |
| `tree_max` | `128` | Obergrenze der Kandidatenzahl für `tree` im Modus `auto` |
| `cache_prompt` | `true` | passendes statisches Präfix zwischen Anfragen wiederverwenden |

## CLI-Worker

`llama-parallel-decision` kann als Worker mit einem zeilenbasierten stdin/stdout-Protokoll gestartet werden. Das Protokoll ist für `llama-mojo` und dessen vorbereitete Felddateien kompatibel; die meisten Anwendungen sollten stattdessen den Server-Endpunkt verwenden.

Umgebungsvariablen: `DECIDE_TREE` (`1` = tree, `0` = greedy, sonst auto), `DECIDE_TREE_MAX` (Standard `128`), `DECIDE_NSEQ` (Sequenzbudget, Standard `24`) und `DECIDE_SPLIT_BOUNDARY=1` (veraltete Tokenisierung).

## Einsatz und Fallback

Der Modus eignet sich für unabhängige Klassifikations-, Routing- und Ja/Nein-Entscheidungen mit begrenztem Wertebereich. Er löst keine Abhängigkeiten zwischen Feldern und führt selbst keine Eskalation aus. Ein aufrufendes System kann niedrige oder mehrdeutige Scores als Anlass nehmen, dieselbe Anfrage an eine normale Chat-Completion mit einem größeren Modell weiterzugeben.

## API-Smoke-Test

Der End-to-End-Test startet einen Server ohne GPU-Offload und prüft einen erfolgreichen `tree`-Entscheid sowie eine ungültige Anfrage. Er verwendet standardmäßig das kleine TinyLlama-Testmodell; falls es noch nicht lokal verfügbar ist, lädt die Testumgebung es herunter.

```bash
cd tools/server/tests
uv run --no-project --with-requirements requirements.txt pytest --confcutdir=unit unit/test_decision.py -q
```

Für einen Test mit einem anderen lokalen GGUF können `LLAMA_DECISION_TEST_MODEL`, `LLAMA_DECISION_TEST_CTK` und `LLAMA_DECISION_TEST_CTV` gesetzt werden. Der Test läuft weiterhin ausschließlich auf der CPU; damit lassen sich insbesondere TurboQuant-K/V-Typen mit `kv_unified` funktional prüfen, nicht aber GPU-Leistung oder -Korrektheit.

## Playground

[decision-playground](https://github.com/thecodacus/decision-playground) ist eine Browser-Oberfläche, die den Entscheidungs-Endpunkt einer normalen Chat-Completion gegenüberstellt und eine kleine Simulation mit darüber gesteuerten Agenten enthält.
