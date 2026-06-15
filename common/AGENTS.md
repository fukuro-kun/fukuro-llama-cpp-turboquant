# common/ — AGENTS.md (Child-DOX)

**Zweck:** Gemeinsame Utilities fuer `llama-cli`, `llama-server` und weitere Tools. Enthaelt CLI-Argument-Parsing, Sampling-Logik, Chat-Formatierung, N-Gramm-Caches, Grammatik-Steuerung und den Draft-Treiber fuer spekulative Decodierung.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Dateien unter `common/`.

---

## Lokale Vertraege

### Fork-spezifische Schluesseldateien

| Modul | Dateien | Beschreibung |
|-------|---------|-------------|
| CLI-Argumente | `arg.cpp`, `arg.h` | Paramater-Definition und Parsing fuer alle Binaerdateien |
| Hilfsfunktionen | `common.cpp`, `common.h` | String-Utils, File-IO, Konsolen-Interaktion, Batching |
| Sampling | `sampling.cpp`, `sampling.h` | Top-K/Top-P/Min-P, Temperature, Repeat-Penalty, Mirostat |
| Spekulative Decodierung | `speculative.cpp`, `speculative.h` | **Draft-Treiber** fuer MTP (Gemma 4) und NextN (Qwen) |
| Chat-Formatierung | `chat.cpp`, `chat.h` | Template-Verarbeitung, Jinja-Engine-Anbindung |
| N-Gramm-Caches | `ngram-cache.cpp`, `ngram-cache.h`, `ngram-map.cpp`, `ngram-map.h`, `ngram-mod.cpp`, `ngram-mod.h` | Kontextbasierte N-Gramm-Vorhersage fuer Drafting |
| Grammatik-Steuerung | `llguidance.cpp` | Integration von llguidance fuer strukturierte Ausgaben |

### Abhaengigkeiten

- `speculative.cpp` greift auf `src/llama-context.cpp` und Modell-spezifische Draft-Logik zu (Gemma 4 MTP, Qwen NextN).
- Aenderungen an Sampling-Parametern (`sampling.cpp`) koennen `tools/server/server-context.cpp` und `tools/llama-cli/` betreffen.
- `chat.cpp` bindet Jinja-Templates aus `common/jinja/` ein.

---

## Arbeitsanleitung

### Aenderungen an `speculative.cpp` / `speculative.h`

Dies ist der zentrale Draft-Treiber des Forks. Bei Arbeit hier:

1. **MTP (Gemma 4):** Pruefe Interaktion mit `src/models/gemma4-assistant.cpp` und `src/llama-context.cpp`.
2. **NextN (Qwen):** Pruefe Interaktion mit `src/models/qwen35-nextn.cpp` / `qwen35moe-nextn.cpp`.
3. **Verifizierung:** Spekulativen Benchmark laufen lassen (`llama-bench` mit `--draft` bzw. Server-Endpoint testen).

### Aenderungen an `arg.cpp` / `arg.h`

- Neue Flags in `arg.cpp` muessen in allen konsumierenden Binaerdateien (`tools/cli/`, `tools/server/`, etc.) validiert werden.
- Flag-Namen und Defaults konsistent mit upstream llama.cpp halten, ausser Fork-spezifische Erweiterungen.

### Aenderungen an `sampling.cpp`

- Sampling-Aenderungen koennen deterministische Ausgaben veraendern — bei Regressionstests auf Token-Gleichheit achten.

### Aenderungen an `chat.cpp`

- Chat-Template-Aenderungen koennen alle Chat-Interfaces betreffen (CLI-Chat-Modus, Server-Chat-Completions).

---

## Verifikation

- [ ] `cmake --build build -j$(nproc)` erfolgreich
- [ ] Bei Aenderungen an `speculative.cpp`: Spekulativer Decodierungs-Benchmark oder Server-Test durchgefuehrt
- [ ] Bei Aenderungen an `sampling.cpp`: Sampling-Tests (`./build/bin/test-sampling` falls vorhanden) bestanden
- [ ] Keine privaten Daten in Commits (siehe Root-DOX)
- [ ] Betroffenes Child-DOX und Root-DOX geprueft, ggf. aktualisiert

---

## Child-DOX-Index

| Pfad | Zweck | Status |
|------|-------|--------|
| `common/jinja/` | Jinja-Template-Engine fuer Chat-Formatierung | [x] Aktiv |

*Keine weiteren Child-DOX erwartet. `common/` ist ein flaches Utility-Verzeichnis.*
