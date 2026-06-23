# tools/ — AGENTS.md (Child-DOX)

**Zweck:** Ausfuehrbare Werkzeuge der Inference Engine. Jeder Unterordner enthaelt ein eigenstaendiges Binary mit CMake-Target. `tools/server/` ist das Hauptwerkzeug fuer Produktiv-Inferenz.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Pfade unter `tools/`.

---

## Lokale Vertraege

### Hauptwerkzeug: `tools/server/`

- **llama-server** ist die primaere Inferenzschnittstelle fuer das Hauptprojekt.
- OpenAI-kompatible HTTP-API (`/v1/chat/completions`, `/v1/embeddings`, etc.).
- Unterstuetzt Multimodal (CLIP via `mtmd/`) und spekulative Decodierung (Gemma 4 MTP, Qwen NextN).
- Schluesseldateien: `server.cpp`, `server-context.cpp`, `httplib.h`, `index.html.hpp`.

### Werkzeug-Hierarchie

| Werkzeug | Zweck |
|----------|-------|
| `server/` | OpenAI-kompatible HTTP-API, multimodal + spekulativ |
| `cli/` | Interaktiver Chat im Terminal |
| `llama-bench/` | Performance-Benchmark fuer unterschiedliche Parameter |
| `perplexity/` | Perplexity-Messung auf Datensaetzen |
| `quantize/` | GGUF-Quantisierung (inkl. TurboQuant) |
| `mtmd/` | Multimodal-Projektor (CLIP-Encoder fuer Server) |
| `gguf-split/` | GGUF-Dateien splitten/mergen |
| `imatrix/` | Importance Matrix fuer Quantisierung |
| `batched-bench/` | Batched-Inferenz-Benchmark |
| `tokenize/` | Tokenizer-Test und -Debugging |
| `completion/` | Text-Completion ohne interaktiven Modus |
| `cvector-generator/` | Control-Vektor-Generierung |
| `export-lora/` | LoRA-Gewichte exportieren |
| `fit-params/` | Parameter-Fitting fuer Modelle |
| `parser/` | Grammatik-Parser fuer constrained sampling |
| `rpc/` | RPC-Backend-Server fuer verteilte Inferenz |
| `tts/` | Text-to-Speech Werkzeug |

### Build

- Jedes Werkzeug ist ein eigenes CMake-Target: `llama-server`, `llama-cli`, `llama-bench`, etc.
- Alle bauen automatisch mit `cmake --build build -j$(nproc)`.
- Keine zusaetzlichen Abhaengigkeiten fuer die meisten Werkzeuge; `server/` bindet `common/` und `mtmd/` ein.

---

## Arbeitsanleitung

### Aenderungen an Werkzeugen

1. **Lesen:** Zustaendiges Child-AGENTS.md (falls vorhanden) und dieses Dokument.
2. **Code:** C++-Quellen bearbeiten; `server/` und `mtmd/` haben die hoechste Komplexitaet.
3. **Build:** `cmake --build build --target llama-<werkzeug> -j$(nproc)`
4. **Test:** Binary unter `build/bin/llama-<werkzeug>` ausfuehren.
5. **DOX:** Bei Aenderungen an Zweck, Schnittstelle oder Verhalten dieses AGENTS.md aktualisieren.

### Neue Werkzeuge

- Neues Unterverzeichnis unter `tools/` anlegen.
- `CMakeLists.txt` im Werkzeug-Ordner mit `add_executable()` und `target_link_libraries()`.
- In `tools/CMakeLists.txt` mit `add_subdirectory()` einbinden.
- Werkzeug in die Tabelle unter "Werkzeug-Hierarchie" eintragen.

---

## Verifikation

- [ ] `cmake --build build -j$(nproc)` erfolgreich
- [ ] Betroffene Binaries unter `build/bin/` ausfuehrbar
- [ ] Bei `server/`: API-Smoke-Test (z.B. `/health` oder Chat-Completion)
- [ ] Keine lokalen Pfade oder Host-Namen im Code
- [ ] Betroffene AGENTS.md aktualisiert

---

## Child-DOX-Index

| Pfad | Zweck | Status |
|------|-------|--------|
| `tools/server/` | OpenAI-kompatible HTTP-API, multimodal + spekulativ | [x] Aktiv |
