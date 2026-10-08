# tests/ — AGENTS.md (Child-DOX)

**Zweck:** Unit-Tests und Test-Programme fuer Core-Funktionalitaet, Backends, Sampling, Chat-Templates, Grammatik, Tokenizer, Modell-Architekturen, GGUF, spekulative Decodierung (MTP) und TurboQuant.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Pfade unter `tests/`.

---

## Lokale Vertraege

### Testdaten und Pfade
- Test-Vokabulare und Modell-Stubs liegen in `models/ggml-vocab-*.gguf` (Repository-Root).
- Keine lokalen absoluten Pfade in Test-Quellen oder CMake-Argumenten verwenden.
- Shell-Testskripte duerfen keine Host-spezifischen Annahmen enthalten.
- `test-state-restore-fragmented` prueft zusaetzlich Fullstate-Byte-Paritaet von drei nicht-unifizierten Streams. f16 ist die Kontrollgruppe; ein expliziter anderer K-Typ erfordert ein dazu kompatibles Modell und zaehlt nicht als geprueft, wenn nur f16 lief.
- `python3 tests/test-bench-flashnext-mtp.py` prueft Runner-Logik mit HTTP-/Prozess-Mocks. Danach echte CLI-/Server-Probe vor GPU-Matrix; Mock-PASS belegt weder unterstuetzte Flags noch MTP-Korrektheit.

### Test-Framework
- `testing.h` — Gemeinsames Header-Framework fuer Test-Makros und Assertions.
- Python-Tests (z. B. `test-tokenizer-0.py`) sind optional und ergaenzen C++-Tests.

---

## Arbeitsanleitung

### Einzelnen Test bauen
```bash
cmake --build build --target test-<name>
```
Beispiele:
- `test-backend-ops`
- `test-backend-sampler`
- `test-chat-template`
- `test-grammar-parser`
- `test-llama-archs`
- `test-gguf`
- `test-turbo-quant`
- `test-expected-attention`
- `test-tokenizer-0`, `test-tokenizer-1-bpe`, `test-tokenizer-1-spm`

### Alle Tests bauen und laufen lassen
```bash
ctest --test-dir build
```

### Einzelnen Test direkt ausfuehren
```bash
./build/bin/test-<name>
```

**Hinweis:** `test-speculative-mtp.cpp` existiert als Quelle, ist aber nicht in `tests/CMakeLists.txt` registriert und daher nicht als Target buildbar — vor einer moeglichen Registrierung ist eine API-Port-/Kompatibilitaets-Pruefung noetig (die Quelle stammt aus einer aelteren API-Generation); siehe `docs/DOX_AUDIT.md`. Andererseits sind z.B. `test-turbo-quant` (`test-turbo-quant.c`), `test-expected-attention` (`test-expected-attention.cpp`), `test-save-load-state`, `test-state-restore-fragmented` und `test-recurrent-state-rollback` registriert (die `model`-LABELs nutzen die CMake-Modell-Fixture `FIXTURES_REQUIRED test-download-model` mit `ARGS -m "${MODEL_DEST}"`; uebersprungene Modell-/dtype-/backend-Faelle zaehlen nicht als bestanden).

### Neue Tests hinzufuegen
1. Quelltext `test-<name>.cpp` (oder `.c`, `.py`, `.sh`) erstellen.
2. In `tests/CMakeLists.txt` mit `llama_build()`, `llama_test()` oder `llama_build_and_test()` registrieren.
3. Falls Modelldaten benoetigt werden: `get-model.cpp` / `gguf-model-data.cpp` als Hilfsmittel verwenden.

---

## Verifikation

- [ ] Betroffener Test baut erfolgreich (`cmake --build build --target test-<name>`)
- [ ] Test laeuft durch (`./build/bin/test-<name>` oder `ctest --test-dir build -R <name>`)
- [ ] `tests/CMakeLists.txt` konsistent mit neuen/entfernten Quellen
- [ ] Keine lokalen Pfade oder Host-Namen in Test-Code oder CMake eingefuehrt

---

## Child-DOX-Index

Keine Children — `tests/` ist ein Blattverzeichnis.
