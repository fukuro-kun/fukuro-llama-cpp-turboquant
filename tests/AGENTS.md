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
- `test-speculative-mtp`
- `test-turbo-quant`
- `test-tokenizer-0`, `test-tokenizer-1-bpe`, `test-tokenizer-1-spm`

### Alle Tests bauen und laufen lassen
```bash
ctest --test-dir build
```

### Einzelnen Test direkt ausfuehren
```bash
./build/bin/test-<name>
```

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
