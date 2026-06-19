# scripts/ — AGENTS.md (Child-DOX)

**Zweck:** Operationale Hilfsskripte fuer Build-Patches, Server-Start, Benchmarks, GGUF-Extraktion/Verifikation und Quantisierung. Kein Core-Code — diese Skripte automatisieren und begleiten die Inference Engine.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Dateien und Unterverzeichnisse unter `scripts/`.

---

## Lokale Vertraege

### Abgrenzung zum Core

- Core-Logik (Modell-Architektur, Tensor-Ops, Server-Engine) liegt in `src/`, `ggml/`, `common/` und `tools/`.
- Skripte in diesem Verzeichnis starten, messen, verifizieren und konvertieren — sie implementieren keine neue Inferenzlogik.

### Datenhygiene

- Keine hartkodierten lokalen Pfade, Host-Namen oder Zugangsdaten in Skripten.
- Konfiguration (Modelldateien, Kontextlaengen, GPU-Layer) wird ueber Umgebungsvariablen oder benutzergesteuerte Parameter eingebracht.

### Konvertierungs-Skript

- `convert_hf_to_gguf.py` liegt im **Repository-Root**, nicht unter `scripts/`. Wird hier referenziert, aber nicht verschoben.

---

## Arbeitsanleitung

### Build

| Skript | Zweck |
|--------|-------|
| `build-cuda-glibc-patch.sh` | Temporaerer Patch fuer `mathcalls.h` bei glibc >= 2.43 vor CMake-Build. |

### Server-Start (Fork-Features)

| Skript | Zweck |
|--------|-------|
| `run-gemma4-31b-mtp-server.sh` | Gemma 4 3.1B MTP-Server starten. |
| `run-gemma4-e2b-mtp-server.sh` | Gemma 4 E2B MTP-Server starten. |
| `run-gemma4-e4b-mtp-mmproj-server.sh` | Gemma 4 E4B multimodaler MTP-Server starten. |

- Anpassen vor Verwendung: Kontextlaenge, GPU-Layer, Modellpfad, Port.

### Benchmarks

| Skript | Zweck |
|--------|-------|
| `bench-*.sh` / `bench-*.py` | Performance- und Qualitaetsbenchmarks fuer Matrix-Tests, parallele Anfragen, TurboQuant-Setups. |
| `bench-mtp-matrix.sh` | Universelles MTP-Matrix Benchmark (konfigurierbar fuer alle Modelle). CSV-Output, VRAM-Monitoring. |
| `bench-mtp-matrix-196k.sh` | Vollstaendige MTP-Matrix @ ctx=196608 fuer Gemma-4 12B (9 Targets × 12 Drafts). |
| `bench-b1-vulkan-q3k-q6k.sh` | Vulkan Q3_K/Q6_K Block-Load Benchmark (Cherry-Pick B1 Ergebnisse). Siehe [docs/fork/2026-06-17_B1_VULKAN_Q3K_Q6K_BENCHMARK.md](../docs/fork/2026-06-17_B1_VULKAN_Q3K_Q6K_BENCHMARK.md). |
| `bench-qwen-udt-matrix-local.sh` | Lokale Matrix-Benchmarks fuer Qwen UDT. |
| `bench-qwen-udt-quality.sh` | Qualitaetsbewertung fuer Qwen UDT. |

**Benchmark-Best-Practices (gelernt aus InferenzQuelle):**

1. **GPU-Bereinigung vor jedem Test:** `killall -9 llama-cli` + 3s Pause (verhindert OOM bei aufeinanderfolgenden Tests).
2. **Datei-Pruefung:** Target und Draft vor dem Test auf Existenz pruefen (vermeidet Exit 127).
3. **Output in Variable:** Statt direkte Log-Umleitung — Output in Variable speichern, dann parsen (sauberes Extrahieren von Metriken).
4. **sync nach CSV-Eintrag:** `sync` nach jedem `echo >> "$CSV"` (Persistenz bei Absturz oder SIGKILL).
5. **Exit 124 als OK:** Timeout (exit 124) als erfolgreich behandeln, wenn Performance-Daten vorhanden sind.
6. **Lock-File:** Wrapper mit Lock-File (`/tmp/<name>.lock`) verhindert parallele Ausfuehrung.
7. **setsid:** Bei SSH-Start `setsid` verwenden, damit der Prozess nicht an die SSH-Session gebunden ist (systemd beendet sonst User-Prozesse beim Logout).

**Voraussetzung:** Kompilierte Binaries in `build/` und vorbereitete GGUF-Modelle.

### Quantisierung

| Skript | Zweck |
|--------|-------|
| `quantize-gemma4-edge-assistant-mtp.sh` | Gemma 4 Edge Assistant MTP quantisieren. |
| `quantize-qwen-udt.sh` | Qwen UDT Quantisierung. |
| `quantize-qwen-udt-matrix.sh` | Matrix-gesteuerte Qwen UDT Quantisierung. |
| `quantize-masks/` | Quantisierungsmasken als Text-Referenz. |

### GGUF-Verifikation und Extraktion

| Skript | Zweck |
|--------|-------|
| `extract-qwen36-nextn-gguf.py` | NextN-Daten aus Qwen 3.6 GGUF extrahieren. |
| `verify-gemma4-assistant-gguf.py` | Gemma 4 Assistant GGUF verifizieren. |
| `verify-qwen36-nextn-gguf.py` | Qwen 3.6 NextN GGUF verifizieren. |

### Weitere thematische Unterverzeichnisse

- `apple/` — Apple-Plattform-Validierung (iOS, macOS, tvOS, visionOS).
- `autoresearch/` — Automatisierte Forschungs-Track-Experimente.
- `qwen-udt/` — Qwen UDT spezifische Hilfsskripte (Download, Upload, Remote-Bootstrap).
- `jinja/` — Jinja-Template-Tester.

---

## Verifikation

- [ ] Build-Skripte fuehren zu erfolgreichem `cmake --build build`.
- [ ] Server-Startskripte starten ohne sofortige Fehlermeldung und binden korrekt.
- [ ] Benchmarks terminieren und liefern reproduzierbare Ausgaben.
- [ ] Neue Skripte enthalten keine hartkodierten Pfade oder private Daten.
- [ ] Neue Benchmark-Skripte implementieren GPU-Bereinigung, Datei-Pruefung und sync.
- [ ] Wrapper-Skripte mit Lock-File und setsid bei SSH-Remote-Ausfuehrung.

---

## Child-DOX-Index

Keine eigenen Child-DOX-Grenzen in `scripts/`. Die thematischen Unterverzeichnisse (`apple/`, `autoresearch/`, `qwen-udt/`, `quantize-masks/`, `jinja/`) sind Hilfsgruppierungen ohne eigenstaendige Vertragsdomaene. Keine separaten AGENTS.md erforderlich, solange sie keinen eigenen Zweck, Workflow oder Qualitaetsstandard definieren, der ueber dieses Dokument hinausgeht.
