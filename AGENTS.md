# fukuro-llama-cpp-turboquant — AGENTS.md (Root-DOX)

**Zweck:** Fork von llama.cpp mit TurboQuant KV-Kompression, Gemma 4 MTP, Qwen NextN spekulativer Decodierung und DiffusionGemma-Integration. Inference Engine (Motor) fuer das Hauptprojekt InferenzQuelle. Siehe [FORKS.md](FORKS.md) fuer die vollstaendige Fork-Lineage und den Feature-Vergleich.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Pfade unter Repository-Root.

---

## Kernvertrag (DOX-Framework)

- **AGENTS.md-Dateien sind verbindliche Arbeitsvertraege** fuer ihre Unterbaeume.
- Arbeitsergebnisse, Quellmaterialien, Anweisungen, Aufzeichnungen, Ressourcen und dauerhafte Dokumente muessen aus dem naechstgelegenen anwendbaren AGENTS.md sowie jedem darueber liegenden Parent-AGENTS.md verstaendlich bleiben.
- **Verlass dich nicht auf dein Gedaechtnis.** Lies die anwendbare DOX-Kette in der aktuellen Sitzung erneut, bevor du bearbeitest.

### Vor dem Bearbeiten lesen

1. Lies dieses Root-AGENTS.md.
2. Identifiziere jede Datei oder jeden Ordner, den du bearbeiten willst.
3. Gehe vom Repository-Root zu jedem Zielpfad.
4. Lies jedes AGENTS.md, das du auf jedem Weg findest.
5. Wenn ein Parent-AGENTS.md ein Child-AGENTS.md auffuehrt, dessen Geltungsbereich den Pfad enthaelt, lies dieses Child und fahre von dort fort.
6. Verwende das naechstgelegene AGENTS.md als lokalen Vertrag und Parent-Dokumente fuer Repository-weite Regeln.
7. Wenn Dokumente widerspruechlich sind, hat das naehere Dokument die Kontrolle ueber lokale Arbeitsdetails, aber kein Child-Dokument darf DOX abschwaech.

### Nach dem Bearbeiten aktualisieren

Jede bedeutsame Aenderung erfordert einen DOX-Durchlauf, bevor die Aufgabe erledigt ist.

Aktualisiere das naechstgelegene zustaendige AGENTS.md, wenn eine Aenderung betrifft:
- Zweck, Geltungsbereich, Eigentuemer oder Verantwortlichkeiten
- Dauerhafte Struktur, Vertraege, Workflows oder Betriebsregeln
- Erforderliche Eingaben, Ausgaben, Berechtigungen, Einschraenkungen, Nebenwirkungen oder Artefakte
- Benutzerpraeferenzen zu Verhalten, Kommunikation, Prozess, Organisation oder Qualitaet
- AGENTS.md-Erstellung, -Loeschung, -Verschiebung, -Umbenennung oder -Index-Inhalte

Aktualisiere Parent-Dokumente, wenn sich Struktur, Eigentuemer, Workflow oder Child-Index auf Parent-Ebene aendert. Aktualisiere Child-Dokumente, wenn Parent-Aenderungen lokale Regeln veraendern. Entferne veralteten oder widerspruechlichen Text sofort. Kleine Bearbeitungen, die Verhalten oder Vertraege nicht aendern, koennen Dokumente unveraendert lassen, aber der DOX-Durchlauf muss dennoch stattfinden.

---

## Hierarchie

- Das Root-AGENTS.md ist die DOX-Leitplanke: Projektweite Anweisungen, globale Praeferenzen, dauerhafte Workflow-Regeln und der Top-Level-Child-DOX-Index.
- Child-AGENTS.md-Dateien besitzen domenenspezifische Anweisungen und ihren eigenen Child-DOX-Index.
- Jedes Parent erklaert, was seine direkten Children abdecken und was beim Parent verbleibt.
- Je naeher ein Dokument an der Arbeit liegt, desto spezifischer und praktischer muss es sein.

---

## Form des Child-Dokuments

- Erstelle ein Child-AGENTS.md, wenn ein Ordner eine dauerhafte Grenze mit eigenem Zweck, Regeln, Verantwortlichkeiten, Workflow, Materialien oder Qualitaetsstandards wird.
- Die Arbeitsanleitung muss die aktuellen Standards des Projekts oder Benutzeranweisungen widerspiegeln; wenn es noch keine spezifischen Standards oder Anweisungen gibt, lasse sie leer.
- Die Verifikation muss eine bestehende Pruefung widerspiegeln; wenn noch kein Verifikations-Framework existiert, lasse sie leer und aktualisiere sie, sobald eines existiert.

**Standard-Reihenfolge der Abschnitte:**
1. Zweck
2. Eigentuemer
3. Lokale Vertraege
4. Arbeitsanleitung
5. Verifikation
6. Child-DOX-Index

---

## Stil

- Halte Dokumente praegnant, aktuell und betriebsbereit.
- Dokumentiere stabile Vertraege, keine Tagebucheintraege.
- Schreibe allgemeine Regeln in Parent-Dokumente und konkrete Details in Child-Dokumente.
- Bevorzuge direkte Aufzaehlungspunkte mit expliziten Namen.
- Dupliziere Regeln nicht ueber viele Dateien, ausser jeder Geltungsbereich braucht eine lokale Version.
- Loesche veraltete Notizen statt Geschichte zu erklaeren.
- Kuerze offensichtliche Aussagen, wiederholte Regeln, fehlplatzierte Details und Warnungen vor Risiken, die nicht mehr existieren.

---

## Lokale Vertraege

### Sicherheit: Keine privaten Daten in oeffentliche Repos

**Striktes Verbot:**
- ❌ Lokale Host-Namen
- ❌ Interne IPs oder Domains
- ❌ Lokale Dateipfade (`/home/fukuro/...`)
- ❌ Persoenliche Identifikatoren
- ❌ SSH-Keys, Tokens, Passwoerter

**Erlaubt:**
- ✅ GPU-Architekturen (NVIDIA CUDA, AMD ROCm/Vulkan, Intel)
- ✅ Generische Beschreibungen

**Pruefung vor jedem Commit:**
```bash
grep -ri "hydra\|uranus\|mars\|venus\|styx\|helene\|telesto\|/home/fukuro\|/media/fukuro" .
```
Falls Treffer → Bereinigen!

### Lokale Gegebenheiten

- Host-spezifische Pfade, GPU-Architekturen und Build-Besonderheiten stehen in `LOKAL.md` (in `.gitignore`, nicht committet).
- Agenten muessen `LOCAL.md` lesen, bevor sie Host-spezifische Aktionen durchfuehren.
- GPU-Architektur-Build-Matrix (generisch):
  - **Pascal (GTX 1070):** `-DLLAMA_CUDA=ON`, FP16 nur via emulation, kein FlashAttention
  - **Ampere/Ada (RTX 3070/4060):** Volle Feature-Unterstuetzung, FlashAttention, TurboQuant
  - **AMD iGPU/APU:** `-DLLAMA_VULKAN=ON`, ROCm experimentell

### Build-System

- **CMake** mit Backend-Optionen (`-DLLAMA_CUDA=ON`, `-DLLAMA_VULKAN=ON`, etc.)
- **Build-Verzeichnis:** `build/` (in `.gitignore`, nie committen)
- **glibc >= 2.43:** `scripts/build-cuda-glibc-patch.sh` verwenden (temporaerer Patch fuer `mathcalls.h`)

### Git-Workflow

- **Primary Remote:** `git@codeberg.org:fukuro/fukuro-llama-cpp-turboquant.git`
- **Branch:** `master` (Hauptentwicklung)
- **Commits:** Deutsch, kurz und sachlich (keine generischen Marketing-Floskeln)
- **Keine automatischen Devin-Eintraege** in Commit-Nachrichten

### Fork-Features und Schluesseldateien

| Feature | Primaere Dateien |
|---------|-----------------|
| TurboQuant KV/Weights | `ggml/src/ggml-turbo-quant.c`, `src/llama-quant.cpp` |
| Gemma 4 MTP | `src/models/gemma4-assistant.cpp`, `src/llama-context.cpp`, `common/speculative.cpp` |
| Qwen 3.6 NextN | `src/models/qwen35-nextn.cpp`, `src/models/qwen35moe-nextn.cpp` |
| Multimodal + Spec | `tools/server/server-context.cpp`, `docs/speculative.md` |
| DiffusionGemma | `src/models/diffusion-gemma.cpp`, `src/llama-model.cpp`, `tools/diffusion-cli/` |
| GGUF-Konvertierung | `convert_hf_to_gguf.py` |

---

## Arbeitsanleitung

### Schnelleinstieg fuer Aenderungen

1. **Code verstehen:** Zielverzeichnis scannen, zustaendiges Child-AGENTS.md lesen.
2. **Aendern:** Quelltext bearbeiten (C++, Python, CMake).
3. **Build testen:** `cmake --build build -j$(nproc)`
4. **Verifizieren:** Relevante Tests laufen lassen (`./build/bin/test-*`).
5. **DOX-Durchlauf:** Betroffene AGENTS.md aktualisieren.

### Wichtige Verzeichnisse (Uebersicht)

| Verzeichnis | Inhalt |
|-------------|--------|
| `src/` | C++ Core: Modell-Architekturen, Graphen, KV-Cache, Kontext |
| `ggml/` | GGML-Bibliothek: Tensor-Ops, Backends, Quantisierung, Speicher |
| `common/` | Gemeinsame Utilities: CLI-Argumente, Sampling, spekulative Decodierung |
| `tools/` | Ausfuehrbare Werkzeuge: `llama-server`, `llama-cli`, `llama-bench` |
| `tests/` | Unit-Tests und Test-Framework |
| `scripts/` | Build-Skripte, Benchmark-Automation, GGUF-Verifikation |
| `include/` | Oeffentliche C-API Header (`llama.h`, `llama-cpp.h`) |
| `docs/` | Upstream-Dokumentation (nicht direkt editieren, ausser Fork-Spezifika) |
| `examples/` | Beispielprogramme (upstream, selten aendern) |
| `models/` | Modell-Konfigurationen und Tokenizer-Modelle |
| `gguf-py/` | Python-Bibliothek fuer GGUF-Dateien |
| `ci/` | CI/CD Konfiguration |

---

## Verifikation

- [ ] Vor Arbeit: DOX-Kette gelesen (Root + alle Children auf dem Pfad)
- [ ] Nach Arbeit: Betroffene AGENTS.md aktualisiert
- [ ] Keine privaten Daten in oeffentliche Repos committet
- [ ] Build erfolgreich (`build/` existiert und enthaelt Zielbinaries)

---

## Child-DOX-Index

| Pfad | Zweck | Status |
|------|-------|--------|
| `src/` | C++ Core: Modell-Architekturen, Graphen, KV-Cache, Kontext | [x] Aktiv |
| `ggml/` | GGML-Bibliothek: Tensor-Ops, Backends, Quantisierung | [x] Aktiv |
| `common/` | Gemeinsame Utilities: CLI-Args, Sampling, spekulative Decodierung | [x] Aktiv |
| `tools/` | Ausfuehrbare Werkzeuge: Server, CLI, Bench, Quantize | [x] Aktiv |
| `tests/` | Unit-Tests, Test-Framework, Benchmarks | [x] Aktiv |
| `scripts/` | Build-Skripte, Benchmark-Automation, Konvertierung | [x] Aktiv |

*Hinweis: Bei Aenderungen an Zweck, Grenzen oder Qualitaetsstandards eines Verzeichnisses: Child-AGENTS.md aktualisieren und diesen Index pruefen.*

---

## Benutzerpraeferenzen

- fukuro bevorzugt **deutsche Sprache** in Commits und Dokumentation.
- fukuro wuenscht sich **keine Host-Namen** in oeffentlichen Repos.
- fukuro bevorzugt **SSH statt HTTPS** fuer Git-Authentifizierung.
