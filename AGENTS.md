# fukuro-llama-cpp-turboquant — AGENTS.md (Root-DOX)

**Zweck:** Fork von llama.cpp mit TurboQuant KV-Kompression, Gemma 4 MTP, Qwen NextN spekulativer Decodierung, DiffusionGemma-Integration und Vulkan-WHT-Optimierung. Inference Engine (Motor) fuer das Hauptprojekt InferenzQuelle. Siehe [FORKS.md](FORKS.md) fuer die vollstaendige Fork-Lineage und den Feature-Vergleich.

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

### Python-Umgebungen: Ausschliesslich uv

**Verbindliche Regel:** Python-Pakete werden **nur** via `uv` verwaltet. Nie `pip install --break-system-packages`, nie systemweite Python-Installationen ohne Isolation. Details (Installation, Pfade, zentrale venv) in `LOKAL.md` → "uv und Python-Umgebungen".

### Lokale Gegebenheiten

- Host-spezifische Pfade, GPU-Architekturen und Build-Besonderheiten stehen in `LOKAL.md` (in `.gitignore`, nicht committet).
- Agenten muessen `LOKAL.md` lesen, bevor sie Host-spezifische Aktionen durchfuehren.
- GPU-Architektur-Build-Matrix (generisch):
  - **Pascal (GTX 1070):** `-DLLAMA_CUDA=ON`, FP16 nur via emulation, kein FlashAttention
  - **Ampere/Ada (RTX 3070/4060):** Volle Feature-Unterstuetzung, FlashAttention, TurboQuant
  - **AMD iGPU/APU:** `-DLLAMA_VULKAN=ON`, ROCm experimentell
    - ✅ **turbo3 KV-Cache funktioniert:** `--cache-type-k turbo3 --cache-type-v turbo3` (~5.1x Kompression)
    - ✅ **turbo4 KV-Cache funktioniert:** `--cache-type-k turbo4 --cache-type-v turbo4` (~3.8x Kompression)
    - FlashAttention ist fuer beide TurboQuant-Formate auf Vulkan aktiv
    - ⚠️ **Performance-Klippe bei ~188k Kontext:** Auf AMD APU (shared memory) bricht die Inference-Performance bei ca. 188k Kontext scharf ein (24 t/s → 0.09 t/s, Faktor 243x). Dies ist KEIN VRAM-Bandbreiten-Problem (APU nutzt denselben DDR5), sondern vermutlich ein Code-Pfad-Wechsel im Vulkan-Backend. Workaround: Kontext auf maximal 180k begrenzen. Siehe `docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md` und Trilium `SWumEN7WOXBI` Abschnitt 5.8.

### Build-System

- **CMake** mit Backend-Optionen (`-DLLAMA_CUDA=ON`, `-DLLAMA_VULKAN=ON`, etc.)
- **Build-Verzeichnis:** `build/` (in `.gitignore`, nie committen)
- **glibc >= 2.43:** `scripts/build-cuda-glibc-patch.sh` verwenden (temporaerer Patch fuer `mathcalls.h`)

### Gemma-4 Standard-Sampling-Parameter

Fuer alle Gemma-4 Modelle (sofern kein begruendeter Spezialfall vorliegt):

| Parameter | Wert |
|-----------|------|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 64 |

**llama-cli:** `--temp 1.0 --top-p 0.95 --top-k 64` · **llama-server:** Standard-Sampling im Request

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
| Qwen 3.x NextN | `src/models/qwen3next.cpp`, `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp` |
| Multimodal + Spec | `tools/server/server-context.cpp`, `docs/speculative.md` |
| DiffusionGemma | `src/models/diffusion-gemma.cpp`, `src/llama-model.cpp`, `tools/diffusion-cli/` |
| Vulkan-WHT | `ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |
| GGUF-Konvertierung | `convert_hf_to_gguf.py` |

### DiffusionGemma — Status & Bekannte Probleme

| Feature | Status | Details |
|---------|--------|---------|
| Entropy-Bound Decoder | ✅ Funktioniert | `diffusion-cli.cpp` implementiert |
| Chat-Template Integration | ✅ Funktioniert | Automatisch via `common_chat_templates_apply()` |
| Default-Parameter (steps=48) | ✅ Funktioniert | `t_min=0.4`, `t_max=0.8`, `eb=0.1` |
| Self-Conditioning (SC) | ❌ Nicht verfuegbar | SC-Tensoren fehlen im GGUF |
| Dummy-Memory fuer `llama_decode` | ✅ Funktioniert | `llama_memory_diffusion` in `llama-model.cpp` |
| **KV-Cache Path (PREFILL→DECODE)** | ✅ **Gelöst** | Root Cause: `dg_ensure_pkv_store()` allokierte den gesamten PKV-Store auf `dev_layer(0)`. Bei partiellem GPU-Offload ist das CPU, aber `Kcur`/`Vcur` der GPU-Layer sind CUDA-resident → Cross-Backend `ggml_cpy` schlug fehl. Fix: PKV pro Layer auf dem Buffer-Type des jeweiligen Layer-Device (`m.dev_layer(il)`) allokieren → alle Operationen intra-Backend. Tests: `-ngl 8` (cut=14, "Paris") ✅, `-ngl 0` (cut=8) ✅. Siehe `docs/fork/archive/rca/2026-06-16_DEBUG_SESSION_PKV_FIX.md` fuer Root Cause und `docs/fork/2026-06-15_DIFFUSION_GEMMA_STATUS.md` fuer aktuellen Status. |

### Gemma 4 MTP — Status & Bekannte Probleme

| Feature | Status | Details |
|---------|--------|---------|
| **MTP 0% Akzeptanz (f16 + turbo)** | ✅ **Gelöst** | Root Cause: `n_embd_out_impl`/`n_embd_inp_impl` in `gemma4-assistant.cpp` wurden nicht auf `n_embd_backbone` gesetzt → `pending_h`-Puffer war 1024 statt 3840 Floats → Backbone-Hidden-State trunciert. Fix: `hparams.n_embd_inp_impl = hparams.n_embd_backbone; hparams.n_embd_out_impl = hparams.n_embd_backbone;` nach Laden von GGUF. Verifikation: f16 ~50-60% Akzeptanz, turbo4 ~75-100%. |
| **TurboQuant ISWA innerq_scale** | ✅ **Gelöst** | `llama_kv_cache_iswa_context` überschreibt `get_turbo_innerq_scale_inv()` nicht → inverse WHT in `build_attn_mha` erhielt `nullptr`. Fix: Override hinzugefügt, delegiert an `ctx_base`. Siehe Trilium-Note `K5rDVjhsJt6z` für Details. |
| **turbo3 V-Cache 0% auf GCN (Vega)** | ✅ **Gelöst** | Root Cause: turbo3 SET_ROWS Shader verwendete `subgroupBallot` für Signs-Packing, aber `ballot.x` hält nur 32 Bits. Auf AMD GCN (subgroup size 64, kein `VK_EXT_subgroup_size_control` für Compute) fehlen Bits 32-63 → 50% der Sign-Bits waren 0 → korrupte V-Cache-Daten. Fix: `subgroupBallot` durch `subgroupShuffle` ersetzt (subgroup-size-unabhängig). Verifikation: Venus (Vega) turbo3 MTP 0% → 59.7%, Mars (RDNA3) keine Regression. |

### Vulkan-Optimierungen — Status

| Feature | Status | Details |
|---------|--------|---------|
| **WHT fast path** | ✅ Cherry-picked | Upstream `48e7078ee` + `e82beaa60` (Intel fix). `fwht.comp` Shader fuer schnelle Hadamard-Transformation. Build kompiliert auf System A. Prompt-Verarbeitung +17% bei Gemma 4 12B. |
| **v_dot2_f32_f16** | ❌ Abgebrochen | Zu komplex — 6 Konflikte, FlashAttention-Refactor-Abhaengigkeiten. Siehe [FORKS.md §5.9](FORKS.md#59-vdot2-cherry-pick-abgebrochen). |
| **BFloat16 FA** | ❌ Nicht nutzbar | `VK_KHR_shader_bfloat16` nur fuer GFX12+ (Mesa 25.2.x). Unsere GPUs (RDNA3/Vega) zu alt. Siehe [FORKS.md §5.7](FORKS.md#57-warum-bfloat16-fa-fuer-uns-nicht-nutzbar-ist). |
| **Vulkan-Turbo3** | ✅ Funktioniert | Eigene Dequant-Shaders (`dequant_turbo3_0`, `mul_mat_vec_tq4_1s`). Langsame bei Kontext >4096 (Dequant-Overhead). Siehe `docs/fork/2026-06-15_STATUS_QUO_VULKAN.md`. |
| **coopmat2 Feature-Check** | ✅ Cherry-picked | `5a69c9743` — Prueft 7 coopmat2-Features vor Aktivierung. Verhindert Crashes bei unvollstaendiger Extension. decode_vector-Teil entfernt (nicht in Mesa 25.0.7). Siehe [FORKS.md §5.10](FORKS.md#510-coopmat2-feature-check-ergebnis). |
| **CUDA Fast WHT** | ✅ Cherry-picked | `a817a22bc` (Enum+CPU-WHT in `master`) + `c1f1e28d2` + `192d8ae8b` (CUDA-WHT in `feature/cuda-fast-wht`). `fwht.cu` auf direkte CUDA-Syntax umgeschrieben. Build kompiliert, +11% pp512. **Bereit fuer Merge** in `master`. Siehe [FORKS.md §5.11](FORKS.md#511-cuda-fast-wht-plan). |

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

### Systematische Tests ueber InferenzQuelle (benachbartes Repo)

Das Hauptprojekt **InferenzQuelle** (`~/projects/inferenzquelle/`) besitzt eine umfangreiche Test-Infrastruktur, die auch fuer diesen Fork genutzt werden kann. Vor manuellen Einzeltests dort nachsehen:

| Test-Typ | Befehl (ausgefuehrt in `~/projects/inferenzquelle/`) | Zweck |
|----------|------------------------------------------------------|-------|
| **Smoke-Test** | `uv run pytest tests/performance/mtp/test_mtp_baseline.py::TestMTPBaseline::test_baseline_smoke -v -s` | Laedt Modell, generiert 32 Tokens, prueft t/s > 5 |
| **MTP-Baseline** | `uv run pytest tests/performance/mtp/test_mtp_baseline.py -v -s` | Durchsatz ohne Draft |
| **MTP-mit-Draft** | `uv run pytest tests/performance/mtp/test_mtp_with_draft.py -v -s` | Spekulatives Decoding (MTP/NextN) |
| **Qualitaet-Short** | `uv run pytest tests/qualitaet/test_short_context.py -k 1.1_factual -v -s` | Einzelner Qualitaetstest |
| **Langkontext** | `tests/performance/langkontext/test_context_scaling.sh` | Kontext-Scaling-Benchmark |

**Hinweise:**
- Modell-Pfade sind host-adaptiv — zentrale Uebersicht in Trilium `yWi63z5N6vXc`
- GPU-Speicher vorher leeren: `pkill -f llama-cli`
- Ergebnisse werden automatisch in Trilium exportiert und als HTML-Report gespeichert (`ergebnisse/report_auto_*.html`)
- **Nie** manuelle `curl`-Tests gegen `llama-server` machen, wenn das pytest-Framework verfuegbar ist

**Manuelle Benchmarks:** Ergebnisse **NIE** in Modell-Info-Notes — Wegweiser `8cIsjKEhz7Fd` beachten.

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
| `docs/fork/` | Fork-spezifische Dokumentation: Status, Vergleiche, RCAs | [~] Kein eigenes DOX (Dokumentationssammlung) |

*Hinweis: Bei Aenderungen an Zweck, Grenzen oder Qualitaetsstandards eines Verzeichnisses: Child-AGENTS.md aktualisieren und diesen Index pruefen.*

---

## Benutzerpraeferenzen

- fukuro bevorzugt **deutsche Sprache** in Commits und Dokumentation.
- fukuro wuenscht sich **keine Host-Namen** in oeffentlichen Repos.
- fukuro bevorzugt **SSH statt HTTPS** fuer Git-Authentifizierung.
