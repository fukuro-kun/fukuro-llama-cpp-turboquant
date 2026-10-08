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

Die kanonische Dokumenten-Tabelle (welche Datei fuehrt welchen Stand) liegt in `docs/AGENTS.md`; der letzte Hierarchie-/Link-Audit ist `docs/DOX_AUDIT.md`.

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
- ❌ Lokale Dateipfade (`/path/to/user/...`)
- ❌ Persoenliche Identifikatoren
- ❌ SSH-Keys, Tokens, Passwoerter

**Erlaubt:**
- ✅ GPU-Architekturen (NVIDIA CUDA, AMD ROCm/Vulkan, Intel)
- ✅ Generische Beschreibungen

**Pruefung vor jedem Commit:**
```bash
grep -ri "Dev-Host\|BigGPU-Host\|RDNA3-Host\|GCN-Host\|Pascal-Host\|MoonHost1\|MoonHost2\|/path/to/user\|/path/to/media" .
```
Falls Treffer → Bereinigen!

### Python-Umgebungen: Ausschliesslich uv

**Verbindliche Regel:** Python-Pakete werden **nur** via `uv` verwaltet. Nie `pip install --break-system-packages`, nie systemweite Python-Installationen ohne Isolation. Details (Installation, Pfade, zentrale venv) in `LOKAL.md` → "uv und Python-Umgebungen".

### GPU- und Prozess-Regeln (kritisch)

- Auf dem Entwicklungs-Host wird die GPU von anderen Projekten mitbenutzt: dort **keine** Inferenz-Prozesse mit GPU-Offload (`-ngl >0`) starten. Builds und CPU-only-Tests (`-ngl 0`) sind dort erlaubt. GPU-Tests laufen auf den in `LOKAL.md` genannten Test-Hosts.
- **Prozessbeendigung (UNVERHANDELBAR):** Vor jedem Stoppen Belegung pruefen (`pgrep -f`, `nvidia-smi`/`rocm-smi`). Nur eigene, selbst gestartete PIDs per `TERM` beenden und deren Exit sowie die Speicherfreigabe abwarten (auf UMA/APU-Hosts: GTT-Freigabe pruefen, bevor ein neuer Server startet). **Kein** `killall`/`pkill` auf Modell- oder Servernamen und keine Server unter hartem `timeout` — fremde Prozesse anderer Projekte bleiben unangetastet.
- `/health` des Servers belegt nur Liveness, keine erfolgreiche Inferenz. Fehlende Log-Zeilen sind kein Deadlock-Beweis (stderr-Pufferung); Status via `/health` und Prozesszustand pruefen.

### Lokale Gegebenheiten

- Host-spezifische Pfade, GPU-Architekturen, Service-Namen und Build-Besonderheiten stehen in `LOKAL.md` (in `.gitignore`, nicht committet).
- Agenten muessen `LOKAL.md` lesen, bevor sie Host-spezifische Aktionen durchfuehren.
- **LAN-Deployment:** Welcher Host welchen Commit-Stand und Service faehrt, steht in `LOKAL.md` → "Fork-Deployment im LAN" und in Trilium-Note `eiba6WJDfTiq`. Vor Remote-Arbeiten auf anderen Hosts pruefen.
- **Remote-/Worktree-Regel (UNVERHANDELBAR):** Arbeit auf Remote-Hosts oder an Feature-Branches erfolgt auf dem **vorgesehenen gepinnten Commit bzw. Feature-Branch in einem separaten Worktree** — kein blinder `git pull` eines Produktiv-Checkouts auf `master`. Stale Builds sind eine bekannte Crash-Ursache; der verwendete Commit wird pro Aufgabe festgelegt. **Produktiv-Services werden nicht neu gestartet**, ausser der Auftrag ist explizit ein Deployment. Deployment-Tabelle in `LOKAL.md` nach Deployments aktualisieren.
- GPU-Architektur-Build-Matrix (generisch — Backend-Verfuegbarkeit ist von dtype-/head_dim-/Spec-Gates getrennt zu bewerten; keine universellen Feature-Garantien):
  - **Pascal (GTX 1070):** `-DGGML_CUDA=ON`; Kernel-Verfuegbarkeit und Durchsatz pro Op/dtype/head_dim pruefen, keine pauschale FA- oder FP16-Zusicherung.
  - **Ampere/Ada (RTX 3070/4060):** CUDA-Backend mit FA- und TurboQuant-Pfaden verfuegbar; konkrete Unterstuetzung weiterhin je Op/dtype/Spec-Gate pruefen.
  - **AMD iGPU/APU (RADV):** `-DGGML_VULKAN=ON`, ROCm experimentell. Dauerhafte Caveats: turbo3-FA ist deaktiviert (glslc haengt bei SPIR-V-Generierung; scalar-Fallback), turbo4-FA aktiv via `flash_attn_cm1.comp`. Auf UMA-Geraeten niemals zwei GPU-Server gleichzeitig (GTT == System-RAM, OOM legt den Host lahm) und `-fit off` wenn `--mmproj` genutzt wird (fit_params-Margin draengt sonst das Modell auf CPU). Datierte Benchmark-/RCA-Belege: `docs/fork/`.
  - **Intel iGPU (ANV/Mesa):** `-DGGML_VULKAN=ON`, keine coopmat2-Unterstuetzung. Skalarer FA-Pfad produziert bei >~2K Prompt-Tokens Gibberish; Production-Loesung ist Hybrid-Attention (Prefill unfused, Decode FA; Gate in `llama-context.cpp`, Override `GGML_VK_ENABLE_FA_INTEL=1`, `GGML_VK_DISABLE_DOT2=1` als Zusatz-Option). iGPU-Decode ist hardwarelimitiert (~1.4-5.75 t/s je nach KV-Laenge).

### Build-Verifikation

**CPU-only Builds und Tests sind eine legitime Verifikation** — nicht ueberall schneller oder universell verfuegbar, aber sie decken Logik ohne GPU-Abhaengigkeit ab. Wenn der Build fuer ein GPU-Backend bestimmt ist, muss nach `cmake -B build` das intendierte Backend verifiziert werden — ein Build ohne GPU-Backend faellt stillschweigend auf CPU und startet normal, nur langsam:

```bash
# 1. CMakeCache.txt pruefen (intendiertes GPU-Backend muss ON sein)
grep -E "GGML_VULKAN:|GGML_CUDA:" build/CMakeCache.txt

# 2. Binary gegen GPU-Library linken
ldd build/bin/llama-server | grep -E "vulkan|cuda"

# 3. Device-Erkennung pruefen
build/bin/llama-server --list-devices 2>&1 | head -5
```

**Bei fehlendem intendierten Backend:** Build abbrechen, `CMakeCache.txt` loeschen, mit korrektem Flag neu konfigurieren. Niemals einen CPU-only Build als GPU-Produktiv-Service starten (RCA 2026-08-12: ein Service lief 1,5 Tage unbemerkt auf CPU).

### Produktiv-Modelle (dauerhafter Vertrag, Betrieb in `LOKAL.md`)

- **Heavy:** `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (QAT) — Standard fuer grosse Systeme.
- **Light:** `granite-4.0-h-tiny-UD-Q4_K_XL.gguf` — Standard fuer schmale Systeme.
- Beispiel Gemma-4-Produktivmodell: `context_length = 262144` (256K) im GGUF. Das Maximum ist **modellspezifisch**: `context_length` pro GGUF pruefen, keinen universalen Maximalwert annehmen. Kontexte jenseits der im GGUF hinterlegten Trainings-Kontext-Metadaten erfordern eine explizite, separat verifizierte Erweiterung — kein stiller Override.
- Wirksame Kontext-Limits, Slots und Start-Skripte pro Host: `LOKAL.md` sowie `scripts/start-*-server.sh` / `scripts/run-*-server.sh`.
- Gemma-4-MTP mit Q4_0-Draft bremste in der Messung vom 2026-07-10 (`docs/fork/2026-07-10_QAT_MTP_Q4_0_BENCHMARK.md`) — seitdem abgeschaltet; eine Wiederaktivierung braucht eine neue kontrollierte Messung.

### Build-System

- **CMake** mit Backend-Optionen (`-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`, etc.)
- **Build-Verzeichnis:** `build/` (in `.gitignore`, nie committen)
- **glibc >= 2.43:** `scripts/build-cuda-glibc-patch.sh` verwenden (temporaerer Patch fuer `mathcalls.h`)
- **libssl-dev (HTTPS-Support):** `LLAMA_OPENSSL=ON` ist CMake-Default, aber CMake baut stillschweigend ohne SSL weiter wenn `libssl-dev` fehlt (`OPENSSL_CRYPTO_LIBRARY-NOTFOUND`). **Vor jedem Build pruefen:** `dpkg -l libssl-dev`. Falls fehlt: `sudo apt install libssl-dev`. **Nach Installation:** `rm build/CMakeCache.txt` (sonst bleibt `OPENSSL_CRYPTO_LIBRARY-NOTFOUND` gecacht). Verifikation: `strings build/bin/libllama-server-impl.so | grep -c SSL_new` muss >0 sein (SSL-Code liegt in der Shared Library, nicht im Binary).

### Gemma-4 Standard-Sampling-Parameter

Fuer alle Gemma-4 Modelle gilt `temperature = 1.0` — ausnahmslos; kein Spezialfall rechtfertigt eine Abweichung:

| Parameter | Wert |
|-----------|------|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 64 |

**llama-cli:** `--temp 1.0 --top-p 0.95 --top-k 64` · **llama-server:** Standard-Sampling im Request

### Gemma-4 Thinking-Deaktivierung (UNVERHANDELBAR)

**Gemma-4 26B-A4B-it** hat "ghost thought channels" (Google bestätigt fuer
12B/26B/31B). Ohne Thinking-Suppression emittiert das Modell `{thought}`-Tokens
in Endlosschleife und fuellt den gesamten `max_tokens`-Budget mit Muell.

**WICHTIG:** `--reasoning off` als Server-Default ist NICHT gesetzt — andere
Projekte (RPG, etc.) benötigen Thinking. Thinking-Suppression erfolgt
**ausschließlich per-Request**.

**Per-Request Suppression (PRIMÄR):** `thinking_budget_tokens: 1` im Request-Body
(server-common.cpp:1137). Zwingt den Sampler den Thought-Channel sofort
zu schließen. `chat_template_kwargs.enable_thinking: false` funktioniert
seit dem Template-Fix (2026-08-31) auch bei Tool-Call-Konversationen
zuverlässig — zuvor wurde der Suppression-Marker bei Tool-Call-Kontext
vom Template übersprungen. `thinking_budget_tokens: 1` bleibt der
zuverlässigste Mechanismus (template-unabhängig, Sampler-Ebene).

**Defensive Decke:** `num_predict=4096` begrenzt Blast-Radius sporadischer
Ghost-Thoughts. `max_tokens` ist nur eine Decke (Issue #5517), kein Trigger.

**Fork-Code-Positionen:**
- `common/arg.cpp:3264` — `--reasoning` CLI-Flag (on/off/auto, Default: auto)
- `common/chat.cpp:1238-1239` — `thinking_start_tag`, `thinking_end_tag` (PR #21697)
- `common/chat.cpp:1223-1232` — C++-Workaround: Tool-Call-Prompt + Suppression (Fix 2026-08-31)
- `tools/server/server-common.cpp:1088-1094` — `chat_template_kwargs.enable_thinking`
- `tools/server/server-common.cpp:1137` — `thinking_budget_tokens` (per-Request)
- `models/templates/google-gemma-4-31B-it.jinja:340-347` — Template-Suppression (Fix 2026-08-31)

**Quellen:** llama.cpp #21338, #20297, #21697, #5517 · Google Gemma Docs ·
Trilium `B3dfpx01pApY`

### Git-Workflow

- **Primary Remote:** Self-hosted Forgejo (`origin`); Codeberg ist Legacy-Mirror, GitHub ist Mirror-Remote. Tatsaechliche URLs: `git remote -v` bzw. `LOKAL.md`.
- **Branches:** `master` ist die gepflegte Basislinie; Entwicklung auf `feat/<thema>`-Branches, die auf `origin` verteilt werden. Vergleichs-/Benchmark-Arbeit laeuft auf gepinnten Commits in separaten Worktrees — Produktiv-Checkouts bleiben unangetastet.
- **Commit-Format:** `<type>: <Was> — <Warum>` — Types: `feat`, `docs`, `fix`, `security`, `refactor`, `bench`
- **Commits:** Deutsch, kurz und sachlich (keine generischen Marketing-Floskeln)
- **Keine automatischen Devin-Eintraege** in Commit-Nachrichten
- **CHANGELOG.md** (Root): Fuehrt alle bedeutsamen Aenderungen chronologisch. Nach jedem Feature/Bugfix eintragen. Solo-Session-Agenten tragen hier ihren Fortschritt ein.

### ROADMAP-Workflow (Fork-Beschleunigung)

**Zentrale Datei:** `docs/fork/ROADMAP.md` — alle Optimierungs-Ansaetze als Abhakliste mit Meilensteinen (M1-M6).

**Solo-Plaene:** `docs/fork/plans/SESSION_PLAN_<thema>.md` — pro Item >=1 Tag. Items <1 Tag werden zu einem kombinierten Plan zusammengefasst.

**Regeln:**
- Solo-Plaene werden nur fuer den **aktuellen und naechsten Meilenstein** erstellt. Tier 3-4 Plaene entstehen wenn der Meilenstein naeher rueckt (verhindert veraltete Plaene).
- Nach Abschluss eines Items: ROADMAP.md Status auf ✅ setzen, Solo-Plan Status auf ✅ setzen.
- Bei Blockaden: ROADMAP.md Status auf ❌ mit Begruendung, Solo-Plan dokumentiert die Blockade.
- Monatliche Recherche via `fork-speed-research` Skill (`.devin/skills/fork-speed-research/`) — neue Ansaetze in ROADMAP.md aufnehmen.
- Meta-Entscheidungen (welche Meilensteine, welche Items zuerst) werden hier in AGENTS.md festgehalten, nicht in der ROADMAP selbst.
- Aktive Umsetzungsplaene enthalten `[x]`-/`[ ]`-Checklisten fuer Diagnose, Implementierung, Tests und Freigabe. Nach jedem verifizierten Fortschritt den Status sofort mit Commit-/Testbeleg aktualisieren; Teilfortschritt, laufende Tests und Blockaden bleiben ungehaekt und werden benannt. Ein Phasen-Gate erst bei vollstaendig bestandenen Kriterien abhaken; bei Regression wieder oeffnen. ROADMAP und Ergebnisbericht bei Gate-Aenderungen synchron halten.

**Solo-Session-Ablauf fuer ROADMAP-Items:**
1. Agent liest `docs/fork/ROADMAP.md` → waehlt naechstes offenes Item im aktuellen Meilenstein
2. Agent liest `docs/fork/plans/SESSION_PLAN_<thema>.md` → fuehrt Solo-Session aus (nach `solo-session` Skill)
3. Agent aktualisiert ROADMAP.md (Status ☐→✅) und Solo-Plan (Status)
4. Agent committet + schreibt TTT-Eintrag
5. fukuro reviewt: ROADMAP zeigt Fortschritt als Abhakliste

### Fork-Features und Schluesseldateien

| Feature | Primaere Dateien |
|---------|-----------------|
| TurboQuant KV/Weights | `ggml/src/ggml-turbo-quant.c`, `src/llama-quant.cpp` |
| Spekulative Decodierung (draft-mtp u.a.) | `common/speculative.cpp`, `tools/server/server-context.cpp`, `docs/speculative.md` |
| Qwen4Exp (QSA, GDN, HC) | `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp` |
| DiffusionGemma | `src/models/diffusion-gemma.cpp`, `src/llama-model.cpp`, `tools/diffusion-cli/` |
| Vulkan-WHT | `ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |
| Expected Attention (EA Phase 3) | `src/llama-expected-attention.cpp`, `src/llama-expected-attention.h`, `src/llama-context.cpp` |
| Parallele Schema-Entscheidungen (Jev) | `tools/parallel-decision/`, `tools/server/server-context.cpp`, `common/arg.cpp` |
| GGUF-Konvertierung | `convert_hf_to_gguf.py` |

### Abgeschlossene Feature-Arbeiten (Kurzregister)

Detaillierte RCAs, Benchmarks und Commit-Belege stehen in `docs/fork/` (aktive Berichte) und `docs/fork/archive/` (historisch) sowie im root `CHANGELOG.md` und `docs/fork/CHANGELOG.md`:

- **DiffusionGemma:** Forward-Pass und Entropy-Bound Decoder funktional; Self-Conditioning blockiert (SC-Tensoren fehlen in den GGUFs). gguf-py-Registrierung inzwischen vorhanden (`MODEL_ARCH.DIFFUSION_GEMMA`).
- **Gemma 4 MTP (historischer Gemma-Pfad):** bekannte geloest Punkte — `n_embd_backbone`-Kontrakt (0-%-Akzeptanz-Bug), TurboQuant-ISWA-`innerq_scale`-Override, GCN-`subgroupBallot`-Sign-Bug, QAT-GGUF-Metadaten-Adapter, E4B+MTP `head_dim=512`-FA-Kernel-Routing. Belege: `docs/fork/2026-07-11_E4B_MTP_8GB_CRASH.md` u.a.
- **Vulkan:** WHT-fast-path, coopmat2-Feature-Check und CUDA-Fast-WHT-Codepfade sind im Baum vorhanden (Merge-Ancestry nicht verifiziert, kein Merge-Status-Claim); v_dot2_f32_f16 und BFloat16-FA nicht nutzbar/abgebrochen (siehe FORKS.md §5.4 Cherry-Pick-Tabelle). turbo3/turbo4-KV auf Vulkan belegt (`docs/fork/2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md`); APU-GPU-Hang-Fix `nodes_per_submit=10` fuer UMA-Geraete (`docs/fork/archive/rca/2026-06-21_VULKAN_GPU_HANG_ROOT_CAUSE.md`).
- **thecodacus MoE-Optimierungen:** `GGML_CUDA_REGISTER_HOST=1` (Pinning) und `GGML_SCHED_PREFETCH_EXPERTS=1` (async Expert-Prefetch) portiert; Wirksamkeit nur bei partiellem MoE-CPU-Offload. Beleg: `docs/fork/archive/sessions/2026-07-08_SOLO_SESSION_REPORT.md`.

---

## Arbeitsanleitung

### Schnelleinstieg fuer Aenderungen

1. **Code verstehen:** Zielverzeichnis scannen, zustaendiges Child-AGENTS.md lesen.
2. **Aendern:** Quelltext bearbeiten (C++, Python, CMake).
3. **Build testen:** `cmake --build build -j$(nproc)`
4. **Verifizieren:** Relevante Tests laufen lassen (`./build/bin/test-*`).
5. **DOX-Durchlauf:** Betroffene AGENTS.md aktualisieren.

**Maximal zwei parallele Agenten/Subagenten pro Session** (Rate-Limit- und Kontextschutz).

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
| `docs/` | Dokumentation (Upstream + Fork; Vertrag siehe `docs/AGENTS.md`) |
| `examples/` | Beispielprogramme (upstream, selten aendern) |
| `models/` | Modell-Konfigurationen und Tokenizer-Modelle |
| `gguf-py/` | Python-Bibliothek fuer GGUF-Dateien |
| `ci/` | CI/CD Konfiguration |

---

## Verifikation

- [ ] Vor Arbeit: DOX-Kette gelesen (Root + alle Children auf dem Pfad)
- [ ] Nach Arbeit: Betroffene AGENTS.md aktualisiert
- [ ] Keine privaten Daten in oeffentliche Repos committet
- [ ] Build erfolgreich (`build/` existiert und enthaelt Zielbinaries); bei intendiertem GPU-Backend Backend-Verifikation durchgefuehrt

### Systematische Tests ueber InferenzQuelle (benachbartes Repo)

Das Hauptprojekt **InferenzQuelle** (`~/projects/inferenzquelle/`) besitzt eine umfangreiche Test-Infrastruktur, die auch fuer diesen Fork genutzt werden kann. Vor manuellen Einzeltests dort nachsehen:

| Test-Typ | Befehl (ausgefuehrt in `~/projects/inferenzquelle/`) | Zweck |
|----------|------------------------------------------------------|-------|
| **MTP-Baseline** | `uv run pytest tests/performance/mtp/test_mtp_baseline.py -v -s` | Target-only Baseline (u.a. `TestMTPBaseline::test_baseline_smoke`: laedt Modell, generiert 32 Tokens, prueft t/s) |
| **MTP mit Draft** | `uv run pytest tests/performance/mtp/test_mtp_with_draft.py -v -s` | Spekulatives Decoding mit Draft (MTP/NextN) |
| **Qualitaet-Short** | `uv run pytest tests/qualitaet/test_short_context.py -k 1.1_factual -v -s` | Einzelner Qualitaetstest |
| **Langkontext** | `tests/performance/langkontext/test_context_scaling.sh` | Kontext-Scaling-Benchmark |

**Hinweise:**
- Modell-Pfade sind host-adaptiv — zentrale Uebersicht in Trilium `yWi63z5N6vXc`
- Vor Teststart GPU-Belegung pruefen; nur eigene Testprozesse per TERM beenden (siehe Prozess-Regeln oben)
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
| `docs/` | Dokumentation: kanonische Fork-Docs, Upstream-Doku, Audits | [x] Aktiv |

*Hinweis: Verzeichnisse ohne eigenen Index-Eintrag bleiben direkt vom Root-DOX abgedeckt — der Index listet registrierte Child-Vertraege, nicht jedes Verzeichnis. Bei Aenderungen an Zweck, Grenzen oder Qualitaetsstandards eines Verzeichnisses: Child-AGENTS.md aktualisieren und diesen Index pruefen.*

---

## Benutzerpraeferenzen

- fukuro bevorzugt **deutsche Sprache** in Commits und Dokumentation.
- fukuro wuenscht sich **keine Host-Namen** in oeffentlichen Repos.
- fukuro bevorzugt **SSH statt HTTPS** fuer Git-Authentifizierung.
- MoE-Optimierungen werden am korrekt gemessenen Prefill-/Decode-Ergebnis bewertet, nicht an einer bestimmten Arena-/Cache-Architektur. Fuer Flash-Next ist Strata die Same-Quant-/Same-Hardware-Vergleichslatte; groessere physische Batches und gezielte Scheduler-/Kernel-/Layout-Adaptionen sind erlaubt. Spezialisierten Feature-Branch je Mechanismus, echte Gegenprobe und weitere Iteration statt Abschluss nach einem kleinen Gewinn.
- MTP-Gewinne nur nach identischen Text-/Tokenbudget-/Kontext-/Sampling-Vergleichen mit Wiederholungen und Korrektheits-/Delivery-Pruefung behaupten. Akzeptanz, endliches Reasoning oder einzelne Timing-Werte sind keine Gesamtfreigabe.
