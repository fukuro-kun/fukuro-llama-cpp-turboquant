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

### GPU-Nutzungs-Regel (kritisch!) aktuell am 19.7.2026

**Auf Dev-Host (Dev-Host) wird die GPU von anderen Projekten genutzt (xtts-api, InferenzQuelle).**

- **VERBOTEN:** `llama-server`, `llama-bench`, `llama-cli` mit GPU-Offload (`-ngl >0`) auf Dev-Host zu starten.
- **VERBOTEN:** GPU-Prozesse auf Dev-Host abzuwürgen (`pkill`, `kill`), die nicht zum Fork gehören.
- **VERBOTEN:** Den GPU-Speicher auf Dev-Host zu füllen — andere Projekte brauchen ihn.
- **ERLAUBT:** Build auf Dev-Host (`cmake --build build`), CPU-only Tests (`-ngl 0`).
- **FUER GPU-TESTS:** Pascal-Host verwenden (`ssh Pascal-Host`, GTX 1070, Pascal). Siehe `LOKAL.md` → "GPU-Nutzungs-Regeln" und "Fork-Deployment im LAN" für Workflow und Pfade.

### Lokale Gegebenheiten

- Host-spezifische Pfade, GPU-Architekturen und Build-Besonderheiten stehen in `LOKAL.md` (in `.gitignore`, nicht committet).
- Agenten muessen `LOKAL.md` lesen, bevor sie Host-spezifische Aktionen durchfuehren.
- **LAN-Deployment:** Wo der Fork im LAN geklont ist (welcher Host, welcher Commit-Stand, welcher Service) steht in `LOKAL.md` → "Fork-Deployment im LAN" und in Trilium-Note `eiba6WJDfTiq` → Sektion "LAN-Deployment". Agenten muessen diese Info pruefen, bevor sie Remote-Arbeiten auf anderen Hosts durchfuehren.
- **Kernaufgabe: Remote-Hosts aktuell halten (UNVERHANDELBAR):** Wenn auf einem Remote-Host (Uranus, Styx, Mars, Venus) gearbeitet wird — sei es Debugging, Benchmarking oder Server-Start — muss der Host **vor der Arbeit** auf den aktuellen master-Stand gebracht werden: `git fetch origin && git pull origin master` + `cmake --build build -j$(nproc)`. Ein stale Build fuehrt zu Crashes in bereits gefixtem Code (RCA 2026-07-14: `GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)` Crash auf Uranus war ein 32h-staler Build, im aktuellen Code bereits behoben). **Nach der Arbeit:** Wenn der Host als Produktiv-Server dient, den Service mit dem frischen Build neu starten. Deployment-Tabelle in `LOKAL.md` nach Sync aktualisieren.
- GPU-Architektur-Build-Matrix (generisch):
  - **Pascal (GTX 1070):** `-DLLAMA_CUDA=ON`, FP16 nur via emulation, kein FlashAttention
  - **Ampere/Ada (RTX 3070/4060):** Volle Feature-Unterstuetzung, FlashAttention, TurboQuant
  - **AMD iGPU/APU:** `-DLLAMA_VULKAN=ON`, ROCm experimentell
    - ✅ **turbo3 KV-Cache funktioniert:** `--cache-type-k turbo3 --cache-type-v turbo3` (~5.1x Kompression)
    - ✅ **turbo4 KV-Cache funktioniert:** `--cache-type-k turbo4 --cache-type-v turbo4` (~3.8x Kompression)
    - ⚠️ **FlashAttention nur fuer turbo4 aktiv:** turbo3 FA ist DEAKTIVIERT (glslc hängt in infinite optimizer loop bei SPIR-V Generation, `vulkan-shaders-gen.cpp` Zeilen 692-704 auskommentiert). turbo3 fällt auf scalar Attention-Pfad zurück. turbo4 FA aktiv via `flash_attn_cm1.comp`.
    - ✅ **turbo3/turbo4 (mixed) ist die optimale Vulkan-Konfiguration** (Benchmark 2026-07-09, 26B-A4B, pp512-8192 + pp96k-128k): turbo3 K hat geringeren Dequant-Overhead (3.125 bit vs 4.25 bit), turbo4 V hat höhere Präzision (4.25 bit). Bei pp@96k-128k ist turbo3/4 **+31% schneller** als turbo4/4, bei tg gleichauf (±0.5%). Empfehlung: **K=turbo3, V=turbo4**. Siehe `docs/fork/2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md`.
    - ✅ **188k-"Klippe" geklärt (12.07.2026):** Die scharfe Performance-Klippe bei ~188k Kontext war **kein Vulkan-Backend-Bug**, sondern ein **OOM-Artefakt bei konkurrierenden GPU-Prozessen**. Zwei llama-server gleichzeitig → GTT-Overflow (2×16 GB > 26 GB GTT) → OOM-Kill → GPU-Buffer-Eviction → 0.10 t/s. Solo-Betrieb mit bis zu 256k Kontext funktioniert einwandfrei (28-32 t/s). **Regel:** Niemals zwei llama-server gleichzeitig auf derselben GPU. Die 180k-Grenze ist obsolet. Siehe `docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md` (RCA Update) und Trilium `SWumEN7WOXBI` Abschnitt 5.8.

### Produktiv-Standard (seit 2026-07-10): QAT + 196k Kontext (Styx seit 2026-07-19)

**Modell:** `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14.2G) — QAT (Quantization-Aware Training) von Unsloth. ersetzt `google_gemma-4-26B-A4B-it-IQ4_NL.gguf` (14.7G) als Standard.

**Vorteile QAT:**
- Mars (Vulkan): +10% pp, +16.6% tg vs IQ4_NL
- Styx (CUDA, MoE-Offload): tg gleichauf (CPU-limitiert)
- 0.5G kleiner → mehr VRAM/RAM für KV-Cache → **+44k Kontext (Mars), +64k Kontext (Styx)**

**Kontextfenster (Modell-Maximum: 256K / 262144 tokens):**
| System | Altes Limit (IQ4_NL) | Neues Limit (QAT) | Produktiv-ctx | Limit-Grund |
|--------|---------------------|-------------------|---------------|-------------|
| Mars (AMD APU, LXC phobos) | ~~180k~~ (obsolet) | **256k (lädt)** | **262144 (256k), 2×128k Slots** | RAM (30GB) — voll ausgenutzt, Modell-Maximum erreicht. Altes "256k OOM" war IQ4_NL + konkurrierende Server |
| Styx (GTX 1070) | 160k | 224k (lädt) | **196608 (196k, seit 2026-07-19)** | VRAM (8GB) + RAM (31GB) — 224k volles Ctx → RSS 31.4GB > 31GB RAM → Swap → MoE-Bottleneck → 503. 196k: RSS ~28.4GB, 2.6GB Reserve. Siehe `scripts/start-styx-26b-server.sh` |

Kontexte über 256k sind sinnlos — das Modell hat `context_length = 262144` (256K) als Maximum in der GGUF.

**MTP Q4_0 Draft: AUS** auf beiden Systemen. Q4_0 Draft (48-57% Acceptance) bremst: Mars -2.4%, Styx -14%. Siehe `docs/fork/2026-07-10_QAT_MTP_Q4_0_BENCHMARK.md`.

**Services:**
- Mars: `scripts/start-mars-26b-server.sh` — läuft **im LXC 240 (phobos)** als `~/.config/systemd/user/llama-server.service` (seit 12.07.2026, bare-metal Service gestoppt)
- Styx: `scripts/start-styx-26b-server.sh` als `llama-server-styx.service` (system, Port 18080, 196k Ctx seit 2026-07-19)
- Venus: `scripts/start-venus-26b-server.sh` als `llama-server-venus.service` (system, Port 18080, 256k Ctx, 2 Slots, Vulkan — seit 16.07.2026, Suspend 08-13h)
- Uranus: `scripts/start-uranus-26b-server.sh` / `scripts/stop-uranus-26b-server.sh` (user service, Port 18080, 1 Instanz, MoE-Offload 10 — seit 15.07.2026)

### Build-System

- **CMake** mit Backend-Optionen (`-DLLAMA_CUDA=ON`, `-DLLAMA_VULKAN=ON`, etc.)
- **Build-Verzeichnis:** `build/` (in `.gitignore`, nie committen)
- **glibc >= 2.43:** `scripts/build-cuda-glibc-patch.sh` verwenden (temporaerer Patch fuer `mathcalls.h`)
- **libssl-dev (HTTPS-Support):** `LLAMA_OPENSSL=ON` ist CMake-Default, aber CMake baut stillschweigend ohne SSL weiter wenn `libssl-dev` fehlt (`OPENSSL_CRYPTO_LIBRARY-NOTFOUND`). **Vor jedem Build prüfen:** `dpkg -l libssl-dev`. Falls fehlt: `sudo apt install libssl-dev`. **Nach Installation:** `rm build/CMakeCache.txt` (sonst bleibt `OPENSSL_CRYPTO_LIBRARY-NOTFOUND` gecacht). Verifikation: `strings build/bin/libllama-server-impl.so | grep -c SSL_new` muss >0 sein (SSL-Code liegt in der Shared Library, nicht im Binary).

### Gemma-4 Standard-Sampling-Parameter

Fuer alle Gemma-4 Modelle (sofern kein begruendeter Spezialfall vorliegt):

| Parameter | Wert |
|-----------|------|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 64 |

**llama-cli:** `--temp 1.0 --top-p 0.95 --top-k 64` · **llama-server:** Standard-Sampling im Request

### Gemma-4 Thinking-Deaktivierung (UNVERHANDELBAR)

**Gemma-4 26B-A4B-it** hat "ghost thought channels" (Google bestätigt fuer
12B/26B/31B). Ohne `--reasoning off` emittiert das Modell `{thought}`-Tokens
in Endlosschleife und fuellt den gesamten `max_tokens`-Budget mit Muell.

**Alle 26B-Server-Startskripte verwenden `--reasoning off`:**
`start-{styx,mars,venus,uranus}-26b-server.sh`.

- `--reasoning off` (CLI, PR #20297): Setzt `enable_thinking=false` zuverlaessig
- `chat_template_kwargs.enable_thinking: false` (Request-Body): Doppelsicherung,
  respektiert ab `server-common.cpp:1088-1094`
- `--reasoning-budget 0`: Zwingt den Sampler den End-Tag zu setzen (zusaetzlich)
- PR #21697 (Gemerged 2026-04-10): Gemma4 Budget-Sampler mit
  `thinking_start_tag = "<|channel>thought"`, `thinking_end_tag = "<channel|>"`
  in `common/chat.cpp:1238-1239`
- `max_tokens`/`num_predict` ist eine reine Decke (Issue #5517: "the model has
  no idea about it"), triggert KEINE laengeren Antworten. Defensive Decke
  `num_predict=4096` begrenzt sporadische Ghost-Thought-Blast-Radius.
- Trilium: `B3dfpx01pApY` (Thinking Referenz)

### Git-Workflow

- **Primary Remote:** `git@codeberg.org:fukuro/fukuro-llama-cpp-turboquant.git`
- **Branch:** `master` (Hauptentwicklung)
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
| Gemma 4 MTP | `src/models/gemma4-assistant.cpp`, `src/llama-context.cpp`, `common/speculative.cpp` |
| Qwen 3.x NextN | `src/models/qwen3next.cpp`, `common/speculative.cpp` |
| Qwen 3.5 Dense/MoE | `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp` |
| Multimodal + Spec | `tools/server/server-context.cpp`, `docs/speculative.md` |
| DiffusionGemma | `src/models/diffusion-gemma.cpp`, `src/llama-model.cpp`, `tools/diffusion-cli/` |
| Vulkan-WHT | `ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |
| Expected Attention (EA Phase 3) | `src/llama-expected-attention.cpp`, `src/llama-expected-attention.h`, `src/llama-context.cpp` |
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
| **turbo3 V-Cache 0% auf GCN (Vega)** | ✅ **Gelöst** | Root Cause: turbo3 SET_ROWS Shader verwendete `subgroupBallot` für Signs-Packing, aber `ballot.x` hält nur 32 Bits. Auf AMD GCN (subgroup size 64, kein `VK_EXT_subgroup_size_control` für Compute) fehlen Bits 32-63 → 50% der Sign-Bits waren 0 → korrupte V-Cache-Daten. Fix: `subgroupBallot` durch `subgroupShuffle` ersetzt (subgroup-size-unabhängig). Verifikation: AMD-GCN (Vega) turbo3 MTP 0% → 59.7%, AMD-RDNA3 keine Regression. |
| **Neue GGUF-Metadata-Keys (QAT + Q4_0 Draft)** | ✅ **Gelöst** | Upstream PR #23398 nutzt `gemma4-assistant` (Bindestrich) als arch-Namen und `embedding_length_out` statt `n_embd_backbone`. Adapter (Commit `f9a3dfc62`): (1) `LLM_KV arch_name-Override` nutzt originalen arch_name aus GGUF als Key-Prefix, (2) Arch-Aliase für `gemma4-assistant`/`gemma4_assistant`/`gemma4_mtp`, (3) `embedding_length_out` als Fallback für `n_embd_backbone`. QAT-Modelle und MTP Q4_0 Drafts laden jetzt korrekt. MTP Q4_0 Acceptance: ~57% auf Mars (QAT-Modell). |
| **E4B+MTP FA Crash (head_dim=512)** | ✅ **Gelöst** | Root Cause: E4B full-attention Layer haben `head_dim=512`, MMA-Kernel hat keine Template-Instanz für DKQ=512 (`fattn.cu:110` abort). TILE-Kernel hatte auch Lücke: kein Fallback für DV>256 ohne Mask, keine Config für 512/512 bei ncols=2 (MTP draft gqa_ratio=2). Fix: (1) Route head_dim=512 zu TILE-Kernel, (2) Fallback für DV>256 ohne GQA-opt mit gqa_ratio-basiertem ncols2, (3) Neue TILE-Config für 512/512 bei ncols=2. Verifikation: E4B+MTP auf 16GB (RTX 4060 Ti) mit turbo4/turbo3 KV → 103 t/s, f16 → 112 t/s. Siehe `docs/fork/2026-07-11_E4B_MTP_8GB_CRASH.md`. |

### Vulkan-Optimierungen — Status

| Feature | Status | Details |
|---------|--------|---------|
| **WHT fast path** | ✅ Cherry-picked | Upstream `48e7078ee` + `e82beaa60` (Intel fix). `fwht.comp` Shader fuer schnelle Hadamard-Transformation. Build kompiliert auf System A. Prompt-Verarbeitung +17% bei Gemma 4 12B. |
| **v_dot2_f32_f16** | ❌ Abgebrochen | Zu komplex — 6 Konflikte, FlashAttention-Refactor-Abhaengigkeiten. Siehe [FORKS.md §5.9](FORKS.md#59-vdot2-cherry-pick-abgebrochen). |
| **BFloat16 FA** | ❌ Nicht nutzbar | `VK_KHR_shader_bfloat16` nur fuer GFX12+ (Mesa 25.2.x). Unsere GPUs (RDNA3/Vega) zu alt. Siehe [FORKS.md §5.7](FORKS.md#57-warum-bfloat16-fa-fuer-uns-nicht-nutzbar-ist). |
| **Vulkan-Turbo3** | ✅ Funktioniert | Eigene Dequant-Shaders (`dequant_turbo3_0`, `mul_mat_vec_tq4_1s`). FA deaktiviert (glslc bug), scalar fallback. Trotzdem schneller als turbo4/4 (Benchmark 2026-07-09). Siehe `docs/fork/2026-06-15_STATUS_QUO_VULKAN.md` und `docs/fork/2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md`. |
| **coopmat2 Feature-Check** | ✅ Cherry-picked | `5a69c9743` — Prueft 7 coopmat2-Features vor Aktivierung. Verhindert Crashes bei unvollstaendiger Extension. decode_vector-Teil entfernt (nicht in Mesa 25.0.7). Siehe [FORKS.md §5.10](FORKS.md#510-coopmat2-feature-check-ergebnis). |
| **CUDA Fast WHT** | ✅ Cherry-picked | `a817a22bc` (Enum+CPU-WHT in `master`) + `c1f1e28d2` + `192d8ae8b` (CUDA-WHT in `feature/cuda-fast-wht`). `fwht.cu` auf direkte CUDA-Syntax umgeschrieben. Build kompiliert, +11% pp512. **Bereit fuer Merge** in `master`. Siehe [FORKS.md §5.11](FORKS.md#511-cuda-fast-wht-plan). |

### thecodacus MoE-Optimierungen — Status

| Feature | Status | Details |
|---------|--------|---------|
| **Memory Pinning** | ✅ In `feature/thecodacus-pinning` | `GGML_CUDA_REGISTER_HOST=1` — cudaHostRegister für mmap-pages. Verhindert OS-Paging der Expert-Gewichte im System-RAM. +19-23% pp bei MoE-Offloading. Portiert von `thecodacus/llama.cpp` Commit `20f5994`. |
| **Async Expert Prefetch** | ✅ In `feature/thecodacus-pinning` | `GGML_SCHED_PREFETCH_EXPERTS=1` (default 3 Slots) — Overlap Expert-Upload mit GPU-Compute durch zweite Backend-Instanz. +43-67% pp zusätzlich zum Pinning. Portiert von Commits `1163cb3`+`5f83fbb`. |
| **Kombiniert** | ✅ Getestet | Pinning+Prefetch zusammen: **+72-106% pp, +30% tg** auf GTX 1070 mit 26B-A4B MoE-Offloading. Siehe `docs/fork/archive/sessions/2026-07-08_SOLO_SESSION_REPORT.md`. |

**Wichtig:** Diese Optimierungen wirken NUR bei partiellem GPU-Offload von MoE-Modellen (Experten auf CPU). Bei vollständigem GPU-Offload (z.B. E4B) gibt es keine H2D-Copies → kein Effekt.

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
