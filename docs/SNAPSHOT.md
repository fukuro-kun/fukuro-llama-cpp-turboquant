# Momentaufnahme — fukuro-llama-cpp-turboquant + InferenzQuelle

**Datum:** 2026-07-13 | **Branch:** `master` | **Letzter Commit:** `21fdf8c88` (ROADMAP updates)

---

## Die Projekte und ihre Nahtstelle

| Projekt | Rolle | Repo |
|---------|-------|------|
| **fukuro-llama-cpp-turboquant** | Inference Engine (Motor) — C++/CUDA/Vulkan | Codeberg |
| **InferenzQuelle** | Infrastruktur & Steuerung (Auto) — Python/Bash | Codeberg |

**Nahtstelle:** InferenzQuelle steuert den `llama-server` (aus dem Fork) per HTTP-API. Test-Framework (pytest) läuft gegen die Binary. Modell-Pfade, Draft-Modelle und TurboQuant-Parameter werden über Config-Files geteilt. Produktiv-Endpunkte: phobos:18080, styx:18080, uranus:8080 — gebündelt über janus:8010 (InferenzQuelle Router).

---

## Was zuletzt passiert ist

### 2026-07-13: thecodacus Expert Prefetch 2-Slot + UMA Cached + Research-Sweep #3 (Solo-Session)

- **thecodus Prefetch 2-Slot Sweet-Spot** — `GGML_SCHED_PREFETCH_SLOTS` env var hinzugefügt. Styx: +28.9% pp512, +36.2% pp2048, +36.7% pp8192, +2.1% tg8. Mars: +8.8% pp512, +3.8% tg128. Beide Server-Scripts auf 2-Slot umgestellt. 3-Slot war -35% tg auf Mars.
- **Vulkan UMA Cached Host Memory (#56)** — PR #23762 portiert. HostCached statt write-combining auf UMA. memory_property_flags Bug fix (actual vs requested). Mars: tg128 +7% auf tg, +11% vs baseline.
- **Research-Sweep #3** — 4 parallele Subagents, 21 neue ROADMAP-Items (#56-#76). 9 PRs als bereits im Fork identifiziert. M3 abgeschlossen, M6 erweitert.
- **M3 Meilenstein abgeschlossen** — #37✅ (Prefetch), #15❌ (PipeShard), #56✅ (UMA Cached), #60❌ (RADV), #68✅ (already in fork).

### 2026-07-13: thecodacus Expert Prefetch repariert + 2-Slot Sweet-Spot (Solo-Session)

- **thecodacus Prefetch repariert** — Root Cause: `nodes[0]` war SCALE nicht MUL_MAT_ID. Fix: Graph-Suche im gesamten Split. `GGML_SCHED_PREFETCH_SLOTS` env var hinzugefügt.
- **2-Slot Sweet-Spot** — Styx: +28.9% pp512, +36.2% pp2048, +36.7% pp8192, +2.1% tg8. Mars: +8.8% pp512, +3.8% tg128. Beide Server-Scripts auf 2-Slot umgestellt. 3-Slot: -35% tg auf Mars (Queue-Konflikt RDNA3).
- **M3 Meilenstein abgeschlossen** — #37✅ (Prefetch), #15❌ (PipeShard: zu großer Aufwand, marginaler Benefit), #40⏭️ (Phase 2 negativ).
- **Expert-Cache (#37) postponed** — Buffer-Recycling-Crash, benötigt Buffer-Pinning oder Per-Expert Tensor-Splitting.

### 2026-07-13: E4B-Deployment Uranus + Doku-Pflege + 256k-Migration

- **E4B QAT Server auf Uranus gestartet** — `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (4.2 GB) + MTP-Draft (Q4_0, n_max=2), 256k/2×128k, Port 8080, turbo4/turbo3 KV, FA on. tg 107.6 t/s (kurz), 47-63 t/s (lang). Manuell (nohup, kein systemd). VRAM knapp neben xtts-api (GPU 1 nur 1.0 GB frei).
- **256k LXC-Migration committed** (`e09da1df4`) — Mars bare-metal → LXC 240 (phobos), 256k Kontext (Modell-Maximum), 2×128k Slots. AGENTS.md, CHANGELOG, 188k-RCA-Doc, Mars-Startskript.
- **Fork-Hauptnote §8a aktualisiert** — Uranus-Zeile + Produktiv-Server-Abschnitt (E4B) in Trilium `eiba6WJDfTiq` eingetragen.
- **188k-Performance-Klippe RCA geklärt** — Kein Vulkan-Bug, sondern OOM-Artefakt durch konkurrierende llama-server (2×16 GB GTT > 26 GB Limit). Solo-Betrieb 224k/256k: 28-32 t/s. 180k-Grenze obsolet.

### 2026-07-13: E4B+MTP FA Crash Fix (Solo-Session)

- **head_dim=512 FA Crash gelöst** — E4B full-attention Layer haben `head_dim=512`, MMA-Kernel ohne Template für DKQ=512. Drei Commits: Route zu TILE-Kernel, Fallback für DV>256, neue TILE-Config für 512/512 bei ncols=2. Verifikation Uranus: 103 t/s (turbo4/turbo3), 112 t/s (f16). Commits `f9e7564bd`, `bd8ef5978`, `1a0af56dc`.
- **#35 Row-Packing DMMV RDNA3** — +1% (erwartet +10-20%), DMMV nicht der Bottleneck für kleine MoE-Modelle. Change beibehalten. Commit `acd8dfe3a`.
- **#45 CUDA Concurrent Streams QKV** — Bereits im Fork, Regression mit CUDA Graphs (-10.7% auf Uranus, kein Effekt auf Styx wegen MoE-Offload). ❌.
- **Phase 3 Batch-Eval Tier 2 (8 Items)** — #16✅ bereits integriert, #38⏭️ verschoben, 6 Items ☐ offen.

### 2026-07-11: Pascal MMVQ + M1/M2 Batch-Eval

- **#3 Pascal CUDA MMVQ** — manuell portiert, +8.9% Generation auf GTX 1070 (E2B MoE). Commit `5c1884929`.
- **M1/M2 Batch-Eval** — 11 Items evaluiert: #2✅, #4✅, #5✅, #6✅ bereits integriert, #7✅ bereits integriert, #11✅, #12✅, #20✅, #28✅ eigene Implementierung, #1❌, #9❌, #10❌, #13❌.

### InferenzQuelle (2026-07-13)

- **Branch `uranus-local`** — Letzte Commits: 4-fach Audit Fixes, Judge-Pipeline (Subagent LLM-Judge), Lang-Context-Tests, Test-Cleanup (9 tote Dateien), Kahlschlag (142 tote Dateien).
- **Uncommittet:** AGENTS.md, docs/AGENTS.md, docs/ARCHITECTURE.md + neue config/hosts.json, docs/CACHE_PERFORMANCE.md, router/, shared/.

---

## Wo wir stehen — das große Bild

### Fork — Meilensteine

| Meilenstein | Status | Notiz |
|---|---|---|
| **M1** Quick-Win-Welle | ✅ | #2✅, #4✅, #5✅, #1❌ |
| **M2** Vulkan-Offensive | ✅ | #6✅, #7✅, #12✅ bereits integriert, #9❌, #10❌ |
| **M3** MoE-Offloading v2 | ⏳ teils | #3✅, #13❌, #14⏭️, #15☐ |
| **M4** Speculative Decoding v2 | ✅ | #11✅, #28✅ eigene Implementierung |
| **M5** Coopmat2 + Multi-GPU | ⏳ teils | #12✅, #20✅, #21 offen |
| **M6** Forschung | ☐ offen | #17, #18, #19, #25-27, #29, #30 |

### Fork — Produktiv-Server

| Server | Modell | Kontext | Backend | Status |
|---|---|---|---|---|
| **Mars/phobos** (LXC 240) | 26B-A4B QAT | 256k/2×128k | Vulkan | ✅ Produktiv, systemd |
| **Styx** | 26B-A4B QAT | 224k/1 Slot | CUDA (MoE-Offload) | ✅ Produktiv, systemd |
| **Uranus** (E4B) | E4B QAT + MTP | 256k/2×128k | CUDA | ✅ Produktiv, manual (nohup) |
| **Hydra** | — | — | CUDA | Dev-only (GPU shared) |
| **Venus** | — | — | Vulkan | ⏸️ Suspend-Policy, Service deaktiviert |

### Fork — Komponenten

| Komponente | Status |
|---|---|
| TurboQuant KV/Weights | ✅ turbo3/turbo4 auf CUDA + Vulkan |
| Gemma 4 MTP | ✅ 0%-Bug gefixt, 50-100% Akzeptanz |
| Qwen 3.x NextN | ✅ Shared-Model-Draft |
| DiffusionGemma | ✅ PKV-Cross-Backend-Fix |
| Vulkan-WHT / CUDA Fast WHT | ✅ |
| thecodacus Pinning+Prefetch | ✅ +95% pp, +50% tg mit MTP |
| E4B+MTP FA (head_dim=512) | ✅ TILE-Kernel-Fix (2026-07-13) |

---

## Aktuell in Arbeit (uncommitted)

**Fork:** Sauber — keine uncommitteten Änderungen (gerade committed `e09da1df4`).
**InferenzQuelle:** 3 modifizierte + 4 neue unversionierte Dateien (AGENTS.md, docs/ARCHITECTURE.md, config/hosts.json, docs/CACHE_PERFORMANCE.md, router/, shared/) — warten auf Review.

---

## Offene Aufgaben

### [F] Fukuro — Hochpriorisiert
1. **ROADMAP Tier 2 Priorisierung** — Welche der 6 offenen Items (#8 Mixed Precision KV, #15 PipeShard, #34 UBBoost, #36 Auto-TP, #37 LFRU, #40 MoE Load Balancing) als nächstes?
2. **E4B systemd-Service auf Uranus** — Persistent einrichten (wie Mars/Styx) oder ad-hoc lassen?
3. **Trilium `Czii4MdFKb3i`** — Leere Momentaufnahme-Note löschen oder befüllen?
4. **InferenzQuelle uncommittete Änderungen** — router/, shared/, config/hosts.json, CACHE_PERFORMANCE.md reviewen und committen.

### [D] Devin — Mittelpriorisiert
5. **ROADMAP Tier 2 Items** — Nach Priorisierung implementieren (je 2-6 Wochen).
6. **#21 Multi-GPU** (M5) — Noch offen, Tier 3.
7. **M6 Forschungs-Items** — Recherche + Evaluation.

---

## Nächste Schritte

1. **ROADMAP Tier 2 Priorisierungs-Entscheidung** von fukuro einholen → dann Solo-Session für gewähltes Item.
2. **InferenzQuelle Änderungen reviewen** — router/ und shared/ deuten auf neue Architektur, braucht Klärung.
3. **Trilium-Aufräumen** — Leere Note `Czii4MdFKb3i`, ggf. Momentaufnahme-Verlauf konsolidieren.
