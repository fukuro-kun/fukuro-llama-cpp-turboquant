# Sync-Merge Masterplan: AtomicBot `sync/upstream-2026-06-19` → master

**Datum:** 2026-06-21
**Status:** Geplant (finalisiert)
**Autor:** fukuro + Devin

## 1. Ausgangslage

### 1.1 Was ist passiert?

Upstream (ggml-org/llama.cpp) hat einen grossen Refactor durchgeführt:
- **Klassenhierarchie-Refactor:** `llama_model` ist jetzt abstrakt mit `virtual load_hparams()`, `load_tensors()`, `load_arch_hparams()`, `load_arch_tensors()`, `build_arch_graph()`. Jede Architektur hat ihre eigene Subklasse (`llama_model_qwen35`, `llama_model_gemma4_assistant`, etc.).
- **Native NextN/MTP:** Qwen3.5/3.6 NextN und Gemma4 MTP sind jetzt nativ in upstream integriert — nicht mehr als separate Arch-Typen.
- **Neue interne Contracts:** MTP/NextN/delta-net wurden neu implementiert, inkompatibel mit Atomic's bisherigen Implementationen.

AtomicBot hat diesen Sync am 2026-06-19 durchgeführt (`0117ea367`) und dabei:
- Upstream's refactored Subsysteme übernommen (ggml, llama-context/graph/cparams/arch, spec-decode, model class system, qwen35/gemma4 model builders, gguf-py, server tools)
- Atomic-only Artefakte bewahrt (multimodal projectors, release CI, macOS entitlements, README branding)
- TurboQuant Core beibehalten (28 Referenzen in ggml-common.h, ggml-turbo-quant.c, turbo types in ggml.h)
- Orphaned NextN-Dateien gedroppt (`qwen35-nextn.cpp`, `qwen35moe-nextn.cpp`)

### 1.2 Unser Stand

Wir haben AtomicBot's Stand vom 2026-06-13 (`16fe4bc2e`) + eigene Commits:
- Vulkan GPU-Hang Fix (`nodes_per_submit=10` für UMA)
- **TurboQuant Vulkan FA-Pipelines** (turbo3/turbo4 CREATE_FA, mul_mat_vec, dequant — AtomicBot hat diese NICHT!)
- DiffusionGemma Integration (monolithischer Port von PR #24423)
- Diverse Vulkan Cherry-Picks + Reverts
- Dokumentation (docs/fork/, FORKS.md, README.md, AGENTS.md)

### 1.3 Was fehlt uns

9 AtomicBot-Commits seit unserem letzten Merge:

| Commit | Beschreibung | Ausmass |
|--------|-------------|---------|
| `35ac80d55` | Merge PR #182 (TheTom catchup) | 3021 Dateien, 1.2M Zeilen |
| `b0e900a28` | GitHub Actions Vulkan Release CI | 2696 Dateien |
| `8f3fbc0f0` | fix: leading whitespace vor `imd` tags | 2 Dateien |
| `2f8661751` | vulkan: fuse TurboQuant K/V dequant into FA | 3 Dateien |
| `58d401ef9` | Merge PR #183 (think-whitespace) | 2 Dateien |
| `4595fff0b` | Merge PR #184 (vulkan-turbo-fa) | 3 Dateien |
| `0117ea367` | **merge: sync upstream** | 1710 Dateien, +211k/-114k |
| `148200c45` | fix: adopt native Qwen NextN, drop orphaned | 5 Dateien, -338 Zeilen |
| `dd27e6d34` | build: Windows x64 release CI | 6 Dateien |

## 2. Rollback-Sicherheit

### 2.1 Angewandte Massnahmen

| Ebene | Was | Status |
|-------|-----|--------|
| Git-Tag | `pre-sync-2026-06-21` → `fe7586c71` | Erstellt |
| Backup-Branch | `backup/pre-sync-2026-06-21` → `fe7586c71` | Erstellt |
| Codeberg (origin) | `origin/master` → `fe7586c71` | Gesichert |
| GitHub-Mirror | `github/master` → `39282c6a4` | Gesichert (1 Commit hinter) |

### 2.2 Rollback-Prozedur

Bei jeglichem Fehler oder wenn das Ergebnis nicht akzeptabel ist:

```bash
cd <repo-root>
git checkout master
git reset --hard pre-sync-2026-06-21
git push origin master --force
```

Das stellt den exakten Stand vom 2026-06-21 wieder her. Der Tag ist unverlierbar.

## 3. Fork-spezifischer Mehrwert — Detaillierte Analyse

### 3.1 Was AtomicBot bereits hat (keine Re-Application nötig)

| Feature | In AtomicBot? | Aktion |
|---------|--------------|--------|
| TurboQuant Core (`ggml-turbo-quant.c`, types in `ggml.h`, `ggml-common.h`) | Ja | Upstream übernehmen |
| TurboQuant Vulkan set_rows (turbo3/turbo4) | Ja | Upstream übernehmen |
| TurboQuant Vulkan dispatcher cases | Ja | Upstream übernehmen |
| `dequant_turbo3_0.comp` | Ja (identisch) | Upstream übernehmen |
| `dequant_tq4_1s.comp`, `mul_mat_vec_tq4_1s.comp` | Ja (identisch) | Upstream übernehmen |
| TheTom's `flash_attn_dequant.glsl` (fused dequant) | Ja (14959 bytes) | Upstream übernehmen |
| Gemma4 MTP (`gemma4-assistant.cpp`, native NextN) | Ja (upstream native) | Upstream übernehmen |
| Qwen3.5/3.6 NextN (native) | Ja | Upstream übernehmen, unsere orphaned Dateien droppen |
| CUDA FWHT | Ja | Upstream übernehmen |
| PEG Parser whitespace fix | Ja | Upstream übernehmen |
| PR #22455 (transfer queue UMA) | Ja (Zeile 3213, 6287) | Upstream übernehmen — ggf. revertieren bei Perf-Regression |

### 3.2 Was wir re-applien müssen — P0 Kritisch

| Feature | Dateien | Warum kritisch | Schwierigkeit |
|---------|---------|----------------|---------------|
| **Vulkan turbo3/turbo4 FA-Pipelines** | `ggml-vulkan.cpp` (8 CREATE_FA calls) | AtomicBot hat nur "deferred" Platzhalter — keine funktionierenden Pipelines! Ohne das: turbo3/turbo4 FA auf Vulkan unbrauchbar. | Hoch — manuell in AtomicBot's umstrukturierte ggml-vulkan.cpp einpflegen |
| **`dequant_turbo4_0.comp`** | Vulkan Shader (1133 bytes) | AtomicBot hat leere Datei (0 bytes)! turbo4 dequant auf Vulkan broken ohne. | Trivial — Datei kopieren |
| **`turbo_wht.comp` group_size Fix** | Vulkan Shader | AtomicBot hat hardcoded 128, wir haben group_size-Handling (64+128). Broken für group_size=64 ohne Fix. | Trivial — Datei kopieren |
| **Vulkan nps Fix** | `ggml-vulkan.cpp` (1 Zeile) | `nodes_per_submit = ctx->device->uma ? 10 : 100`. GPU-Hang bei >188k Kontext auf AMD APU ohne. | Trivial — 1 Zeile |

### 3.3 Was wir re-applien müssen — P1 Hoch

| Feature | Dateien | Warum wichtig | Schwierigkeit |
|---------|---------|----------------|---------------|
| **DiffusionGemma** | `src/models/diffusion-gemma.cpp`, `tools/diffusion-*/`, `conversion/diffusion_gemma.py`, CUDA `diffusion-sampling.cu/.cuh` | Vollständig fork-only. AtomicBot hat nur `examples/diffusion/` Basis. | Hoch — muss auf neue Klassenhierarchie portiert werden |
| **CUDA glibc 2.43 Patch** | `scripts/build-cuda-glibc-patch.sh` | CUDA-Build auf modernen Systemen broken ohne. | Trivial — Skript kopieren |

### 3.4 Was wir re-applien müssen — P2 Mittel

| Feature | Dateien | Warum wichtig | Schwierigkeit |
|---------|---------|----------------|---------------|
| **Gemma4 MTP Vulkan Fix** | `src/llama-context.cpp` (Commit `656456804`) | Multi-Slot-Crashes bei `parallel=2` auf Vulkan. | Mittel — in neue llama-context.cpp einpflegen |
| **CUDA KV-Cache Reserve** | `src/llama-kv-cache.cpp` oder ähnlich (Commit `c1b8a86dc`) | OOM bei großen Kontexten. | Mittel |
| **Tests** | `tests/test-turbo-quant.c`, `tests/test-speculative-mtp.cpp` | Fork-spezifische Tests. | Trivial — Dateien kopieren + CMakeLists |

### 3.5 Was wir droppen

| Was | Warum |
|-----|-------|
| `src/models/qwen35-nextn.cpp` | Orphaned, upstream hat native NextN in `qwen35.cpp` |
| `src/models/qwen35moe-nextn.cpp` | Orphaned, upstream hat native NextN in `qwen35moe.cpp` |
| `hparams.nextn_predict_layers` | ersetzt durch `hparams.n_layer_nextn` |
| Unsere 6 Vulkan Reverts (PRs #22455, #22930, #23770, #24326, #23665, #23667) | Nur #22455 ist in AtomicBot. Nach Merge: testen ob 5% pp512 Regression auftritt → ggf. #22455 re-revertieren |
| Unsere alten TurboQuant Vulkan Commits (23 Commits) | Werden durch gezielte Re-Application der P0-Features ersetzt |

### 3.6 Vulkan TurboQuant: Unser Ansatz vs. AtomicBot

| Aspekt | Unser Fork | AtomicBot |
|--------|-----------|-----------|
| turbo3 FA Pipelines | ✅ 4 CREATE_FA (SCALAR, SCALAR_fp32, COOPMAT1, COOPMAT2) | ❌ "turbo3 FA SPIR-V generation deferred; no dedicated pipeline yet" |
| turbo4 FA Pipelines | ✅ 4 CREATE_FA (gleiche 4 Pfade) | ❌ Gar nicht vorhanden |
| turbo3/turbo4 mul_mat_vec | ✅ Implementiert | ❌ Nicht vorhanden |
| turbo4 dequant Shader | ✅ 1133 Bytes (vollständig) | ❌ 0 Bytes (leere Datei!) |
| turbo3 dequant Shader | ✅ Identisch mit AtomicBot | ✅ Identisch |
| turbo_wht.comp | ✅ Mit group_size-Handling (64+128) | ⚠️ Hardcoded 128, kein group_size |
| flash_attn_dequant.glsl | ❌ Nicht vorhanden | ✅ 14959 Bytes (TheTom's fused approach) |
| set_rows turbo3/turbo4 | ✅ Vorhanden | ✅ Vorhanden |
| Dispatcher cases | ✅ Vorhanden | ✅ Vorhanden |

**Strategie:** Wir übernehmen AtomicBot's `flash_attn_dequant.glsl` (TheTom's fused approach) UND re-applien unsere CREATE_FA Pipelines + mul_mat_vec + dequant_turbo4_0.comp. Beide Ansätze sind komplementär.

## 4. Konflikt-Analyse

### 4.1 Quantifizierung

| Kategorie | Dateien | Risiko |
|-----------|---------|--------|
| Total geänderte Dateien (beide Seiten) | ~295 | — |
| Davon kritische Core-Dateien | 34 | Hoch |
| Davon TurboQuant Vulkan-spezifisch | ~5 | Hoch (unsere P0-Features) |
| Davon non-kritisch (auto-merge) | ~256 | Niedrig |

### 4.2 Kritische Konflikt-Dateien (Top 10)

| Datei | Konflikt-Grund | Strategie |
|-------|---------------|-----------|
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Massiv umstrukturiert in AtomicBot. Unsere 8 CREATE_FA + nps Fix + mul_mat_vec müssen in neue Struktur eingepflegt werden. | **Manuell mergen** — AtomicBot als Basis, unsere P0-Features gezielt re-applien |
| `ggml/src/ggml-vulkan/vulkan-shaders/dequant_turbo4_0.comp` | AtomicBot: leer (0 bytes). Wir: 1133 bytes. | **Unsere Version** — Datei kopieren |
| `ggml/src/ggml-vulkan/vulkan-shaders/turbo_wht.comp` | AtomicBot: hardcoded 128. Wir: group_size-Handling. | **Unsere Version** — Datei kopieren |
| `src/llama-model.h/.cpp` | Monolithisch vs. Klassenhierarchie | AtomicBot Version (Klassenhierarchie) |
| `src/llama-context.cpp` | MTP-Integration vs. upstream's native MTP | AtomicBot Version + MTP Vulkan Fix re-applien |
| `src/llama-hparams.h/.cpp` | `nextn_predict_layers` vs. `n_layer_nextn` | AtomicBot Version (n_layer_nextn) |
| `common/speculative.cpp/.h` | Unsere NextN-Logik vs. upstream | AtomicBot Version |
| `src/models/qwen35.cpp` | Unsere `nextn_predict_layers` vs. native NextN | AtomicBot Version |
| `tools/mtmd/models/models.h` | Unsere gemma4-arch declarations vs. upstream | AtomicBot Version + DiffusionGemma re-add |
| `tools/server/server.cpp` | Unsere MTP-Endpunkte vs. upstream | AtomicBot Version |

## 5. Ausführungsplan (12-20 Stunden, KI-Solo)

### Phase 0: Vorbereitung (30 min)

- [x] Git-Tag `pre-sync-2026-06-21` erstellt
- [x] Backup-Branch `backup/pre-sync-2026-06-21` erstellt
- [ ] Feature-Branch erstellen: `git checkout -b feature/sync-atomicbot-2026-06-19`
- [ ] Merge starten: `git merge atomictemp/sync/upstream-2026-06-19 --no-commit`

### Phase 1: Merge durchführen — Auto-merge + Konflikt-Katalog (1-2h)

Merge starten und katalogisieren was automatisch gemerged wurde und was Konflikte hat.

**Subagent-Einsatz:** 1x `subagent_explore` (background):
- Katalogisiere alle Konflikt-Dateien
- Gruppiere nach: (a) Core-Subsysteme → `--theirs`, (b) Fork-only → `--ours`, (c) Manuell → Liste

### Phase 2: Core-Subsysteme übernehmen (2-3h)

**Strategie:** Für alle refactored Subsysteme AtomicBot's Version übernehmen (`git checkout --theirs`).

Betroffene Dateien:
- `src/llama-model.h`, `src/llama-model.cpp` (Klassenhierarchie)
- `src/llama-context.cpp`, `src/llama-context.h`
- `src/llama-graph.cpp`, `src/llama-graph.h`
- `src/llama-hparams.h`, `src/llama-hparams.cpp`
- `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`
- `src/llama-kv-cache-iswa.cpp`, `src/llama-kv-cache-iswa.h`
- `src/llama-memory-hybrid.cpp`, `src/llama-memory-hybrid.h`
- `src/llama-cparams.h`
- `common/speculative.cpp`, `common/speculative.h`
- `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`
- `src/models/gemma4-assistant.cpp`
- `tools/mtmd/models/models.h`
- `tools/server/server.cpp` und verwandte

**Subagent-Einsatz:** 2x `subagent_explore` (parallel, background):
1. SA-A: Vergleiche jede unserer Core-Dateien mit AtomicBot's Version — was haben wir fork-spezifisch geändert das upstream nicht hat?
2. SA-B: Prüfe ob unsere MTP-Endpunkte in `server.cpp` noch kompatibel mit upstream's neuer MTP-Logik sind

### Phase 3: TurboQuant Vulkan re-applien — P0 Kritisch (3-4h)

Das ist der heikelste Teil. AtomicBot's `ggml-vulkan.cpp` ist massiv umstrukturiert.

**Schritte:**
1. AtomicBot's `ggml-vulkan.cpp` als Basis übernehmen
2. Unsere 8 CREATE_FA calls für turbo3/turbo4 finden und re-applien
3. Unsere turbo3/turbo4 mul_mat_vec Pipelines re-applien
4. Unsere turbo4 dequant Pipeline re-applien
5. `dequant_turbo4_0.comp` (1133 bytes) kopieren — AtomicBot hat leere Datei
6. `turbo_wht.comp` mit group_size-Handling kopieren
7. Vulkan nps Fix re-applien: `nodes_per_submit = ctx->device->uma ? 10 : 100`
8. PR #22455 prüfen — falls 5% pp512 Regression → re-revertieren

**Subagent-Einsatz:** 1x `subagent_explore` (background):
- Vergleiche unsere ggml-vulkan.cpp Struktur mit AtomicBot's — wo sind die CREATE_FA, mul_mat_vec, dequant Sektionen in AtomicBot's Version?

### Phase 4: DiffusionGemma portieren — P1 Hoch (3-4h)

DiffusionGemma muss auf die neue Klassenhierarchie portiert werden.

**Schritte:**
1. `src/models/diffusion-gemma.cpp` als neue Klasse `llama_model_diffusion_gemma : public llama_model_base` portieren
2. `load_arch_hparams()`, `load_arch_tensors()`, `build_arch_graph()` implementieren
3. In `src/llama-arch.cpp` registrieren
4. In `tools/mtmd/models/models.h` deklarieren
5. `tools/diffusion-cli/`, `tools/diffusion-gemma-server/`, `tools/diffusion-gemma-eval/` re-applien
6. `conversion/diffusion_gemma.py` re-applien
7. CUDA `diffusion-sampling.cu/.cuh` re-applien
8. CMakeLists.txt Anpassungen

**Subagent-Einsatz:** 1x `subagent_general` (background):
- Portiere DiffusionGemma auf neue Klassenhierarchie (selbstständige Arbeit mit Build-Verifikation)

### Phase 5: Fork-only Triviales re-applien (1h)

- `scripts/build-cuda-glibc-patch.sh` kopieren
- `tests/test-turbo-quant.c`, `tests/test-speculative-mtp.cpp` kopieren + CMakeLists
- `.github/workflows/build-turboquant-macos.yml` prüfen
- Dokumentation (`docs/fork/`, `FORKS.md`, `README.md`, `AGENTS.md`, `MTP.md`) behalten

### Phase 6: Build-Verifikation (2-3h)

- [ ] Vulkan Build: `cmake -B build -DGGML_VULKAN=ON && cmake --build build -j$(nproc)`
- [ ] Build-Fehler beheben (wahrscheinlich `#include` Pfade, fehlende Model-Registrierungen)
- [ ] CUDA Build (falls möglich): `cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda -j$(nproc)`

**Subagent-Einsatz:** 1x `subagent_general` (background):
- Kompiliere und fixe Build-Fehler iterativ

### Phase 7: Funktions-Tests (2-3h)

- [ ] Vulkan turbo3: `llama-bench -m <model> -ctk turbo3 -ctv turbo3 -fa 1 -p 512 -n 32`
- [ ] Vulkan turbo4: `llama-bench -m <model> -ctk turbo4 -ctv turbo4 -fa 1 -p 512 -n 32`
- [ ] Vulkan nps Fix: `llama-bench -m <model> -p 16384 -n 32` (darf nicht hangen)
- [ ] Vulkan tg@188k: `llama-bench -m <model> -p 512 -n 32 -c 188000` (≥20 t/s)
- [ ] Gemma4 MTP: Server starten mit MTP-Draft
- [ ] Qwen NextN: Server starten mit Qwen3.6 NextN Modell (upstream native!)
- [ ] DiffusionGemma: Build erfolgreich (Funktionstest optional)

### Phase 8: Cleanup & Merge nach master (1h)

- [ ] `qwen35-nextn.cpp` und `qwen35moe-nextn.cpp` löschen (falls noch vorhanden)
- [ ] FORKS.md aktualisieren (Sync dokumentiert)
- [ ] Commit mit klarer Message
- [ ] Merge nach master: `git checkout master && git merge feature/sync-atomicbot-2026-06-19`
- [ ] Push: `git push origin master`
- [ ] Tag: `git tag post-sync-2026-06-21`

## 6. Subagent-Plan (Parallelisierung)

| Phase | Subagent | Typ | Hintergrund? | Aufgabe |
|-------|----------|-----|-------------|---------|
| 1 | SA-1 | explore | ja | Konflikt-Katalog: welche Dateien auto-merged, welche manuell |
| 2 | SA-A | explore | ja | Core-Diff: unsere Änderungen vs. AtomicBot in Core-Dateien |
| 2 | SA-B | explore | ja | MTP-Kompatibilität: unsere server.cpp Endpunkte vs. upstream MTP |
| 3 | SA-C | explore | ja | Vulkan-Struktur: wo sind CREATE_FA/dequant Sektionen in AtomicBot's ggml-vulkan.cpp |
| 4 | SA-D | general | ja | DiffusionGemma auf neue Klassenhierarchie portieren |
| 6 | SA-E | general | ja | Build-Fehler fixen (iterativ) |

**Parallelität:** SA-1 in Phase 1. SA-A + SA-B parallel in Phase 2. SA-C in Phase 3. SA-D in Phase 4. SA-E in Phase 6.

## 7. Risiken

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|--------|-------------------|-----------|------------|
| TurboQuant Vulkan FA Pipelines passen nicht in AtomicBot's umstrukturierte ggml-vulkan.cpp | Mittel | turbo3/turbo4 FA defekt | Manuelle Anpassung; AtomicBot's flash_attn_dequant.glsl als Fallback |
| DiffusionGemma portiert nicht auf neue Klassenhierarchie | Mittel | DiffusionGemma defekt | Kann nach Merge separat fixen; Rollback wenn kritisch |
| PR #22455 verursacht 5% pp512 Regression | Hoch | Performance-Verlust | Re-revert nach Test |
| Build bricht komplett | Mittel | Keine Binaries | Iteratives Fixen mit Subagent; Rollback als Notfall |
| Vulkan nps Fix geht verloren | Niedrig | GPU-Hang zurück | Re-applien nach Merge (1 Zeile) |

## 8. Erfolgskriterien

Der Sync gilt als erfolgreich wenn:

1. ✅ `cmake --build` erfolgreich (Vulkan)
2. ✅ Vulkan turbo3 FA Benchmark: funktioniert, ≥100 t/s pp512
3. ✅ Vulkan turbo4 FA Benchmark: funktioniert
4. ✅ Vulkan pp16384: kein GPU-Hang, ≥100 t/s
5. ✅ Vulkan tg32@188k: ≥20 t/s
6. ✅ Gemma4 MTP Server: startet und produziert Tokens
7. ✅ Qwen3.6 NextN Server: startet und produziert Tokens (upstream native)
8. ✅ DiffusionGemma: Build erfolgreich
9. ✅ Keine Regression bei Standard-Modellen

## 9. Zeitschätzung

| Phase | Dauer | Kumuliert |
|-------|-------|-----------|
| 0: Vorbereitung | 30 min | 0.5h |
| 1: Merge + Katalog | 1-2h | 2.5h |
| 2: Core-Subsysteme | 2-3h | 5.5h |
| 3: TurboQuant Vulkan (P0) | 3-4h | 9.5h |
| 4: DiffusionGemma (P1) | 3-4h | 13.5h |
| 5: Triviales re-applien | 1h | 14.5h |
| 6: Build | 2-3h | 17.5h |
| 7: Tests | 2-3h | 20.5h |
| 8: Cleanup & Merge | 1h | 21.5h |

**Gesamt: 15-20 Stunden** (mit Subagent-Parallelisierung an der unteren Grenze)

## 10. Autonomie-Regeln

### 10.1 Entscheidungen die ich selbständig treffe

- Welche Dateien mit `--theirs` vs `--ours` vs manuell gemerged werden
- Build-Fehler beheben (inkl. CMakeLists, `#include` Pfade, Model-Registrierungen)
- Subagent-Aufgaben definieren und parallelisieren
- Reihenfolge der Phasen anpassen wenn Blockaden auftreten
- PR #22455 re-revertieren nach Benchmark-Test
- Commit-Messages (deutsch, wie in AGENTS.md definiert)

### 10.2 Was ich NICHT ohne User mache

- **Force-push auf master** — nur im Rollback-Fall, und dann nur auf `origin/master`
- **Push auf `origin/master` nach Sync** — erst nach vollständiger Verifikation
- **Löschen des Tags `pre-sync-2026-06-21`** — niemals
- **Änderungen an `.gitignore` die Sicherheitsregeln betreffen**

### 10.3 Kommunikations-Protokoll

- **Trilium-Tagebuch (TTT):** Bei Meilensteinen (Phase abgeschlossen, Build erfolgreich, Tests bestanden) Eintrag im Technik-Tagebuch
- **FORKS.md:** Am Ende aktualisieren mit Sync-Status
- **docs/fork/:** Dieser Masterplan wird mit Status-Updates versehen (`[x]` bei erledigten Phasen)
