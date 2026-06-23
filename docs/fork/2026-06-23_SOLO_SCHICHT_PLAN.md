# Solo-Schicht Plan: 2026-06-23 (8h)

**Erstellt:** 2026-06-23, auf lokalem System (NVIDIA RTX 3070, 8GB CUDA)
**Branch:** `feature/sync-atomicbot-2026-06-19` (sauberer Squash auf master)
AMD-APU: tg32-Tests laufen im Hintergrund (llama-bench, 2h Timeout pro Test)

---

## Parallelisierungs-Strategie (WICHTIG)

**Regel:** Maximal beschleunigen durch Subagents und Webrecherche. Nicht sequentiell arbeiten, wenn Parallelität möglich ist.

### Subagent-Einsatz (max. 8 parallel)

| Profil | Einsatzgebiet | Typische Phasen |
|--------|---------------|-----------------|
| `subagent_explore` | Code-Analyse, Architektur-Recherche, gguf-py-Struktur | Phase 2 (gguf-py), Phase 5 (Branch-Analyse) |
| `subagent_explore` | Webrecherche: DiffusionGemma PR #24423, gguf-py API, upstream-Commits | Phase 2, Phase 6 |
| `subagent_general` | Build auf Remote-Hosts (AMD-APU), Datei-Änderungen, Commits | Phase 1 (Build), Phase 4 (Merge) |
| `subagent_general` | Trilium-Doku schreiben, TTT-Einträge | Phase 3, Phase 5 |

**Parallelisierbare Tasks pro Phase:**
- **Phase 1:** Build lokal (direkt) + Build auf AMD-APU (subagent) parallel
- **Phase 2:** gguf-py-Analyse (subagent_explore) + Webrecherche PR #24423 (subagent_explore) + constants.py-Edit (direkt) parallel
- **Phase 3:** APU-Test-Log abholen (subagent) + Trilium-Doku (direkt) parallel
- **Phase 5:** diffusion-gemma-v2 Squash (subagent_general) + TTT-Eintrag (direkt) + git gc (direkt) parallel
- **Phase 6:** Modell-Suche auf AMD-APU (subagent) + Webrecherche DiffusionGemma GGUF (subagent_explore) parallel

### Webrecherche

- `web_search` primär für: gguf-py API-Doku, DiffusionGemma PR-Details, upstream-Commits
- `arxiv-mcp` falls akademische Diffusion-Decoder-Paper nötig
- Brave/Exa als Fallback

**Prinzip:** Jede Phase startet mit 2-4 parallel laufenden Subagents für Recherche/Analyse, während die eigentliche Arbeit (Build, Edit, Commit) direkt erfolgt. Subagents liefern Kontext, der Agent entscheidet und implementiert.

---

## Phase 1: Build-Verifikation des Feature-Branch (0-1.5h)

**Ziel:** Beweisen, dass der Squash-Branch kompiliert und funktioniert.

**Arbeiten:**
- `cmake --build build -j$(nproc)` auf lokalem System (CUDA)
- `./build/bin/test-chat` laufen lassen (Parser-Tests)
- `./build/bin/llama-bench -m <kleines Modell> -p 512 -n 32` als Smoke-Test
- Bei Fehlern: Fix auf dem Branch committen

**Verifikation:**
- [x] Build exit code 0
- [x] `test-chat` output: "All tests passed!"
- [x] `llama-server --version`: 9157 (abb380935)
- [x] `git log --oneline` zeigt saubere Historie ohne Abbrüche
- [x] `llama-diffusion-cli` gebaut

**Zusätzliche Arbeiten in Phase 1:**
- 14 fehlende Klassendeklarationen in models.h hinzugefügt (ISWA, NextN, T5, Pangu, Qwen3VLMoE)
- bq/bk/bv/bo Bias-Tensoren zu llama_layer hinzugefügt
- 6 orphaned Modell-Dateien vom Build ausgeschlossen (API-Inkompatibilität)
- diffusion-cli CMakeLists.txt: common → llama-common

---

## Phase 2: DiffusionGemma gguf-py Registrierung (1.5-3.5h)

**Ziel:** Das letzte bekannte Feature-Gap schließen. `MODEL_ARCH.DIFFUSION_GEMMA` und zugehörige Tensor-Mappings in gguf-py hinzufügen.

**Arbeiten:**
1. `gguf-py/gguf/constants.py`: `MODEL_ARCH.DIFFUSION_GEMMA` zum IntEnum + Namen-Mapping
2. `gguf-py/gguf/tensor_mapping.py`: DiffusionGemma-Tensoren mappen (`pkv_k`, `pkv_v`, `canvas_length` etc.)
3. `gguf-py/gguf/quants.py`: Falls spezielle Quantisierungs-Typen nötig
4. `convert_hf_to_gguf.py`: DiffusionGemma-Model-Klasse hinzufügen (wenn nötig)
5. Test: `python3 -c "from gguf import MODEL_ARCH; print(MODEL_ARCH.DIFFUSION_GEMMA)"`

**Verifikation:**
- [x] `python3 -c "from gguf import MODEL_ARCH; print(MODEL_ARCH.DIFFUSION_GEMMA)"` funktioniert (133)
- [x] `grep DIFFUSION_GEMMA gguf-py/gguf/constants.py` zeigt den Eintrag
- [x] Build mit den Änderungen: `cmake --build build` erfolgreich
- [x] Commit auf Branch mit deutscher Nachricht (9a7d2d9f4)

---

## Phase 3: APU Test-Ergebnisse einsammeln + Trilium-Doku (3.5-5h)

**Ziel:** tg32-Scaling-Ergebnisse von AMD-APU abholen, in Trilium dokumentieren.

**Arbeiten:**
1. APU-Test-Log abholen: `/tmp/tg32_bench_results.log`
2. Ergebnisse extrahieren (pp t/s, tg t/s für 180k, 188k, 196k, 262k)
3. In Trilium (LLM-Benchmarks — AMD-APU) eintragen
4. §5.8 in `SWumEN7WOXBI` (Vulkan Performance-Klippe) aktualisieren
5. TTT-Eintrag für die Session erstellen

**Verifikation:**
- [ ] 4 tg32-Werte in Trilium-Tabelle (180k, 188k, 196k, 262k)
- [ ] TTT-Eintrag existiert mit Fazit
- [ ] `SWumEN7WOXBI` §5.8 zeigt aktualisierte Werte

---

## Phase 4: Sync-Merge nach master + Push (5-6.5h)

**Ziel:** Verifizierten Feature-Branch nach master mergen und pushen.

**Arbeiten:**
1. `git checkout master && git merge feature/sync-atomicbot-2026-06-19` (fast-forward)
2. Build-Verifikation auf master (quick smoke test)
3. `git push origin master` (Codeberg)
4. `git push github master` (Mirror)
5. FORKS.md §7 aktualisieren

**Verifikation:**
- [ ] `git log --oneline master` zeigt Sync-Commit als HEAD
- [ ] `git push` exit code 0
- [ ] Codeberg-Web-UI zeigt den neuen Commit
- [ ] `./build/bin/llama-server --version` funktioniert auf master

---

## Phase 5: Aufräumen + TTT (6.5-8h)

**Ziel:** Wissensstand sichern, offene Punkte dokumentieren.

**Arbeiten:**
1. `feature/diffusion-gemma-v2` als Squash neu aufbauen (korrupte Historie entfernen)
2. Obsolete Remote-Tracking-Branches aufräumen
3. `git gc` versuchen (sollte nach Reparatur funktionieren)
4. TTT-Eintrag: komplette Session dokumentieren
5. AGENTS.md aktualisieren, falls neue Erkenntnisse
6. Offene TODO-Liste für nächste Session erstellen

**Verifikation:**
- [x] `feature/diffusion-gemma-v2` als Squash neu aufgebaut (Subagent)
- [x] TTT-Eintrag erstellt (Tag 23, ID: oBLB6OpLONkq)
- [x] Trilium: Kontext-Scaling Note für 12B erstellt (WjqL5Ky9Z3Hf)
- [x] Alle Branches in `git branch -v` sind sauber und lesbar
- [ ] `git gc --prune=now` — scheitert noch an korrupten Objekten in alten Packs
  (kosmetisches Problem, alle Branches funktionieren)
- [ ] `git fsck` zeigt noch 1 fehlerhafte Verknüpfung (unreferenziert)

---

## Phase 6 (Stretch): DiffusionGemma Live-Test (falls Zeit übrig)

**Ziel:** DiffusionGemma-Modell mit echtem Text-Output testen — ca. 1000 Token Ausgabe zum Thema "KI".

**System:** AMD-APU (760M, Vulkan, 32GB RAM — mehr VRAM via GTT) bevorzugt. NVIDIA-System (RTX 3070, 8GB CUDA) als Fallback.

**Arbeiten:**
1. Auf AMD-APU: DiffusionGemma GGUF laden (falls verfügbar)
2. `llama-cli -m <diffusion-gemma.gguf> -p "Schreibe einen Aufsatz über Künstliche Intelligenz" -n 1000`
3. Falls auf AMD-APU nicht möglich (kein Modell): lokales System mit kleinstem verfügbaren DiffusionGemma-Modell
4. Output prüfen: Ist der Text kohärent? Entspricht er dem Diffusion-Decoder-Pattern?
5. Ergebnisse in Trilium dokumentieren (unter DiffusionGemma-Note)

**Verifikation:**
- [x] `llama-diffusion-cli --help` funktioniert
- [x] DiffusionGemma GGUF verfügbar (26B Q4_K_M, 16GB) — lokal
- [x] Tensor `enc_layer_output_scale` wird korrekt geladen (Build-Fix: alter Build hatte `enc_layer_out_scale`)
- [x] Assert-Fehler behoben: `--diffusion-eps` Parameter muss gesetzt werden (Default 0 → Assert)
- [ ] `llama-diffusion-cli` startet: **NICHT MÖGLICH auf hydra** (26B Q4_K_M = 16GB, OOM bei 17GB freiem RAM)
- [ ] Auf APU: kein DiffusionGemma GGUF verfügbar (16GB Transfer zu groß)
- [x] Fazit: DiffusionGemma 26B benötigt mindestens 32GB freien RAM oder eine kleinere Quantisierung

**Erkenntnis:** Der `llama-diffusion-cli` benötigt zwingend `--diffusion-eps F` oder `--diffusion-block-length N` (Default beides 0 → Assert-Fehler). Das 26B-Modell ist für 8GB-VRAM-Systeme zu groß. Eine kleinere Quantisierung (IQ2/IQ3) oder ein kleineres Modell wäre nötig.

---

## Risiko-Bewertung

| Phase | Risiko | Mitigation |
|-------|--------|------------|
| Build | Kompilierungsfehler durch Squash | Falls Fehler: gezielt fixen, nicht abbrechen |
| gguf-py | API-Änderungen in gguf-py | Erst `constants.py` lesen, dann minimal-invasiv ändern |
| APU | Tests timeout (>2h pro Test) | Wenn timeout: kleinere Kontextgrößen als Fallback |
| Merge | master könnte neue Commits haben | `git fetch origin` vor Merge, rebase falls nötig |
| Push | Codeberg/GitHub nicht erreichbar | Push kann auf nächste Session verschoben werden |
| Diffusion | Keine GGUF verfügbar | Phase 6 entfällt, mehr Zeit für Phase 5 |

## Was NICHT getan wird

- Keine `git push --force` auf master
- Keine Löschung von `feature/diffusion-gemma-v2` ohne Squash-Backup
- Keine Änderungen an AGENTS.md-Verträgen ohne DOX-Durchlauf
- Keine Commits mit Host-Namen oder privaten Daten

---

*Plan erstellt: 2026-06-23*
*Gültig für: Solo-Schicht 2026-06-23*
*Nächste Session: Verifikation der Checkboxen, offene Punkte übernehmen*
