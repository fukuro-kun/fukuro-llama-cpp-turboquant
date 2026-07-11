# SESSION_PLAN: Solo-Session 2026-07-11 — M1+M2 Roadmap

**Erstellt:** 2026-07-11 05:10
**Typ:** Solo-Session (M1 Quick-Wins + M2 Vulkan-Offensive Start)
**Status:** ⏳ in Arbeit
**Zeitfenster:** 05:00–14:00 Uhr (~9 Stunden)
**Roadmap-Items:** #1, #2, #4, #5, #6, #7, #9

---

## Session-Ziel

M1 (Quick-Win-Welle) komplett abschließen und M2 (Vulkan-Offensive) starten. 7 Items bearbeiten: 3 Cherry-Picks aus mainline, 1 Multi-GPU-Test auf Uranus, 1 Konfiguration, 2 Vulkan-Shader-Cherry-Picks.

## Entscheidungen (Interview-Protokoll)

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Styx Server stoppen? | Ja, für heute erlaubt | Produktiv-Service kann kurz offline sein |
| Mars git pull + rebuild? | Ja, aber Performance prüfen | Falls Degradierung ohne Ursache: revert auf schnelleren Stand |
| Uranus #2 Tensor Split Regex? | Uranus ist online! | Extra für diese Session hochgefahren |
| Reihenfolge? | M1 dann M2, parallel möglich | Max 2 subagents + ich selbst |

## Phasen

### Phase 0: Vorbereitung (05:10–05:40, 30 min)
- Mars: git pull + rebuild (Vulkan)
- Uranus: git pull (auf neuesten Stand)
- PRs verifizieren: #23198, #24710, #15524, #19625, #20897
- Styx: Server stoppen für später Benchmarks

### Phase 1: M1 Quick-Win-Batch (05:40–08:00, ~2.5h)
- #1 MTP Logits Copy (PR #23198) — cherry-pick, build, Mars benchmark
- #2 Tensor Split Regex (PR #24710) — cherry-pick, Uranus benchmark (Multi-GPU!)
- #4 n-gram Decoding — Mars aktivieren/testen
- #5 GTT Size Tuning — Mars konfigurieren

### Phase 2: M2 Vulkan-Offensive (08:00–13:00, ~5h)
- #6 MUL_MAT_ID Subgroup (PR #15524) — cherry-pick, Mars benchmark (größter Hebel!)
- #7 Vulkan FA Refactor (PR #19625) — cherry-pick, Mars benchmark
- #9 Vulkan Shmem-Staging (PR #20897) — cherry-pick, Mars benchmark

### Phase 3: Doku + Cleanup (13:00–14:00, 1h)
- ROADMAP.md aktualisieren (alle erledigten Items auf ✅)
- CHANGELOG.md Einträge
- Venus Suspend prüfen (08-13 Uhr → muss schlafen)
- TTT-Eintrag
- Commit + Push

## Akzeptanzkriterien

- [ ] M1 komplett: #1, #2, #4, #5 alle auf ✅ oder ❌ (mit Begründung)
- [ ] M2 gestartet: #6 auf ✅ oder ❌; #7 und #9 mindestens cherry-picked + build grün
- [ ] Mars Build ist grün nach allen Cherry-Picks
- [ ] Uranus Build ist grün für #2
- [ ] Keine Performance-Regression auf Mars (Baseline: 25 t/s tg mit QAT @ 224k)
- [ ] ROADMAP.md ist aktualisiert
- [ ] CHANGELOG.md hat Einträge für alle erledigten Items
- [ ] TTT-Eintrag erstellt

## Verifikations-Strategie

| Item | Metrik | Vorher | Nachher | System |
|------|--------|--------|---------|--------|
| #1 MTP Logits | PP t/s mit MTP | Baseline | Mit PR #23198 | Mars |
| #2 Tensor Split Regex | decode thread util | Baseline | Mit PR #24710 | Uranus |
| #4 n-gram Decoding | tg t/s mit n-gram | 25 t/s | Mit n-gram | Mars |
| #5 GTT Size | VRAM verfügbar | ~15GB | Nach Tuning | Mars |
| #6 MUL_MAT_ID | PP t/s MoE | Baseline | Mit PR #15524 | Mars |
| #7 Vulkan FA | PP/TG t/s | Baseline | Mit PR #19625 | Mars |
| #9 Shmem-Staging | TG t/s | Baseline | Mit PR #20897 | Mars |

## Defaults

- **Cherry-Pick Konflikt:** Nur relevante Dateien manuell portieren. Wenn >5 Konflikte: Item überspringen, als ❌ markieren.
- **Performance-Regression >5%:** Revertieren, als ❌ markieren mit Begründung.
- **PR existiert nicht:** webfetch verifizieren. Wenn nicht existent: ❌ markieren.
- **Mars Degradierung nach git pull:** Dokumentieren, Ursache suchen. Falls keine Ursache: revert auf 517ec94d.
- **Venus:** 08-13 Uhr Suspend-Policy. Nicht anfassen.
- **Styx Server:** Für Benchmarks stoppen, danach wieder starten.

## Recherche-Fallbacks

| Problem | Reaktion |
|---------|----------|
| PR-Nummer falsch | webfetch auf github.com/ggerganov/llama.cpp/pull/<nummer> |
| Cherry-Pick schlägt fehl | `git log --all --oneline --grep="<PR-Titel>"` — alternativen Commit suchen |
| Build schlägt fehl | `git diff` prüfen, inkrementell zurückbauen |
| Vulkan-Shader Konflikt | Nur subgroup/FA-Änderungen portieren, turbo3/turbo4/fwht Shader erhalten |
| Benchmark inkonsistent | 3 Runs mitteln, Ausreißer verwerfen |

## System-Zugriff

| System | Zweck | SSH | Status |
|--------|-------|-----|--------|
| Hydra (lokal) | Cherry-Picks, git, build (CUDA) | — | ✅ online |
| Mars | Vulkan build, benchmark, GTT | `ssh mars` | ✅ online, needs pull |
| Styx | CUDA benchmark (#3 Pascal MMVQ — M3, nicht heute) | `ssh styx` | ✅ online, server läuft |
| Uranus | Multi-GPU benchmark (#2) | `ssh uranus` | ✅ online |
| Venus | — | `ssh venus` | ⚠️ Suspend 08-13 Uhr |

## Offene Fragen

Keine — alle im Interview geklärt.
