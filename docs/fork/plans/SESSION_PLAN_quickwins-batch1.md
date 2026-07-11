# SESSION_PLAN: Quick-Win-Batch 1 (M1)

**Erstellt:** 2026-07-11
**Typ:** Solo-Session (Cherry-Pick + Konfiguration + Aktivierung)
**Status:** ☐ offen
**Meilenstein:** M1 — Quick-Win-Welle
**ROADMAP-Items:** #1 (MTP Logits Copy), #2 (Tensor Split Regex), #4 (n-gram Decoding), #5 (GTT Size Tuning)

---

## Session-Ziel

Vier Quick-Wins (<1 Tag jeweils) in einer kombinierten Solo-Session umsetzen: Drei Cherry-Picks aus mainline und eine Konfigurationsänderung. Alle vier sind konfigurations- oder code-level Änderungen ohne komplexe Implementierung.

## Akzeptanzkriterien

- [ ] PR #23198 (MTP Logits Copy) ist cherry-picked und kompiliert auf Hydra (CUDA build)
- [ ] PR #24710 (Tensor Split Regex) ist cherry-picked und kompiliert auf Hydra
- [ ] n-gram Decoding ist auf Mars getestet (`--spec-type ngram-mod,ngram-map-k4v`)
- [ ] GTT Size ist auf Mars erhöht und VRAM-Verfügbarkeit verifiziert
- [ ] Build ist grün auf Hydra (`cmake --build build -j$(nproc)`)
- [ ] Keine Regressionen: `llama-bench` auf Mars zeigt keine Verschlechterung vs. Baseline
- [ ] ROADMAP.md Status für #1, #2, #4, #5 auf ✅ gesetzt

## Verifikations-Strategie

| Item | Metrik Vorher | Metrik Nachher | Test-Befehl |
|------|---------------|----------------|-------------|
| #1 MTP Logits Copy | PP t/s mit MTP (Mars) | PP t/s mit MTP nach PR | `llama-bench -m <qat> -p 512 -n 0 --spec ...` auf Mars |
| #2 Tensor Split Regex | decode thread utilization | decode thread utilization | Nur auf Uranus messbar (Multi-GPU), auf Hydra nicht testbar |
| #4 n-gram Decoding | tg t/s ohne n-gram | tg t/s mit n-gram | `llama-bench -m <qat> -p 512 -n 128 --spec-draft-n-max 4 --spec-type ngram-mod` auf Mars |
| #5 GTT Size | `radeontop` VRAM verfügbar | `radeontop` VRAM verfügbar nach Tuning | `ssh mars 'cat /sys/class/drm/card0/device/gtt_size'` oder `radeontop` |

## Schritte

1. ☐ **PR-Verifikation:** `gh pr view 23198 --repo ggerganov/llama.cpp --json title,state,mergedAt` und `gh pr view 24710` — prüfen ob PRs existieren und merged sind
2. ☐ **Cherry-Pick #1:** `git cherry-pick <commit-hash>` für PR #23198 — bei Konflikten: nur relevante Dateien, nicht blind mergen
3. ☐ **Cherry-Pick #2:** `git cherry-pick <commit-hash>` für PR #24710
4. ☐ **Build auf Hydra:** `cmake --build build -j$(nproc)` — muss grün sein
5. ☐ **Build auf Mars:** `ssh mars 'cd ~/fukuro-llama-cpp-turboquant && cmake --build build -j$(nproc)'` — muss grün sein (Vulkan)
6. ☐ **Benchmark #1 auf Mars:** `llama-bench` mit MTP vor und nach dem Cherry-Pick — PP-Speed vergleichen
7. ☐ **n-gram Decoding auf Mars:** `llama-bench -m <qat-modell> -p 512 -n 128 --spec-draft-n-max 4 --spec-type ngram-mod` — tg-Speed mit/ohne n-gram vergleichen
8. ☐ **GTT Size Tuning auf Mars:** Aktuelle GTT-Größe prüfen, erhöhen, `radeontop` verifizieren
9. ☐ **ROADMAP.md aktualisieren:** #1, #2, #4, #5 auf ✅
10. ☐ **CHANGELOG.md Eintrag**
11. ☐ **Commit + Push**
12. ☐ **TTT-Eintrag**

## Blockaden / User-Eingriffe

| Blockade | Wahrscheinlichkeit | Loesung |
|----------|-------------------|---------|
| PR #23198 existiert nicht oder ist nicht merged | Mittel | webfetch auf github.com/ggerganov/llama.cpp/pull/23198 — wenn nicht existent: Item als ❌ markieren, überspringen |
| PR #24710 hat Merge-Konflikte mit Fork | Mittel | Nur relevante Dateien manuell portieren, nicht blind cherry-pick |
| GTT Size kann nicht geändert werden (BIOS-Lock) | Niedrig | Wenn nicht änderbar: Item als ❌ markieren mit Begründung |
| Mars nicht erreichbar (SSH) | Niedrig | Build/Benchmark auf Hydra (CPU-only) durchführen, Mars-Tests verschieben |
| n-gram Decoding nicht im Fork verfügbar | Niedrig | Prüfen ob `--spec-type` Flag existiert; wenn nicht: Feature-Branch von mainline pullen |

## Defaults

- **Cherry-Pick Konflikte:** Nur relevante Dateien manuell portieren, nicht blind mergen. Wenn >5 Konflikte: Item überspringen und als ❌ markieren.
- **Benchmark-Regression:** Wenn ein Cherry-Pick die Performance verschlechtert: revertieren und als ❌ markieren mit Begründung.
- **GTT Size:** Default-Wert 3GB (50% von 6GB RAM auf Mars LXC). Ziel: 12GB (40% von 30GB Host-RAM). Wenn nicht machbar: maximal möglicher Wert.

## Recherche-Fallbacks

| Problem | Reaktion |
|---------|----------|
| PR-Nummer falsch | webfetch auf github.com/ggerganov/llama.cpp/pull/<nummer> — prüfen ob Titel passt |
| Cherry-Pick schlägt fehl | `git log --all --oneline --grep="<PR-Titel>"` — alternativen Commit suchen |
| n-gram Flag unbekannt | `llama-bench --help | grep -i spec` — verfügbare Spec-Optionen prüfen |
| GTT-Konfiguration unklar | web_search "AMD GTT size tuning Linux kernel parameter" |

## System-Zugriff

| System | Zweck | SSH |
|--------|-------|-----|
| Hydra (lokal) | Build (CUDA), Cherry-Pick | — |
| Mars | Vulkan Build, Benchmark, GTT Tuning | `ssh mars` |
