# SESSION_PLAN: MUL_MAT_ID Subgroup Optimization (M2)

**Erstellt:** 2026-07-11
**Typ:** Solo-Session (Cherry-Pick + Vulkan Benchmark)
**Status:** ☐ offen
**Meilenstein:** M2 — Vulkan-Offensive
**ROADMAP-Item:** #6 — Vulkan MUL_MAT_ID Subgroup Optimization (PR #15524)

---

## Session-Ziel

PR #15524 (subgroup ballot operations für MoE-Expert-Selection bei Vulkan) cherry-picken und auf Mars (RDNA3) und Venus (GCN) benchmarken. Erwarteter Gain: bis zu +657% MoE PP auf AMD GPUs.

## Akzeptanzkriterien

- [ ] PR #15524 ist cherry-picked und kompiliert auf Mars (Vulkan build)
- [ ] Build ist grün auf Mars: `cmake --build build -j$(nproc)` mit `-DLLAMA_VULKAN=ON`
- [ ] `llama-bench` auf Mars zeigt PP-Speedup bei MoE-Modell (Gemma-4 26B A4B QAT)
- [ ] Keine Regression bei tg (token generation)
- [ ] Keine Regression bei non-MoE-Modellen (falls testbar)
- [ ] ROADMAP.md Status für #6 auf ✅ gesetzt

## Verifikations-Strategie

| Metrik | Vorher | Nachher | Test-Befehl |
|--------|--------|---------|-------------|
| PP t/s (Mars, MoE) | Baseline messen | Mit PR #15524 | `llama-bench -m <qat-26b> -p 512 -n 0` auf Mars |
| TG t/s (Mars, MoE) | Baseline messen | Mit PR #15524 | `llama-bench -m <qat-26b> -p 0 -n 128` auf Mars |
| PP t/s (Mars, non-MoE) | Baseline messen | Mit PR #15524 | `llama-bench -m <non-moe-modell> -p 512 -n 0` auf Mars |

**Baseline:** Aktuelle Produktion (QAT @ 224k, turbo3/4, 25 t/s tg auf Mars)

## Schritte

1. ☐ **PR-Verifikation:** `gh pr view 15524 --repo ggerganov/llama.cpp --json title,state,mergedAt` — prüfen ob PR existiert
2. ☐ **PR-Inhalt analysieren:** webfetch auf github.com/ggerganov/llama.cpp/pull/15524 — welche Dateien, welche Shader, welche Änderungen
3. ☐ **Baseline-Benchmark auf Mars:** `llama-bench -m <qat-26b> -p 512 -n 128` — PP und TG Baseline erfassen
4. ☐ **Cherry-Pick:** `git cherry-pick <commit-hash>` — bei Konflikten: Vulkan-Shader-Dateien manuell mergen
5. ☐ **Build auf Mars:** `ssh mars 'cd ~/fukuro-llama-cpp-turboquant && cmake --build build -j$(nproc)'` — muss grün sein
6. ☐ **Benchmark auf Mars:** `llama-bench -m <qat-26b> -p 512 -n 128` — mit PR #15524
7. ☐ **Vergleich:** Vorher/Nachher-Tabelle erstellen
8. ☐ **Venus-Test (optional):** Wenn Zeit: Build auf Venus (GCN) und Benchmark — 657% Speedup wurde auf GCN gemessen
9. ☐ **ROADMAP.md aktualisieren:** #6 auf ✅ oder ❌
10. ☐ **CHANGELOG.md Eintrag**
11. ☐ **Commit + Push**
12. ☐ **TTT-Eintrag**

## Blockaden / User-Eingriffe

| Blockade | Wahrscheinlichkeit | Loesung |
|----------|-------------------|---------|
| PR #15524 existiert nicht | Niedrig | webfetch verifiziert |
| Merge-Konflikt mit Vulkan-Shadern | Hoch | Shader-Dateien manuell mergen; Fork hat eigene Shader (turbo3, fwht) — Konflikte expected |
| Subgroup ballot auf RDNA3 nicht unterstützt | Niedrig | RDNA3 unterstützt subgroup ballot; Venus (GCN) auch |
| Performance-Regression statt Speedup | Mittel | Wenn Regression: revertieren, als ❌ markieren, Root Cause analysieren |
| Venus nicht erreichbar (Suspend 08-13 Uhr) | Hoch | Venus-Test nur außerhalb 08-13 Uhr; Mars-Test reicht als Akzeptanz |

## Defaults

- **Merge-Konflikt bei Vulkan-Shadern:** Nur die subgroup-ballot-Änderungen portieren, nicht ganze Dateien überschreiben. Fork-spezifische Shader (turbo3, fwht) müssen erhalten bleiben.
- **Regression:** Wenn PP nicht verbessert wird: trotzdem behalten wenn tg nicht regressiert (neutral). Wenn tg regressiert: revertieren.
- **Venus-Test:** Optional — Mars-Test ist primäre Akzeptanz. Venus nur wenn Zeit und nicht in Suspend-Zeit.

## Recherche-Fallbacks

| Problem | Reaktion |
|---------|----------|
| PR-Inhalt unklar | webfetch auf PR-Seite + verlinkte Commits |
| Subgroup ballot Syntax unklar | web_search "Vulkan subgroupBallot compute shader" |
| Merge-Konflikt Lösung unklar | `git diff` auf konfliktierende Dateien, manuelle Integration |
| Benchmark inkonsistent | 3 Runs mitteln, Ausreißer verwerfen |

## System-Zugriff

| System | Zweck | SSH |
|--------|-------|-----|
| Hydra (lokal) | Cherry-Pick, git-Operationen | — |
| Mars | Vulkan Build, Benchmark | `ssh mars` |
| Venus | Optional: GCN Benchmark | `ssh venus` (nicht 08-13 Uhr!) |
