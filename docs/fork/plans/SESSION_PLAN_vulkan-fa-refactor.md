# SESSION_PLAN: Vulkan FlashAttention Refactor (M2)

**Erstellt:** 2026-07-11
**Typ:** Solo-Session (Cherry-Pick + Vulkan Benchmark)
**Status:** ☐ offen
**Meilenstein:** M2 — Vulkan-Offensive
**ROADMAP-Item:** #7 — Vulkan FlashAttention Refactor (PR #19625)
**Abhängigkeit:** Keine (unabhängig von #6, aber #6 sollte zuerst kommen für saubere Shader-Basis)

---

## Session-Ziel

PR #19625 (Vulkan FlashAttention Refactor: row splitting, shared memory staging, Q caching in registers, fused Lf accumulation, vendor-specific Br selection) cherry-picken und auf Mars (RDNA3) benchmarken. Erwarteter Gain: 10-20% improvement auf scalar FA path.

## Akzeptanzkriterien

- [ ] PR #19625 ist cherry-picked und kompiliert auf Mars (Vulkan build)
- [ ] Build ist grün auf Mars
- [ ] `llama-bench` auf Mars zeigt FA-Performance-Verbesserung (10-20% erwartet)
- [ ] turbo4 FA funktioniert weiterhin (keine Regression bei turbo4/turbo4)
- [ ] turbo3 scalar fallback funktioniert (keine Regression bei turbo3/turbo3)
- [ ] MTP-Akzeptanz nicht regressiert (turbo4 ~75-100%, turbo3 ~50-60%)
- [ ] ROADMAP.md Status für #7 auf ✅ gesetzt

## Verifikations-Strategie

| Metrik | Vorher | Nachher | Test-Befehl |
|--------|--------|---------|-------------|
| PP t/s (turbo4/4) | Baseline | Mit Refactor | `llama-bench -m <qat-26b> -p 512 -n 0 -ctk turbo4 -ctv turbo4` auf Mars |
| PP t/s (turbo3/3) | Baseline | Mit Refactor | `llama-bench -m <qat-26b> -p 512 -n 0 -ctk turbo3 -ctv turbo3` auf Mars |
| TG t/s (turbo3/4) | 25 t/s | Mit Refactor | `llama-bench -m <qat-26b> -p 0 -n 128 -ctk turbo3 -ctv turbo4` auf Mars |
| MTP Akzeptanz | turbo4 ~75-100% | Mit Refactor | `llama-cli -m <qat-26b> --spec ... --spec-draft-n-max 4` auf Mars |

## Schritte

1. ☐ **PR-Verifikation:** `gh pr view 19625 --repo ggerganov/llama.cpp --json title,state,mergedAt`
2. ☐ **PR-Inhalt analysieren:** Welche Shader-Dateien werden geändert? flash_attn.comp, flash_attn_cm1.comp?
3. ☐ **Baseline-Benchmark auf Mars:** PP/TG mit turbo3/4, turbo4/4, turbo3/3
4. ☐ **Cherry-Pick:** `git cherry-pick <commit-hash>` — bei Konflikten: FA-Shader manuell mergen
5. ☐ **Build auf Mars:** `cmake --build build -j$(nproc)` mit `-DLLAMA_VULKAN=ON`
6. ☐ **Benchmark auf Mars:** PP/TG mit turbo3/4, turbo4/4, turbo3/3 — Vorher/Nachher vergleichen
7. ☐ **MTP-Test:** Akzeptanzrate mit turbo4 verifizieren (nicht regressiert)
8. ☐ **Vergleich:** Vorher/Nachher-Tabelle
9. ☐ **ROADMAP.md aktualisieren:** #7 auf ✅ oder ❌
10. ☐ **CHANGELOG.md + Commit + Push**
11. ☐ **TTT-Eintrag**

## Blockaden / User-Eingriffe

| Blockade | Wahrscheinlichkeit | Loesung |
|----------|-------------------|---------|
| Merge-Konflikt mit turbo3/turbo4 FA-Shadern | Hoch | Fork hat turbo3 FA deaktiviert (glslc bug) und turbo4 FA aktiv (flash_attn_cm1.comp) — Refactor muss beide Pfade erhalten |
| glslc bug bei turbo3 FA tritt wieder auf | Mittel | turbo3 FA bleibt deaktiviert (scalar fallback) — Refactor darf das nicht ändern |
| Performance-Regression | Mittel | Wenn Regression >5%: revertieren, als ❌ markieren |
| MoltenVK+AMD subgroupShuffleXor bug | Niedrig | Workaround in PR #23218 — prüfen ob benötigt |

## Defaults

- **turbo3 FA bleibt DEAKTIVIERT:** Der Refactor darf den glslc-bug-Workaround nicht entfernen. turbo3 nutzt weiterhin scalar fallback.
- **turbo4 FA bleibt aktiviert:** flash_attn_cm1.comp muss weiterhin funktionieren.
- **Regression >5%:** Revertieren. Neutral (±2%) ist akzeptabel wenn langfristig besserer Code-Pfad.

## Recherche-Fallbacks

| Problem | Reaktion |
|---------|----------|
| FA-Shader Syntax unklar | webfetch auf PR #19625 + verlinkte Commits |
| Br-Selection unklar | web_search "Vulkan FlashAttention row splitting Br selection AMD" |
| Merge-Konflikt | `git diff` auf flash_attn*.comp, manuelle Integration unter Erhalt von turbo3/turbo4 Pfaden |
| MTP-Akzeptanz regressiert | Root Cause: FA-Änderung beeinflusst Attention-Output → prüfen ob Shader korrekt gemergt |

## System-Zugriff

| System | Zweck | SSH |
|--------|-------|-----|
| Hydra (lokal) | Cherry-Pick, git-Operationen | — |
| Mars | Vulkan Build, Benchmark, MTP-Test | `ssh mars` |
