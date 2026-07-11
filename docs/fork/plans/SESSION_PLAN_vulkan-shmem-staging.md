# SESSION_PLAN: Vulkan Shared Memory Staging Kernel (M2)

**Erstellt:** 2026-07-11
**Typ:** Solo-Session (Cherry-Pick + Vulkan Benchmark)
**Status:** ☐ offen
**Meilenstein:** M2 — Vulkan-Offensive
**ROADMAP-Item:** #9 — Vulkan Shared Memory Staging Kernel (PR #20897)

---

## Session-Ziel

PR #20897 (shmem-staging für Matrix-Vector Kernel: zwei-Phasen-Ansatz mit raw uint64 loads → shared memory → dot product) cherry-picken und auf Mars (RDNA3) testen. Auf Intel Arc >2.5x TG Speedup gemessen — RDNA-Potential unklar, muss benchmarkt werden.

## Akzeptanzkriterien

- [ ] PR #20897 ist cherry-picked und kompiliert auf Mars (Vulkan build)
- [ ] Build ist grün auf Mars
- [ ] `llama-bench` auf Mars zeigt TG-Performance-Verbesserung (Ziel: >10%)
- [ ] Keine Regression bei PP
- [ ] Keine Regression bei turbo3/turbo4 KV-Cache
- [ ] ROADMAP.md Status für #9 auf ✅ gesetzt

## Verifikations-Strategie

| Metrik | Vorher | Nachher | Test-Befehl |
|--------|--------|---------|-------------|
| TG t/s (turbo3/4) | 25 t/s Baseline | Mit shmem-staging | `llama-bench -m <qat-26b> -p 0 -n 128 -ctk turbo3 -ctv turbo4` auf Mars |
| TG t/s (turbo4/4) | Baseline messen | Mit shmem-staging | `llama-bench -m <qat-26b> -p 0 -n 128 -ctk turbo4 -ctv turbo4` auf Mars |
| PP t/s | Baseline messen | Mit shmem-staging | `llama-bench -m <qat-26b> -p 512 -n 0` auf Mars |

## Schritte

1. ☐ **PR-Verifikation:** `gh pr view 20897 --repo ggerganov/llama.cpp --json title,state,mergedAt`
2. ☐ **PR-Inhalt:** Welche Dateien? mul_mat_vec Shader? Ist es Intel-spezifisch oder generisch?
3. ☐ **Baseline-Benchmark auf Mars:** TG und PP Baseline
4. ☐ **Cherry-Pick:** `git cherry-pick <commit-hash>`
5. ☐ **Build auf Mars:** `cmake --build build -j$(nproc)` mit `-DLLAMA_VULKAN=ON`
6. ☐ **Benchmark auf Mars:** TG und PP — Vorher/Nachher
7. ☐ **Vergleich:** Vorher/Nachher-Tabelle
8. ☐ **ROADMAP.md aktualisieren:** #9 auf ✅ oder ❌
9. ☐ **CHANGELOG.md + Commit + Push**
10. ☐ **TTT-Eintrag**

## Blockaden / User-Eingriffe

| Blockade | Wahrscheinlichkeit | Loesung |
|----------|-------------------|---------|
| PR ist Intel-spezifisch (nicht auf RDNA portierbar) | Mittel | webfetch prüfen ob AMD-spezifischer Pfad existiert; wenn Intel-only: als ❌ markieren |
| Merge-Konflikt mit TurboQuant mul_mat_vec Shadern | Hoch | Fork hat eigene `mul_mat_vec_tq4_1s` Shader — Konflikte expected, manuell mergen |
| Kein Speedup auf RDNA | Mittel | Wenn <5% Speedup: neutral behalten wenn keine Regression. Wenn Regression: revertieren. |
| shared memory Limit auf RDNA3 | Niedrig | RDNA3 hat 64KB shared memory per workgroup — sollte ausreichen |

## Defaults

- **Intel-only PR:** Wenn der PR Intel-spezifische Shader enthält die nicht auf RDNA laufen: als ❌ markieren, aber Erkenntnis dokumentieren für späteren RDMA-Port.
- **Regression:** Wenn TG regressiert >3%: revertieren.
- **Neutral (±3%):** Behalten wenn Code-Pfad sauberer ist (langfristig besser).

## System-Zugriff

| System | Zweck | SSH |
|--------|-------|-----|
| Hydra (lokal) | Cherry-Pick, git-Operationen | — |
| Mars | Vulkan Build, Benchmark | `ssh mars` |
