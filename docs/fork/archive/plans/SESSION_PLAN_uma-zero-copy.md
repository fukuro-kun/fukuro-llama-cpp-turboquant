# SESSION_PLAN: UMA Zero-Copy für Mars APU (M2)

**Erstellt:** 2026-07-11
**Typ:** Solo-Session (Cherry-Pick + Vulkan UMA Optimization)
**Status:** ☐ offen
**Meilenstein:** M2 — Vulkan-Offensive
**ROADMAP-Item:** #10 — UMA Zero-Copy für Mars APU (PR #22462)

---

## Session-Ziel

PR #22462 (Vulkan UMA optimization: true zero-copy für Unified Memory Architectures, DMA-BUF für pinned shared memory, cache flushing) cherry-picken und auf Mars (RDNA3 APU, 30GB shared memory) implementieren. Erwarteter Gain: +112x transfer speed (Host-to-Backend), +82.9% UMA read throughput.

## Akzeptanzkriterien

- [ ] PR #22462 ist cherry-picked und kompiliert auf Mars (Vulkan build)
- [ ] Build ist grün auf Mars
- [ ] Host-to-Backend Transfer-Zeit signifikant reduziert (Ziel: >10x schneller)
- [ ] `llama-bench` auf Mars zeigt TG-Verbesserung (Ziel: >20% bei großem Kontext)
- [ ] Keine Regression bei PP
- [ ] Modell lädt korrekt (keine korrupten Gewichte durch Zero-Copy)
- [ ] Kontextfenster 224k bleibt stabil (kein OOM durch Zero-Copy-Änderung)
- [ ] ROADMAP.md Status für #10 auf ✅ gesetzt

## Verifikations-Strategie

| Metrik | Vorher | Nachher | Test-Befehl |
|--------|--------|---------|-------------|
| Modell-Ladezeit | Baseline messen | Mit Zero-Copy | `time llama-cli -m <qat-26b> -p 1 -n 1` auf Mars |
| TG t/s (224k Kontext) | 25 t/s | Mit Zero-Copy | `llama-bench -m <qat-26b> -p 0 -n 128 -c 229376` auf Mars |
| TG t/s (8k Kontext) | Baseline | Mit Zero-Copy | `llama-bench -m <qat-26b> -p 0 -n 128 -c 8192` auf Mars |
| PP t/s | Baseline | Mit Zero-Copy | `llama-bench -m <qat-26b> -p 512 -n 0` auf Mars |
| Speicherverbrauch | `radeontop` | Mit Zero-Copy | `radeontop -d 1 -c 5` während Inference |

## Schritte

1. ☐ **PR-Verifikation:** `gh pr view 22462 --repo ggerganov/llama.cpp --json title,state,mergedAt`
2. ☐ **PR-Inhalt analysieren:** Welche Dateien? ggml-vulkan.cpp? Buffer-Allocation-Logik? DMA-BUF?
3. ☐ **Baseline-Benchmark auf Mars:** Ladezeit, TG@224k, TG@8k, PP, Speicherverbrauch
4. ☐ **Cherry-Pick:** `git cherry-pick <commit-hash>` — bei Konflikten: Buffer-Allocation manuell mergen
5. ☐ **Build auf Mars:** `cmake --build build -j$(nproc)` mit `-DLLAMA_VULKAN=ON`
6. ☐ **Funktionstest:** Modell lädt korrekt? `llama-cli -m <qat-26b> -p "Test" -n 10` — Output plausibel?
7. ☐ **Benchmark auf Mars:** Ladezeit, TG@224k, TG@8k, PP, Speicherverbrauch
8. ☐ **Vergleich:** Vorher/Nachher-Tabelle
9. ☐ **Stabilitätstest:** 224k Kontext laden, 5 Minuten Inference, kein OOM/Crash
10. ☐ **ROADMAP.md aktualisieren:** #10 auf ✅ oder ❌
11. ☐ **CHANGELOG.md + Commit + Push**
12. ☐ **TTT-Eintrag**

## Blockaden / User-Eingriffe

| Blockade | Wahrscheinlichkeit | Loesung |
|----------|-------------------|---------|
| PR #22462 ist SYCL-spezifisch (nicht Vulkan) | Mittel | webfetch prüfen — wenn SYCL-only: Vulkan-Äquivalent in Issue #22930 suchen |
| DMA-BUF nicht verfügbar auf Mars LXC | Hoch | Mars läuft in Proxmox LXC (phobos) — DMA-BUF braucht evtl. Host-Durchreichung. Wenn nicht verfügbar: Alternative ohne DMA-BUF (nur mmap-Optimization) |
| Merge-Konflikt mit Fork Buffer-Logik | Mittel | ggml-vulkan.cpp hat Fork-spezifische Änderungen (turbo3/turbo4, coopmat2) — manuell mergen |
| Zero-Copy verursacht korrupte Gewichte | Mittel | Funktionstest (Schritt 6) muss bestanden werden. Wenn korrupt: revertieren. |
| OOM bei 224k Kontext | Niedrig | Zero-Copy sollte VRAM sparen, nicht mehr verbrauchen. Wenn OOM: Memory-Logik prüfen. |

## Defaults

- **DMA-BUF nicht verfügbar:** Nur mmap-Optimization portieren (ohne DMA-BUF). Teil-Gain ist besser als kein Gain.
- **Korrupte Gewichte:** Sofort revertieren. Zero-Copy muss korrekte Ergebnisse liefern.
- **Regression bei PP:** Wenn PP regressiert >5%: revertieren. PP ist wichtig für User-Experience.
- **LXC-Limitierung:** Mars ist ein LXC-Container. Wenn Vulkan-Erweiterungen nicht durchgereicht werden: auf Host-Level (Proxmox) prüfen oder als ❌ markieren.

## Recherche-Fallbacks

| Problem | Reaktion |
|---------|----------|
| DMA-BUF in LXC unklar | web_search "Vulkan DMA-BUF LXC Proxmox passthrough" |
| UMA detection unklar | webfetch auf PR #22462 — wie wird UMA erkannt? |
| Buffer-Allocation Konflikt | `git diff` auf ggml-vulkan.cpp, manuelle Integration |
| Zero-Copy Syntax unklar | web_search "Vulkan zero-copy unified memory mmap device" |

## System-Zugriff

| System | Zweck | SSH |
|--------|-------|-----|
| Hydra (lokal) | Cherry-Pick, git-Operationen | — |
| Mars | Vulkan Build, Benchmark, UMA Test | `ssh mars` |
