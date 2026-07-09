# Momentaufnahme — fukuro-llama-cpp-turboquant + InferenzQuelle

**Datum:** 2026-07-09 | **Branch:** `master` | **Build:** 424 (633d6d772)

---

## Die Projekte und ihre Nahtstelle

| Projekt | Rolle | Repo |
|---------|-------|------|
| **fukuro-llama-cpp-turboquant** | Inference Engine (Motor) — C++/CUDA/Vulkan | Codeberg |
| **InferenzQuelle** | Infrastruktur & Steuerung (Auto) — Python/Bash | Codeberg |

**Nahtstelle:** InferenzQuelle steuert den `llama-server` (aus dem llama-cpp-Fork) per HTTP-API. Test-Framework (pytest) läuft gegen die Binary. Modell-Pfade, Draft-Modelle und TurboQuant-Parameter werden über Config-Files geteilt.

---

## Was zuletzt passiert ist

### Pascal-Host: ext4 → btrfs Migration (2026-07-09)

- **Problem:** `ext4_dirty_folio` Kernel-Bug (6.8.0-134) crashhte Pascal-Host bei aktivem Memory Pinning (`GGML_CUDA_REGISTER_HOST=1`)
- **Lösung:** `/data` Partition in-place von ext4 → btrfs migriert (`btrfs-convert`), fstab aktualisiert
- **Ergebnis:** Kernel-Bug eliminiert, Pinning + Async Expert Prefetch reaktiviert
- **Verifikation:** 24.6 t/s, dmesg clean, Reboot-Test bestanden (btrfs + XFS HDD auto-mount)
- **Service:** 26B-A4B Service läuft mit IQ4_NL, Pinning+Prefetch aktiv

### AMD-RDNA3: Vulkan KV-Cache Benchmark (2026-07-09)

- **Fragestellung:** Ist K=turbo4 (mit FlashAttention) schneller als K=turbo3 (ohne FA, scalar fallback)?
- **Modell:** Gemma-4 26B-A4B IQ4_NL (14.7GB, MoE 4B aktiv), Vulkan, -ngl 99, FA on
- **Matrix:** turbo3/3, turbo3/4, turbo4/4, f16/f16 × pp512, pp2048, pp4096, pp8192
- **Ergebnis:** **Hypothese WIDERLEGT.** turbo3/3 und turbo3/4 sind praktisch gleichauf (±1%), beide konsistent schneller als turbo4/4 (-1.7% bis -8.0%). Der Dequant-Overhead von turbo4 (4.25 bit vs 3.125 bit) überwiegt den FA-Vorteil bei Kontexten bis 8192.
- **Performance:** pp512: ~200 t/s, tg64: ~22 t/s (26B-A4B MoE)
- **OOM-Lerneffekt:** 26B-A4B funktioniert auf AMD-RDNA3 wenn RAM/GTT clean ist (`killall -9 llama-bench; sleep 8-10` zwischen Tests). Erster Versuch OOMte weil Memory nicht freigegeben wurde.
- **Doku:** `docs/fork/2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md`, Trilium-Subnote `5DTGKZb95DUO`

### thecodacus MoE-Optimierungen integriert (2026-07-08, Solo-Session)
- **3 Patches portiert** aus `thecodacus/llama.cpp`: Memory Pinning + Async Expert Prefetch + UAF Fix
- **Merge nach master** (Commit a4215b3d6), Codeberg-Push erfolgreich
- **Benchmarks auf Pascal-Host (GTX 1070):** +72-106% pp, +30% tg mit Pinning+Prefetch auf 26B-A4B MoE-Offload
- **MTP + Pinning + Prefetch:** 100% Akzeptanz, 31.93 t/s (+50% vs ohne MTP)
- **Korrektheit verifiziert:** token-identisch mit/ohne Pinning

---

## Wo wir stehen — das große Bild

| Komponente | Status | Notiz |
|------------|--------|-------|
| TurboQuant KV/Weights | ✅ | turbo3/turbo4 funktional auf CUDA + Vulkan |
| Gemma 4 MTP | ✅ | 0%-Bug gefixt, 50-100% Akzeptanz je nach Quant |
| Qwen 3.x NextN | ✅ | Shared-Model-Draft implementiert |
| DiffusionGemma | ✅ | PKV-Cross-Backend-Fix, Chat-Template integriert |
| Vulkan-WHT | ✅ | Cherry-picked, +17% pp bei Gemma 4 12B |
| CUDA Fast WHT | ✅ | Cherry-picked, +11% pp512, bereit für Merge |
| thecodacus Pinning+Prefetch | ✅ | +95% pp, +50% tg mit MTP auf MoE-Offload |
| **Pascal-Host 26B-A4B Service** | ✅ | **Läuft** mit btrfs + Pinning + Prefetch, 24.6 t/s |
| **AMD-RDNA3 Vulkan Benchmark** | ✅ | turbo3/3 bestätigt als schnellste KV-Cache-Konfig |
| Vulkan turbo3 FA | ⚠️ | Deaktiviert (glslc bug) — scalar fallback, trotzdem schneller als turbo4/4 |

---

## Offene Aufgaben

### [D] Devin — Hochpriorisiert
1. **CUDA Fast WHT merge** — `feature/cuda-fast-wht` bereit für Merge nach master
2. **Qwen 3.6 MoE Vergleichstest** — Falls Modell verfügbar, Benchmark gegen thecodacus

### [F] Fukuro — Mittelpriorisiert
3. **InferenzQuelle uncommittete Änderungen** — MTP_INDEX, STATUS, ARCHITECTURE, benchmark.py — reviewen und committen
4. **turbo3 FA auf Vulkan aktivieren** — glslc infinite optimizer loop fixen (langfristig)

---

## Nächste Schritte

1. **CUDA Fast WHT mergen** — Branch ist bereit, +11% pp512
2. **InferenzQuelle Änderungen committen** — Uncommittete MTP/Benchmark-Änderungen reviewen
3. **26B-A4B als Produktiv-Service evaluieren** — Pascal-Host läuft bereits, AMD-RDNA3 als Backup-Node denkbar
