# Vulkan Perf-RCA Baseline — AMD-System (2026-06-21)

**System:** AMD-System — AMD 760M (RADV PHOENIX, RDNA3, UMA)
**Modell:** Gemma 4 26B IQ4_NL (13.67 GiB)
**Branch:** `feature/vulkan-perf-rca` (von master `39282c6a4`)
**Mesa:** 25.0.7-2
**Vulkan:** 1.4.305

---

## 1. Backend-Op-Tests (Korrektheit)

| Test | Ergebnis |
|------|----------|
| SET_ROWS_TURBO3 | ✅ 2/2 backends passed |
| FLASH_ATTN_EXT turbo3 (alle head_dims) | ✅ 2/2 backends passed |
| FLASH_ATTN_EXT turbo3 hsk=256 (Gemma) | ✅ 2/2 backends passed |

**Fazit:** turbo3+FA ist auf AMD-System (coopmat1-Pfad) korrekt — wie auf AMD-System-2 (scalar-Pfad).

---

## 2. Smoke-Test (llama-cli)

```
Prompt: "Die Hauptstadt von Frankreich ist"
Output: Thinking-Phase erkannte korrekt Deutsch → "The capital of France is Paris"
Prompt: 44.4 t/s | Generation: 22.1 t/s
EXIT=0
```

**Memory-Breakdown (turbo3/turbo3 + FA, ctx=32k):**
```
Vulkan0: 27648 MB total
  = 11301 MB free
  + 14703 MB (14001 modell + 183 kontext + 517 compute)
  + 1643 MB unaccounted
Host: 655 MB (577 modell + 0 kontext + 78 compute)
```

**Fazit:** turbo3 KV-Cache = 183 MB (5.1x Kompression bestätigt). Kohärente Ausgabe.

---

## 3. Benchmark (llama-bench, pp256/tg64)

| Config | pp256 (t/s) | tg64 (t/s) |
|--------|-------------|------------|
| turbo3/turbo3 + FA | 125.76 ± 11.48 | 21.52 ± 0.05 |
| f16/f16 + FA | 160.43 ± 3.92 | 22.27 ± 0.08 |
| f16/f16 ohne FA | 126.13 ± 10.86 | 21.08 ± 0.02 |

**Beobachtungen:**
- turbo3+FA pp256 ist langsamer als f16+FA (126 vs 160 t/s) — turbo3 Dequant-Overhead
- turbo3+FA tg64 ist minimal langsamer als f16+FA (21.5 vs 22.3 t/s) — vernachlässigbar
- f16 ohne FA ist ähnlich zu turbo3+FA bei pp256 — FA-Overhead bei kleinem Kontext
- tg64 ist bei allen Konfigurationen ~21-22 t/s — TG nicht betroffen bei kleinem Kontext

---

## 4. PP-Scaling (clean pipeline cache)

| pp | t/s | Status |
|----|-----|--------|
| 512 | 204.75 ± 2.98 | ✅ |
| 4096 | (läuft) | |
| 8192 | | |
| 16384 | | |

---

## 5. AMD-System Memory-Types (vulkaninfo)

| Type | Heap | Flags | Beschreibung |
|------|------|-------|-------------|
| 0 | 1 | DEVICE_LOCAL | VRAM-Carveout (1 GiB) |
| 1 | 1 | DEVICE_LOCAL | VRAM (nicht für Buffer) |
| 2 | 0 | HOST_VISIBLE \| HOST_COHERENT | System-RAM (uncached) |
| 3 | 1 | DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT | GTT (Write-Combining) |
| 4 | 1 | DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT | GTT (alt) |
| 5 | 0 | HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED | System-RAM (gecached) |
| 6 | 0 | HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED | System-RAM (alt) |
| 7 | 1 | DEVICE_LOCAL \| DEVICE_COHERENT_AMD \| DEVICE_UNCACHED_AMD | VRAM (coherent) |

**Kein Type hat sowohl DEVICE_LOCAL als auch HOST_CACHED!**

---

## 6. Vergleich: AMD-System vs AMD-System-2

| Eigenschaft | AMD-System | AMD-System-2 |
|-------------|------|-------|
| GPU | Radeon 760M (RDNA3) | Radeon Vega (GCN) |
| RADV | PHOENIX | RENOIR |
| coopmat | ✅ KHR_coopmat | ❌ keine |
| FA-Pfad | coopmat1 | scalar |
| Mesa | 25.0.7-2 | 25.2.8 |
| VRAM | 27648 MB (UMA) | 32360 MB (UMA) |
| pp256 (turbo3+FA) | 125.76 | ~166 (AMD-System-2) |
| tg64 (turbo3+FA) | 21.52 | ~9.8 (AMD-System-2) |

**Fazit:** AMD-System ist bei TG deutlich schneller (22 vs 10 t/s), bei PP etwas langsamer (126 vs 166 t/s). Die coopmat1-FA-Implementation auf AMD-System funktioniert korrekt.

---

## Nächste Schritte

1. PP-Scaling komplettieren (pp4096, pp8192, pp16384)
2. TG-Scaling messen (ctx 180k-188k)
3. Vulkan Commit-Bisect starten (Gruppe A: UMA + Korrektheit)
