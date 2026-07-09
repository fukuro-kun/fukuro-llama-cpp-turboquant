# Vulkan KV-Cache Benchmark: turbo3/3 vs turbo3/4 vs turbo4/4 (26B-A4B)

**Datum:** 2026-07-09
**System:** AMD-RDNA3 (Ryzen 5 7640HS, Radeon 760M RDNA3, 30GB RAM, Vulkan 1.4.309, Mesa 25.0.7)
**Modell:** Gemma-4 26B-A4B IQ4_NL (14.7GB, 4B aktive Parameter/Token, MoE mit 30 Experten)
**Backend:** Vulkan, -ngl 99, FlashAttention on, 2 Reps

## Hypothese

K=turbo4 (mit FlashAttention) könnte schneller sein als K=turbo3 (ohne FA, scalar fallback) trotz geringerer Kompression, da turbo3 FA deaktiviert ist (glslc hängt in infinite optimizer loop bei SPIR-V Generation).

## Ergebnisse

### Prompt Processing (t/s)

| KV-Cache | pp512 | pp2048 | pp4096 | pp8192 | Kompression |
|----------|-------|--------|--------|--------|-------------|
| **f16/f16** | **201.73** | **199.46** | **193.30** | **183.18** | 1.0x (Baseline) |
| turbo3/turbo4 | 200.60 | 183.01 | 170.56 | 152.90 | ~4.4x (mixed) |
| **turbo3/turbo3** | **200.14** | **184.81** | **171.25** | **152.66** | 5.1x |
| turbo4/turbo4 | 196.70 | 175.87 | 160.67 | 140.45 | 3.8x |

### Token Generation (t/s, tg64)

| KV-Cache | tg64@pp512 | tg64@pp2048 | tg64@pp4096 | tg64@pp8192 |
|----------|------------|-------------|-------------|-------------|
| f16/f16 | 22.31 | 22.29 | 22.17 | 22.14 |
| turbo3/turbo3 | 22.31 | 22.17 | 22.00 | 21.99 |
| turbo3/turbo4 | 22.12 | 21.88 | 21.98 | 21.96 |
| turbo4/turbo4 | 21.94 | 21.86 | 21.57 | 21.70 |

### Relative Performance (vs f16)

| KV-Cache | pp512 | pp2048 | pp4096 | pp8192 |
|----------|-------|--------|--------|--------|
| turbo3/turbo3 | -0.8% | -7.3% | -11.4% | -16.7% |
| turbo3/turbo4 | -0.6% | -8.2% | -11.8% | -16.5% |
| turbo4/turbo4 | -2.5% | -11.8% | -16.9% | -23.3% |

### turbo3/3 vs turbo4/4 (Direktvergleich)

| Prompt | turbo3/3 | turbo4/4 | Diff | FA-Effekt |
|--------|----------|----------|------|-----------|
| 512 | 200.14 | 196.70 | -1.7% | turbo4 FA reicht nicht |
| 2048 | 184.81 | 175.87 | -4.8% | turbo4 FA reicht nicht |
| 4096 | 171.25 | 160.67 | -6.2% | turbo4 FA reicht nicht |
| 8192 | 152.66 | 140.45 | -8.0% | turbo4 FA reicht nicht |

### turbo3/3 vs turbo3/4 (Mixed)

| Prompt | turbo3/3 | turbo3/4 | Diff |
|--------|----------|----------|------|
| 512 | 200.14 | 200.60 | +0.2% |
| 2048 | 184.81 | 183.01 | -1.0% |
| 4096 | 171.25 | 170.56 | -0.4% |
| 8192 | 152.66 | 152.90 | +0.2% |

## Fazit

**Hypothese WIDERLEGT.** K=turbo4 (mit FA) ist NICHT schneller als K=turbo3 (ohne FA). turbo3/3 und turbo3/4 sind praktisch gleichauf und beide konsistent schneller als turbo4/4.

### Warum turbo3 trotz fehlendem FA schneller ist:

1. **KV-Cache-Größe:** turbo3 hat 3.125 bit/Element (5.1x Kompression), turbo4 hat 4.25 bit/Element (3.8x). Bei Prompt Processing muss der gesamte KV-Cache dequantisiert werden — weniger Daten = schneller.
2. **Dequant-Overhead dominiert:** Bei Kontexten bis 8192 ist die Dequantisierung der Flaschenhals, nicht die Attention-Berechnung. Der FA-Vorteil von turbo4 (O(n²) → O(n)) zeigt sich erst bei viel größeren Kontexten.
3. **Generation unbeeinflusst:** tg64 ist nahezu identisch über alle Formate (~21.9-22.3 t/s). Generation ist Compute-bound, nicht Memory-bound.

### turbo3/4 (Mixed) als valide Alternative

turbo3/4 (K=turbo3, V=turbo4) ist bei allen Kontexten praktisch gleich schnell wie turbo3/3 (±1%). Der Vorteil: V=turbo4 hat höhere Präzision (4.25 bit vs 3.125 bit) bei gleichem Speed. Für Anwendungen die V-Cache-Präzision brauchen ist turbo3/4 eine valide Wahl.

### Empfehlung bestätigt

Die aktuelle Empfehlung **K=turbo3, V=turbo3** für AMD-RDNA3 (Vulkan) wird bestätigt. turbo3/3 bietet die beste Performance bei maximaler Kompression (5.1x). turbo3/4 ist eine valide Alternative wenn V-Cache-Präzision wichtiger ist als Kompression.

## Technische Hinweise

- **turbo3 FA ist DEAKTIVIERT:** SPIR-V Generation für `flash_attn.comp` mit `DATA_A_TURBO3_0` ist auskommentiert in `vulkan-shaders-gen.cpp` (Zeilen 692-704). Grund: `glslc` hängt in infinite optimizer loop. Laufzeit-Fallback: scalar Attention-Pfad.
- **turbo4 FA ist AKTIV:** `flash_attn_cm1.comp` unterstützt turbo4 (Zeile 302).
- **26B-A4B auf AMD-RDNA3:** 14.7GB Modell, 4B aktive Parameter (MoE). Funktioniert mit -ngl 99 wenn RAM/GTT clean ist (keine parallelen Prozesse). ~22 t/s generation — identisch mit FINALE EMPFEHLUNG.
- **OOM-Vermeidung:** Vor jedem Benchmark `killall -9 llama-bench; sleep 8-10` um GTT freizugeben. Auf UMA-Systemen mit Proxmox belegen andere Services ~2GB RAM.
- **nodes_per_submit=10:** UMA-Auto-Detect fix ist im Build enthalten, verhindert GPU-Hang bei großen Batches.

## Benchmark-Script

`scripts/bench-vulkan-kv-cache.sh` — Testet turbo3/3, turbo3/4, turbo4/4, f16/f16 bei pp512-8192 mit 26B-A4B. GTT-Cleanup zwischen Tests.
