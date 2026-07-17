# E4B Vulkan Benchmark auf Venus (AMD Vega, GCN/Renoir)

**Datum:** 2026-07-16
**Build:** `70a727dc5` (master)
**Modell:** `gemma-4-E4B-it-UD-Q4_K_XL.gguf` (4.76 GiB, 7.52B params, ~4B aktiv)

## Hardware

| Komponente | Wert |
|------------|------|
| GPU | AMD Radeon Graphics (RADV RENOIR) — Vega iGPU, GCN-Architektur |
| Treiber | RADV (Mesa 25.2.8), Vulkan API 1.4.318 |
| VRAM/GTT | 10.53 GiB GTT + 21.07 GiB host-visible (unified memory) |
| RAM | 62 GB DDR4 |
| CPU | AMD Ryzen 5 5560U (6C/12T, Zen 3) |
| GPU-Features | fp16: yes, bf16: no, int_dot: no, matrix_cores: none, push_desc: yes, subgroup: 64, UMA: yes |

## llama-bench Ergebnisse

### Baseline (f16 KV, kein FlashAttention)

| Test | t/s | Bemerkung |
|------|-----|-----------|
| pp512 | 92.62 ± 0.02 | — |
| pp4096 | 69.60 ± 9.28 | Hohe Varianz (GCN scheduling) |
| tg128 | 7.41 ± 0.01 | — |

### Optimiert (turbo3 K + turbo4 V, FlashAttention on, Prefetch)

| Test | t/s | vs Baseline | Bemerkung |
|------|-----|-------------|-----------|
| pp512 | 60.57 ± 1.73 | **-35%** | turbo3/4 FA ist scalar fallback auf GCN (kein turbo3 FA shader) |
| pp4096 | 45.43 ± 0.05 | **-35%** | — |
| pp16384 | 26.06 ± 0.02 | — | — |
| tg128 | 7.18 ± 0.22 | **-3%** | Rauschen — tg gleichauf |

### Beobachtung: turbo3/4 ist auf GCN **langsamer** als f16 bei PP

Im Gegensatz zu Mars (RDNA3, Phoenix) wo turbo3/4 +31% schneller als f16 bei pp@96k ist, ist turbo3/4 auf Venus (GCN, Renoir) **35% langsamer** bei PP. Grund: turbo3 FA ist auf GCN deaktiviert (glslc infinite optimizer loop), turbo4 FA aktiv aber scalar fallback. Die Dequant-Overhead von turbo3/turbo4 überwiegt den KV-Cache-Größenvorteil auf der kleinen Vega iGPU.

**Empfehlung für Venus:** f16 KV ohne FA für PP-Lastige Workloads. turbo3/4 nur wenn Kontext-Größe der limitierende Faktor ist (f16 braucht mehr VRAM).

## llama-server Live-Test (64k Kontext, turbo3/4, FA on)

**Konfiguration:** `-c 65536 -ngl 99 -ctk turbo3 -ctv turbo4 -fa on`, Prefetch aktiv

| Test | Prompt tokens | Gen tokens | pp t/s | tg t/s | Total |
|------|---------------|------------|--------|--------|-------|
| Short Prompt | 37 | 256 | 19.8 | 8.4 | 32.5s |
| Mittlerer Prompt | 1203 | 512 | 60.6 | 6.2 | 102.1s |
| Großer Prompt | 5002 | 2 | 46.3 | 10.9 | 108.3s |

### tg-Speed (Token Generation)

**E4B auf Venus: ~6-8.4 t/s tg** mit turbo3/4 KV-Cache und 64k Kontext.

Vergleich zur alten Ollama CPU-only Messung (Mai 2026): ~9-10 t/s. Die Vulkan-Zahlen sind leicht niedriger, was an turbo3/4's Dequant-Overhead auf GCN liegt (siehe oben). Mit f16 KV wäre tg vermutlich ~8-9 t/s.

### pp-Speed (Prompt Processing)

pp variiert stark: 19.8 t/s (37 tokens) bis 60.6 t/s (1203 tokens). Bei sehr kurzen Prompts dominiert Setup-Overhead, bei mittleren ist die Vega effizienter. Bei 5002 tokens sinkt es auf 46.3 t/s (KV-Cache wächst, turbo3/4 Dequant kostet).

## Vergleich: E4B vs 26B-A4B im LAN

| Host | GPU | Modell | tg t/s | pp t/s (512) | Kontext |
|------|-----|--------|--------|-------------|---------|
| **Venus** | Vega iGPU (GCN) | E4B Q4_K_XL (4.8 GB) | **6-8** | 61-93 | 64k |
| **Mars/phobos** | Radeon 760M (RDNA3) | 26B-A4B Q4_K_XL (14.2 GB) | **26-27** | 37-40 | 256k |
| **Styx** | GTX 1070 (Pascal) | 26B-A4B Q4_K_XL (14.2 GB) | **27** | 35 | 224k |

Trotz des viel kleineren Modells (4.8 GB vs 14.2 GB) ist Venus **3-4x langsamer** bei tg als Mars/Styx. Grund: Vega iGPU (GCN) hat deutlich weniger Compute-Throughput als RDNA3 oder Pascal, und turbo3/4 Dequant ist auf GCN teurer.

## Fazit

- **E4B auf Venus: ~6-8 t/s tg, ~46-61 t/s pp** (mit turbo3/4, 64k Kontext)
- **64k Kontext funktioniert** — kein OOM, 62 GB RAM sind mehr als genug
- **f16 KV wäre auf Venus schneller als turbo3/4** (GCN-Dequant-Overhead)
- **Venus ist als Inferenz-Host nicht performant** — Vega iGPU ist zu schwach für Echtzeit-Chat. Mars (RDNA3) ist 3-4x schneller trotz 3x größeren Modells.
- **WoWLAN-Service gefixt** (Race Condition phy0, ExecStartPre retry + Restart) und `venus-suspend` Wrapper-Skript installiert
