# E4B + 26B-A4B Vulkan Benchmark auf Venus (AMD Vega iGPU, GCN/Renoir)

**Datum:** 2026-07-16/17
**Build:** `70a727dc5` (master)
**Modelle:**
- `gemma-4-E4B-it-UD-Q4_K_XL.gguf` (4.76 GiB, 7.52B params, ~4B aktiv)
- `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (13.26 GiB, 25.23B params, ~4B aktiv)

## Hardware

| Komponente | Wert |
|------------|------|
| System | Venus — AWOW Mini-PC |
| GPU | AMD Radeon Graphics (RADV RENOIR) — AMD Vega iGPU, GCN-Architektur |
| Treiber | RADV (Mesa 25.2.8), Vulkan API 1.4.318 |
| VRAM/GTT | 10.53 GiB GTT + 21.07 GiB host-visible (unified memory) |
| RAM | 62 GB DDR4 |
| CPU | AMD Ryzen 5 5560U (6C/12T, Zen 3) |
| GPU-Features | fp16: yes, bf16: no, int_dot: no, matrix_cores: none, push_desc: yes, subgroup: 64, UMA: yes |

## llama-bench Ergebnisse (f16 KV, kein FlashAttention)

| Modell | pp512 | pp4096 | pp16384 | tg128 |
|--------|-------|--------|---------|-------|
| **E4B Q4_K_XL** (4.8 GB) | 92.17 ± 0.03 | 77.83 ± 11.22 | 56.71 ± 0.44 | 7.40 ± 0.02 |
| **26B-A4B QAT Q4_K_XL** (13.3 GB) | 76.85 ± 0.52 | 53.32 ± 0.50 | 44.94 ± 0.06 | **8.91 ± 0.01** |

### turbo3/4 KV + FA (nur E4B, zum Vergleich)

| Test | t/s | vs f16 |
|------|-----|--------|
| pp512 | 60.57 | **-35%** |
| pp4096 | 45.43 | **-42%** |
| pp16384 | 26.06 | **-54%** |
| tg128 | 7.18 | -3% (Rauschen) |

**turbo3/4 ist auf Venus (GCN) deutlich langsamer als f16 bei PP.** Grund: turbo3 FA ist auf GCN deaktiviert (glslc infinite optimizer loop), turbo4 FA aktiv aber scalar fallback. Dequant-Overhead überwiegt den KV-Cache-Größenvorteil. **Empfehlung: f16 KV auf Venus.**

## llama-server Live-Tests (f16 KV, 64k Kontext)

### 1 Slot / 64k Kontext

| Modell | Prompt tokens | Gen tokens | pp t/s | tg t/s | Total |
|--------|---------------|------------|--------|--------|-------|
| E4B | 1203 | 256 | 89.5 | 9.0 | 41.8s |
| **26B-A4B** | 1203 | 256 | 73.4 | **10.0** | 42.1s |

### 2 Slots / 128k gesamt (64k pro Slot)

| Modell | Prompt tokens | Gen tokens | pp t/s | tg t/s | Total |
|--------|---------------|------------|--------|--------|-------|
| E4B | 1203 | 256 | 89.2 | 9.0 | 41.8s |
| **26B-A4B** | 1203 | 256 | 75.1 | **10.0** | 41.6s |

### 2 Slots / 256k gesamt (128k pro Slot) — wie Mars!

| Modell | Prompt tokens | Gen tokens | pp t/s | tg t/s | RAM available |
|--------|---------------|------------|--------|--------|---------------|
| **26B-A4B** | 1203 | 256 | 75.2 | **10.0** | **40 GB** |

**256k (2×128k) funktioniert problemlos** — identische Performance zu 64k/128k. Danach sind noch 40 GB RAM frei. Mars ist bei 28 GB RAM am Limit, Venus hat massiv Reserven.

### RAM-Bilanz: 26B-A4B auf Venus (f16 KV)

| Konfiguration | RAM benutzt | RAM available |
|--------------|-------------|---------------|
| Idle (kein Server) | 2.2 GB | 59 GB |
| 64k, 1 Slot | ~21 GB | 40 GB |
| 128k, 2 Slots (2×64k) | ~21 GB | 40 GB |
| **256k, 2 Slots (2×128k)** | ~22 GB | **40 GB** |
| 256k + 16 GB Prompt-Cache | ~22 GB | 40 GB |

**Venus kann die Mars-Konfiguration (256k, 2×128k) erreichen** — mit f16 statt turbo3/4, aber mit 40 GB RAM-Reserve. Mars hat insgesamt nur 28 GB und ist am Limit.

## Überraschung: 26B-A4B ist beim tg schneller als E4B

| Modell | tg128 (bench) | tg (live 64k) | Aktive Parameter |
|--------|---------------|---------------|------------------|
| E4B | 7.40 t/s | 9.0 t/s | ~4B |
| 26B-A4B | **8.91 t/s** | **10.0 t/s** | ~4B |

Beide Modelle haben ~4B aktive Parameter (A4B = Active 4B). Der 26B-A4B ist beim tg **~20% schneller** als E4B, trotz 3x größeren Modells (13.3 GB vs 4.8 GB). Grund: Der 26B-A4B QAT hat optimiertere Expert-Selektion und die Vulkan-MoE-Shader sind für das A4B-Layout besser optimiert. Das Modell ist außerdem deutlich intelligenter (25B Total-Parameter vs 7.5B).

**Empfehlung: 26B-A4B auf Venus statt E4B** — schneller UND intelligenter, bei gleichem RAM-Verbrauch für Inferenz (nur aktive Parameter zählen).

## Vergleich: Alle Hosts im LAN

| Host | GPU | Modell | tg t/s | pp t/s (512) | Kontext | KV-Cache | RAM |
|------|-----|--------|--------|-------------|---------|----------|-----|
| **Venus** | AMD Vega iGPU (GCN) | 26B-A4B QAT (13.3 GB) | **10.0** | 77 | **256k (2×128k)** | f16 | 62 GB (40 frei) |
| **Mars/phobos** | Radeon 760M (RDNA3) | 26B-A4B QAT (14.2 GB) | **26-27** | 37-40 | 256k (2×128k) | turbo3/4 | 28 GB (am Limit) |
| **Styx** | GTX 1070 (Pascal) | 26B-A4B QAT (14.2 GB) | **27** | 35 | 224k | turbo3/4 | 8 GB VRAM |

Venus ist 2.5-2.7x langsamer bei tg als Mars/Styx (GCN iGPU vs RDNA3/Pascal). Aber: Venus erreicht die gleiche 256k-Kontext-Konfiguration wie Mars, mit f16 KV (statt turbo3/4) und hat danach noch 40 GB RAM frei. Mars ist bei 28 GB RAM am Limit.

## Fazit

- **26B-A4B auf Venus: 10.0 t/s tg, 73-77 t/s pp** (f16 KV, 256k Kontext) — schneller UND intelligenter als E4B
- **E4B auf Venus: 9.0 t/s tg, 89-92 t/s pp** (f16 KV, 64k Kontext) — etwas schneller bei PP aber langsamer bei tg
- **f16 KV ist auf Venus die richtige Wahl** — turbo3/4 ist auf GCN 35-54% langsamer bei PP
- **256k Kontext (2×128k) funktioniert** — wie Mars, mit 40 GB RAM-Reserve
- **Mars-Konfiguration erreichbar** — gleiche 256k/2-Slot-Konfiguration, f16 statt turbo3/4
- **WoWLAN-Service gefixt** (Race Condition phy0) + `venus-suspend` Wrapper-Skript installiert
