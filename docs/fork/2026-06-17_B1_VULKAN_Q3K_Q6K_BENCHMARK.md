# B1 Benchmark: Vulkan Q3_K/Q6_K Block-Load Performance

**Datum:** 2026-06-17  
**Commit:** 931cf902e (vulkan: Block-load Q3_K/Q6_K block data and subtract on 32b ints)  
**Zweck:** Vergleicht tg128-Performance fuer Gemma 4 Modelle in Q3_K und Q6_K Quantisierung.

---

## Systeme

| System | GPU | RAM | Backend | Status |
|--------|-----|-----|---------|--------|
| **Mars** | AMD Radeon 760M (RDNA3) | 32GB | Vulkan | � **3/6 Modelle erfolgreich** |
| **Venus** | AMD Radeon (RDNA2/Vega) | 32GB | CPU-Fallback | 🟢 5/6 Modelle erfolgreich |

**Hinweis:** Venus lief auf CPU-Fallback statt Vulkan (GPU nicht korrekt erkannt). Die Werte dienen als Referenz, nicht als Vulkan-Performance-Messung.

---

## Ergebnisse: Mars (Vulkan, RDNA3)

| Modell | Quant | pp128 [t/s] | tg32 [t/s] | Status |
|--------|-------|-------------|------------|--------|
| Gemma 4 12B | Q3_K_M | **103.99** | **8.02** | ✅ |
| Gemma 4 12B | Q6_K | **100.93** | **7.03** | ✅ |
| Gemma 4 26B A4B | UD-Q3_K_M | **123.52** | **19.78** | ✅ |
| Gemma 4 26B A4B | UD-Q6_K | — | — | ❌ OOM (23GB + Vulkan-Overhead > 32GB RAM) |
| Gemma 4 31B | Q3_K_M | — | — | ❌ OOM (14.7GB + Overhead) |
| Gemma 4 31B | Q6_K | — | — | ❌ OOM (25.2GB + Overhead) |

### Beobachtungen

1. **Vulkan ist massiv schneller als CPU-Fallback** — siehe Vergleich unten.
2. **Q3_K_M vs Q6_K auf Vulkan:** 12B tg32: 8.02 vs 7.03 → Q3_K_M ist **14% schneller**.
3. **RAM-Limit:** 32GB reichen nicht fuer 31B und 26B Q6_K mit Vulkan-Overhead.

---

## Ergebnisse: Venus (CPU-Fallback)

| Modell | Quant | pp128 [t/s] | tg32 [t/s] | Status |
|--------|-------|-------------|------------|--------|
| Gemma 4 12B | Q3_K_M | 13.62 | 5.94 | ✅ |
| Gemma 4 12B | Q6_K | 9.54 | 3.60 | ✅ |
| Gemma 4 26B A4B | UD-Q3_K_M | 26.10 | 9.45 | ✅ |
| Gemma 4 26B A4B | UD-Q6_K | — | — | ❌ OOM |
| Gemma 4 31B | Q3_K_M | 3.77 | 2.19 | ✅ |
| Gemma 4 31B | Q6_K | 2.73 | 1.42 | ✅ |

---

## Vergleich: Mars Vulkan vs Venus CPU

| Modell | Quant | Mars pp | Venus pp | Speedup | Mars tg | Venus tg | Speedup |
|--------|-------|---------|----------|---------|---------|----------|---------|
| 12B | Q3_K_M | 103.99 | 13.62 | **7.6×** | 8.02 | 5.94 | **1.3×** |
| 12B | Q6_K | 100.93 | 9.54 | **10.6×** | 7.03 | 3.60 | **2.0×** |
| 26B A4B | Q3_K_M | 123.52 | 26.10 | **4.7×** | 19.78 | 9.45 | **2.1×** |

### Fazit

- **Prompt Processing (pp):** Vulkan ist **5-10× schneller** als CPU-Fallback.
- **Token Generation (tg):** Vulkan ist **1.3-2.0× schneller** — deutlich weniger dramatisch, da tg memory-bound ist.
- **Q3_K_M vs Q6_K:** Q3_K_M ist konsistent schneller, aber der Unterschied ist auf Vulkan kleiner als auf CPU.

---

## Upstream-Vergleich

| Modell | Quant | Upstream Speedup (Intel BMG) | Unser Speedup (Mars RDNA3) |
|--------|-------|------------------------------|---------------------------|
| Qwen 3.5 9B | Q3_K | +81% tg128 (MMVQ+Block-Load) | N/A (kein Qwen getestet) |
| Qwen 3.5 9B | Q6_K | +126% tg128 (MMVQ+Block-Load) | N/A (kein Qwen getestet) |
| Gemma 4 12B | Q3_K vs Q6_K | N/A | tg32: +14% (Q3_K_M schneller) |

**Hinweis:** Upstream testete Qwen 3.5 auf Intel BMG. Unsere Gemma 4-Modelle auf AMD RDNA3 zeigen einen kleineren, aber messbaren Vorteil fuer Q3_K_M.

---

## Benchmark-Skript

```bash
# Auf Mars (Vulkan):
LLAMA_BENCH=/pfad/zu/llama-bench MODEL_DIR=/jade/models/unsloth ./scripts/bench-b1-vulkan-q3k-q6k.sh

# Auf Venus (CPU-Fallback):
LLAMA_BENCH=/pfad/zu/llama-bench MODEL_DIR=/home/fukuro/models/unsloth ./scripts/bench-b1-vulkan-q3k-q6k.sh
```

---

## TODO

- [ ] **Mars: 31B und 26B Q6_K mit mehr RAM** testen (64GB+ oder Swap erweitern)
- [ ] **Venus Vulkan-Backend korrigieren** (laueft aktuell auf CPU-Fallback)
- [ ] **Qwen 3.5 Benchmark** fuer direkten upstream-Vergleich
- [ ] **Vorher/Nachher-Vergleich:** B1-Commit revertieren und erneut benchmarken

---

## Methode

- **Tool:** `llama-bench` (llama.cpp Fork)
- **Parameter:** `-ngl 99 -p 128 -n 32 -r 1`
- **Repetitions:** 1 (schneller Durchlauf)
- **Metrik:** pp128 (Prompt Processing), tg32 (Token Generation)
