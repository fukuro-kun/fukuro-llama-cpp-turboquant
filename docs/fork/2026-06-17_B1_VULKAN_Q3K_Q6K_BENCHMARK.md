# B1 Benchmark: Vulkan Q3_K/Q6_K Block-Load Performance

**Datum:** 2026-06-17  
**Commit:** 931cf902e (vulkan: Block-load Q3_K/Q6_K block data and subtract on 32b ints)  
**Zweck:** Vergleicht tg128-Performance fuer Gemma 4 Modelle in Q3_K und Q6_K Quantisierung.

---

## Systeme

| System | GPU | RAM | Backend | Status |
|--------|-----|-----|---------|--------|
| **Mars** | AMD Radeon 760M (RDNA3) | 32GB | Vulkan | 🔴 Benchmark unterbrochen (System unerreichbar) |
| **Venus** | AMD Radeon (RDNA2/Vega) | 32GB | CPU-Fallback | 🟢 5/6 Modelle erfolgreich |

**Hinweis:** Venus lief auf CPU-Fallback statt Vulkan (GPU nicht korrekt erkannt). Die Werte dienen als Referenz, nicht als Vulkan-Performance-Messung.

---

## Ergebnisse: Venus (CPU-Fallback)

| Modell | Quant | pp128 [t/s] | tg32 [t/s] | Q3→Q6 Δ |
|--------|-------|-------------|------------|---------|
| Gemma 4 12B | Q3_K_M | 13.62 | 5.94 | — |
| Gemma 4 12B | Q6_K | 9.54 | 3.60 | pp: −30%, tg: −39% |
| Gemma 4 26B A4B | UD-Q3_K_M | 26.10 | 9.45 | — |
| Gemma 4 26B A4B | UD-Q6_K | — | — | OOM (23GB Modell, RAM-limitiert) |
| Gemma 4 31B | Q3_K_M | 3.77 | 2.19 | — |
| Gemma 4 31B | Q6_K | 2.73 | 1.42 | pp: −28%, tg: −35% |

### Beobachtungen

1. **Q6_K ist langsamer als Q3_K_M** — erwartet, da Q6_K mehr Speicher und Bandbreite braucht.
2. **26B A4B Q6_K OOM** — 23GB Modell + Overhead > verfuegbarer RAM.
3. **31B Q3_K_M tg32 nur 2.19 t/s** — CPU-Limit deutlich sichtbar bei grossen Modellen.

---

## Erwartete Ergebnisse: Mars (Vulkan, RDNA3)

Basierend auf upstream-Bericht (Intel BMG, Mesa):

| Modell | Quant | Erwarteter Speedup (upstream) |
|--------|-------|------------------------------|
| Qwen 3.5 9B | Q3_K | +57% tg128 (MMVQ) +24% (Block-Load) = +81% |
| Qwen 3.5 9B | Q6_K | +78% tg128 (MMVQ) +48% (Block-Load) = +126% |

**Unsere Gemma 4 Modelle:** Keine direkten upstream-Vergleichswerte, aber der Commit aktiviert denselben Code-Pfad (MMVQ + Block-Load in `mul_mat_vecq_funcs.glsl`).

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

- [ ] **Mars-Benchmark wiederholen** (System war waehrend erstem Durchlauf unerreichbar)
- [ ] **Venus Vulkan-Backend korrigieren** (laueft aktuell auf CPU-Fallback)
- [ ] **26B A4B Q6_K mit mehr RAM** testen (z.B. auf Mars mit 64GB)
- [ ] **Q3_K vs Q6_K Speedup** verifizieren (erwartet: Q3_K deutlich schneller auf Vulkan)

---

## Methode

- **Tool:** `llama-bench` (llama.cpp Fork)
- **Parameter:** `-ngl 99 -p 128 -n 32 -r 1`
- **Repetitions:** 1 (schneller Durchlauf, ggf. mit `-r 3` wiederholen)
- **Metrik:** pp128 (Prompt Processing), tg32 (Token Generation)
