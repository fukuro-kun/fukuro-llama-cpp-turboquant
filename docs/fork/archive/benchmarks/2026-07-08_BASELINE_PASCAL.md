# Benchmark-Ergebnisse Pascal-Host (2026-07-08): thecodacus MoE-Optimierungen

## Hardware
- GTX 1070 (Pascal, compute 6.1), 8GB VRAM
- Build: edd42e60f (9171) mit thecodacus Patches
- Modell: Gemma 4 26B-A4B IQ4_NL (14GB, 25.23B params)

## Hauptergebnis: 26B-A4B mit -ot exps=CPU

| Batch | Baseline | +Pinning | +Pinning+Prefetch | Total Boost |
|-------|----------|----------|-------------------|-------------|
| pp512 | 293.27 | 349.44 | 505.51 | +72.3% |
| pp1024 | 273.70 | 336.92 | 503.05 | +83.8% |
| pp2048 | 251.58 | 309.43 | 489.72 | +94.7% |
| pp4096 | 225.75 | 278.82 | 465.49 | +106.2% |
| tg128 | 12.69 | 13.74 | 16.57 | +30.5% |

### Vergleiche mit thecodacus (RTX 3060, Qwen3.6-35B-A3B)
- thecodacus: 1143 → 1880 t/s (+64.5%) bei pp2048
- wir: 251.58 → 489.72 t/s (+94.7%) bei pp2048
- Erklärung: Pascal (PCIe Gen3, ältere VRAM) profitiert mehr vom Overlap

### Env-Vars
- `GGML_CUDA_REGISTER_HOST=1` — Memory Pinning (cudaHostRegister)
- `GGML_SCHED_PREFETCH_EXPERTS=1` — Async Expert Prefetch (3 slots default)

## E4B Ergebnisse (Dense MoE, voller GPU-Offload)

| Modell | Quant | pp512 | tg128 | Config |
|--------|-------|-------|-------|--------|
| E4B | Q4_K_M | 811.50 | 41.23 | -ngl 999 -fa on |
| E4B | IQ4_XS | 871.63 | 45.61 | -ngl 999 -fa on |

## Beobachtungen
- Pinning allein: +19-23% pp (cudaHostRegister eliminiert pageable-copy Verlust)
- Prefetch allein (mit Pinning): zusätzlicher +43-67% pp (Overlap Upload mit Compute)
- tg128 profitiert auch: +30.5% (Expert-Uploads bei Decode ebenfalls beschleunigt)
- turbo3/4 KV-Cache ist auf Pascal bei MoE-Offloading kontraproduktiv (-6% pp)
- E4B IQ4_XS ist ~7% schneller als Q4_K_M (kleineres Modell, gleiche Architektur)
