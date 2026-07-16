# EA Phase 3 GPU-Benchmark — Styx (GTX 1070)

**Datum:** 2026-07-15
**ROADMAP-Item:** #19
**Status:** ✅ Benchmark abgeschlossen

## Setup

| Parameter | Wert |
|-----------|------|
| System | Styx (GTX 1070, 8GB VRAM, CUDA) |
| Modell | gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf (13.26 GiB) |
| KV-Cache | turbo3 (K) / turbo4 (V) |
| -ngl | 999 (full offload) |
| --n-cpu-moe | 20 |
| FlashAttention | on |
| Threads | 4 |
| Reps | 1 |
| EA Ratio | 0.3 (30% KV-Paare geprunt) |
| Context-Matrix | 512, 4096, 16384, 32768 |

## Ergebnisse

| Test | Baseline (t/s) | EA ratio=0.3 (t/s) | Delta |
|------|---------------:|--------------------:|------:|
| pp512 | 371.31 | 341.74 | -7.9% |
| tg128 | 23.83 | 19.71 | -17.3% |
| pp512+tg128 | 91.68 | 79.12 | -13.7% |
| pp4096+tg128 | 219.66 | 201.07 | -8.5% |
| pp16384+tg128 | 217.33 | 212.73 | -2.1% |
| pp32768+tg128 | 179.46 | 177.80 | -0.9% |

## Analyse

### Overhead-Muster

EA verursacht einen CPU-Overhead durch den Scoring-Pfad (Q-Capture, RoPE-Matrix, Score-Berechnung, Pruning). Dieser Overhead ist bei kleinen Kontexten dominant (-8 bis -17%), wird aber bei großen Kontexten durch die KV-Cache-Reduktion kompensiert (-1 bis -2% bei 32k).

### Warum kein Speedup bei großen Kontexten?

Erwartet war, dass EA bei großen Kontexten einen Speedup bringt (weniger KV-Paare = weniger Attention-Computation). Auf Styx ist das nicht der Fall, weil:

1. **MoE-Offload ist das Bottleneck:** Bei 26B-A4B mit 20 CPU-MoE-Experten ist der Expert-Transfer (CPU→GPU) der dominierende Faktor, nicht die Attention-Computation.
2. **CPU-only EA-Scoring:** Der EA-Scoring-Pfad läuft single-threaded auf der CPU. Bei ratio=0.3 werden 30% der KV-Paare geprunt, aber der Scoring-Overhead frisst den Gewinn auf.
3. **Pascal-Architektur:** GTX 1070 hat keine FlashAttention-Hardware-Beschleunigung. Die Attention läuft auf dem scalar CUDA-Pfad, der weniger von KV-Reduktion profitiert als tensor-core-basierte Attention.

### Was würde helfen?

1. **GPU-Offload des EA-Scoring:** Der Scoring-Pfad (E(A)=exp(K@μ'/√d)) ist eine Matrix-Multiplikation die auf GPU deutlich schneller wäre.
2. **Größere Kontexte (>64k):** Bei 64k+ Kontext würde die KV-Cache-Reduktion deutlicher ins Gewicht fallen — aber auf Pascal sind 64k+ Prefills zu langsam um zu testen.
3. **Andere Systeme:** Auf Uranus (2x RTX 4060 Ti, tensor cores) oder Mars (Vulkan, große RAM) könnte EA einen echten Speedup zeigen.

## Fazit

**EA Phase 3 funktioniert korrekt auf GPU** (keine Crashes, korrekte Output-Qualität). Der Performance-Overhead ist bei kleinen Kontexten spürbar (-8 bis -17%), bei großen Kontexten neutral (-1%). Für Styx als Produktiv-System ist EA nicht empfehlenswert (MoE-Offload-dominiert). Für Systeme mit großen Kontexten und GPU-beschleunigter Attention könnte EA einen Speedup bringen.

**Nächste Schritte:**
- GPU-Offload des EA-Scoring-Pfads (CUDA/Vulkan Backend)
- Benchmark auf Uranus (tensor cores) oder Mars (Vulkan, große RAM)
- Höhere ratio (0.5, 0.7) testen um KV-Reduktion zu maximieren

## Referenzen

- Design-Doc: `docs/fork/2026-07-15_EXPECTED_ATTENTION_DESIGN.md`
- Benchmark-Skript: `scripts/bench-ea-phase3.sh`
- ROADMAP-Item: #19
- Paper: arXiv:2510.00636
