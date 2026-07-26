# SESSION_PLAN: #18 DALI Workload-Aware MoE Offloading (Phase 1)

**Erstellt:** 2026-07-25 (Auto-Fortsetzung nach #71)
**Status:** ❌ NO-GO (2026-07-25). Phase 1 implementiert + benchmarked, Workload-Aware +2.5% (kurz) / -1.1% bis +0.4% (lang) → Rauschbereich.
**ROADMAP-Item:** #18 DALI Workload-Aware MoE Offloading (Tier 3, 2-3 Wochen Phase 1)
**Vorgänger:** #69 Heuristic Benchmark (❌), #71 Set-Associative (❌)

## Session-Ziel

Implementiere **POLICY_WORKLOAD** (DALI-inspired sliding-window workload accumulation) als neue Eviction-Policy in `moe-cache.cu`. Testet die Hypothese: ist windowed frequency (temporal correlation) besser als global frequency (Heuristic) oder recency (LRU)?

## Ergebnis

**❌ NO-GO.** Workload-Aware Policy ist bei längeren, stabilen Benchmarks im Rauschbereich (-1.1% bis +0.4%). Der erste Benchmark (+81%) war ein Artefakt der Cache-Discovery-Phase.

## Benchmark-Ergebnisse

| Config | n | r | Budget | tg | Δ vs LRU |
|--------|---|---|--------|-----|----------|
| LRU | 128 | 3 | 512MB | 2.19 ± 1.04 | — |
| Workload wsize=32 | 128 | 3 | 512MB | 3.97 ± 0.41 | +81% (Artefakt!) |
| LRU | 256 | 5 | 512MB | 4.42 ± 0.20 | — |
| Workload wsize=32 | 256 | 5 | 512MB | 4.53 ± 0.09 | +2.5% |
| Heuristic | 256 | 5 | 512MB | 4.50 ± 0.16 | +1.8% |
| LRU | 512 | 5 | 512MB | 4.45 ± 0.11 | — |
| Workload wsize=32 | 512 | 5 | 512MB | 4.40 ± 0.13 | -1.1% |
| LRU | 512 | 5 | 1024MB | 4.52 ± 0.07 | — |
| Workload wsize=32 | 512 | 5 | 1024MB | 4.54 ± 0.07 | +0.4% (Rauschen) |

## MoE-Cache-Thema erschöpft

Drei Eviction-Policies getestet, alle ❌:
- #69 Heuristic: -3.1% (kurz), +1.8% (lang) → ❌
- #71 Set-Associative: -25% bis -50% → ❌ (Conflict-Misses)
- #18 Workload-Aware: +2.5% (kurz), -1.1% bis +0.4% (lang) → ❌ (Rauschen)

**LRU ist optimal für 128-Expert-Oversubscription (24:1).** PCIe-Transfer dominiert, nicht Eviction-Policy.

## Phase 2/3 Bewertung

- **Phase 2 (Greedy Assignment, +4.1×):** Nicht anwendbar — Fork hat keinen CPU MoE computation path. Alle Experten laufen auf GPU (mit cache misses → CPU→GPU transfer).
- **Phase 3 (Residual-Prefetch, +9%):** Optional für später. Erfordert offline calibration + residual vectors im GGUF. Orthogonal zu thecodacus backfill.

## Referenzen

- **Paper:** arXiv:2602.03495 (DALI)
- **Code:** `ggml/src/ggml-cuda/moe-cache.cu` (POLICY_WORKLOAD, Default OFF)
- **Commit:** d2d3c8ce6
