# MoE Cache Prediction — FlashMoE + ST-MoE Design

**Date:** 2026-07-15
**Status:** Phase 1 (Research + Design)
**ROADMAP Items:** #69 FlashMoE, #70 ST-MoE
**Target Hardware:** Styx (GTX 1070, 8GB VRAM), Hydra (GTX 1060, 6GB VRAM)

## Overview

Zwei komplementäre Ansätze zur Verbesserung des MoE-Expert-Caches:

| Feature | FlashMoE (arXiv:2601.17063) | ST-MoE (arXiv:2606.15453) |
|---------|---------------------------|--------------------------|
| **Fokus** | Cache-Replacement (WER bleibt?) | Prefetching (WANN laden?) |
| **Algorithmus** | ML-FFN approximiert Belady's optimal | CCT + THT statistische Korrelation |
| **Training** | Offline FFN-Training nötig | Training-free (nur Profiling) |
| **Speedup** | 2.6× vs LRU (SSD-Offloading) | 2.5× vs GPU (Hardware-Design) |
| **Realistisch für Styx** | 10-20% (Cache-bound) | 5-15% (Prefetch-Optimierung) |
| **Referenzcode** | Keiner | Keiner |

## Synergie

```
┌─────────────────────────────────────────────────────┐
│              MoE Cache System                       │
├─────────────────────────────────────────────────────┤
│  FlashMoE: Eviction Policy (was stays in cache?)   │
│  - Ersetzt LRU durch ML-basierte Entscheidung       │
│  - FFN: score = f(recency, frequency)               │
├─────────────────────────────────────────────────────┤
│  ST-MoE: Prefetching (when to load?)               │
│  - Ersetzt sequential backfill sweep                │
│  - CCT: Cross-layer expert correlation              │
│  - THT: Temporal token-to-token prediction          │
└─────────────────────────────────────────────────────┘
```

## Styx Hardware Analysis

### Cache Capacity
- **VRAM:** 8GB - 3GB reserve = 5GB für MoE-Cache
- **Expert size (Gemma-4 26B-A4B, Q4_K_XL):** ~80-90MB pro Expert
- **Cache slots:** ~55-62 Experten von 128 = **43-48% Coverage**
- **Top-K:** 8 von 128 = 6.25% aktiv pro Token

### Bottleneck
- **PCIe 3.0 x8:** ~6 GB/s real
- **CPU→GPU Transfer:** ~200-500 MB/s (pinned memory)
- **Bei 50% Hit-Rate:** ~4 Misses/Token × 85MB = 340MB/Token Transfer
- **Cache ist primärer Bottleneck auf Styx**

## Phased Implementation Plan

### Phase 1: Simplified Heuristic (Option B) — ✅ Implementiert + ❌ Benchmark No-Go

**Status:** Vollständig implementiert in `ggml/src/ggml-cuda/moe-cache.cu`.
- `moe_cache_slot` hat `freq` und `last_access` Felder
- `POLICY_HEURISTIC` mit score = alpha*(1/(age+1)) + beta*(freq/max_freq), alpha=0.7, beta=0.3
- Aktivierung via `GGML_CUDA_MOE_CACHE_POLICY=heuristic`
- Eviction: Slot mit niedrigstem Score wird evicted (nicht LRU-Head)
- max_freq wird bei Eviction des heißesten Slots recomputet
- tick wird einmal pro plan() Call advanced (vermeidet Iterations-Order-Bias)

**Benchmark 2026-07-21 (Styx, GTX 1070, 26B-A4B QAT, -ncmoe 20, turbo3/4, FA, Budget=512MB, Reserve=256MB):**

| Run | Policy | Test | t/s | Hit-Rate | Slots | Anmerkung |
|-----|--------|------|-----|----------|-------|-----------|
| 1 | LRU (Default) | pp512 | 368.26 ± 3.67 | — | — | Cache bei PP deaktiviert (decode-only) |
| 1 | LRU (Default) | tg128 | 24.05 ± 0.94 | — | 0 | Cache **aus** (budget 0 MB auto) |
| 2 | LRU | tg128 | **28.16 ± 0.92** | 39.7% | 320 | Cache an (Budget 512MB) |
| 3 | Heuristic | tg128 | **27.28 ± 0.75** | 39.9% | 320 | Cache an (Budget 512MB) |
| 5 | LRU, throttle=1 | tg128 | 22.26 ± 4.17 | 26.9% | 320 | Bail-out (cache-engaged 1326us vs 539us pure-CPU) |
| 6 | Heuristic, throttle=1 | tg128 | 19.66 ± 3.80 | 13-16% | 320 | Bail-out (cache-engaged 1597us vs 615us pure-CPU) |

**Ergebnis:** Heuristic ist **-3.1% langsamer** als LRU bei gleicher Hit-Rate (39.9% vs 39.7%). 5%-Speedup-Schwelle **deutlich verfehlt**.

**Root Cause Analyse:**
- Hit-Rate nahezu identisch (39.7% vs 39.9%) → Heuristic trifft dieselben Eviction-Entscheidungen wie LRU auf diesem Workload
- Heuristic-Overhead: plan=5.8us (vs LRU 5.2us), +0.6us/Node × 22824 Nodes = +13.7ms/Run → bei ~810s Run-Zeit vernachlässigbar
- Tatsächlicher Speedup-Verlust stammt vermutlich aus leicht veränderten Eviction-Patterns die zu schlechterer Locality führen (Heuristic evicted nach Score, nicht nach reinem Age → kann heiße Slots mit niedrigem freq evicten)
- Throttle ist essenziell: throttle=1 → Cache bail-out für beide Policies (Cache-Overhead > CPU-Path)

**Schlussfolgerung:** Auf Styx mit Gemma-4 26B-A4B (128 Experten, Top-4, QAT) und 512MB Cache-Budget ist LRU bereits nah an optimal für diesen Workload. Die Heuristic mit α=0.7/β=0.3 bringt keinen Mehrwert — die Frequency-Komponente korreliert stark mit Recency auf kurzen tg128-Runs.

### Phase 2: FlashMoE FFN Cache-Replacement — ❌ NO-GO

**Entscheidung 2026-07-21:** Phase 2 wird **nicht durchgeführt**.

**Begründung:**
- Phase 1 Heuristic zeigt **-3.1%** statt der benötigten >+5% → Go-Bedingung nicht erfüllt
- FlashMoE FFN wäre eine komplexere Version derselben Idee (score = f(recency, frequency) mit ML statt Hand-Formula) → wenn schon die einfache Heuristic keinen Win bringt, ist auch die FFN-Version skeptisch zu betrachten
- Paper's 2.6×-Speedup bezieht sich auf **SSD-Offloading** (FlashMoE Paper) — bei CPU-Offload (Styx) ist die Transfer-Latenz niedriger, der Cache-Optimierung-Hebel kleiner
- Aufwand (2-3 Wochen) steht in keinem Verhältnis zum erwarteten Negativ-Ergebnis

**Phase 2 → ❌ verworfen.** ROADMAP #69 Status: 🔬 Phase 1 ✅ → ❌ (Heuristic implementiert aber kein Speedup, Phase 2 No-Go).

### Phase 3: ST-MoE Temporal Prefetching — ⏭️ später

**Status:** Design 2026-07-15 (THT viable, CCT ❌ bei 6.6% Cross-Layer-Overlap). Unabhängig von Phase 2 — ST-MoE ist **Prefetching** (WANN laden?), nicht Eviction (WER bleibt?). Kann auch ohne FlashMoE umgesetzt werden.

**Aufwand:** 2-3 Wochen (THT only, CCT skippen).

**Bedingung:** Wenn #71 (CPU-GPU Collaborative MoE) oder #18 (DALI Cache-Policy) keinen ausreichenden Speedup bringen, THT als nächste Prefetch-Optimierung.

1. **Woche 1:** FFN-Inferenz-Engine (C++, hand-rolled MLP, 2→8→1)
   - Input: (1/r_t_norm, f_t_norm)
   - Output: eviction_score
   - ~158µs Inferenz (vernachlässigbar vs 3ms Transfer)

2. **Woche 2:** Training-Infrastruktur
   - `scripts/train_flashmoe.py`: PyTorch Training-Script
   - Routing-Trace-Extraction aus Fork (512 Samples, 512 Tokens)
   - Belady-Optimal Labels als Training-Target
   - FFN-Weights als JSON/CSV serialisieren

3. **Woche 3:** Integration + Benchmarking
   - `--moe-cache-policy flashmoe` mit FFN-Weights-File
   - Benchmark auf Styx vs LRU vs Heuristic
   - Generalisierungstest: Andere Modelle/Workloads

### Phase 3: ST-MoE Temporal Prefetching — 2-3 Wochen

**THT (Temporal History Table) zuerst — höchster ROI:**

1. **THT implementieren:**
   - Sliding window über letzte N Token's expert activations
   - Predict nächste Token's Experten basierend auf History
   - Ersetzt `moe_cache_backfill_next()` sequential sweep

2. **CCT (Cross-layer Correlation Table) — optional:**
   - Profiling-Phase: Expert co-activation zwischen Layern aufzeichnen
   - Wenn Layer L Experten {3, 17, 42} aktiviert → prefetch Layer L+1's korrelierte
   - **Achtung:** Fork hat bereits Cross-Layer-Overlap gemessen = 6.6% (nahe Random)
   - CCT wahrscheinlich **nicht viable** für Gemma-4 26B-A4B (128 Experten, Top-8)

3. **Integration:**
   - THT-Prediction → Prefetch-Queue
   - Bestehende Worker-Threads nutzen für async Prefetch
   - `--moe-prefetch-policy backfill|tht|cct|tht+cct`

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| FFN generalisiert nicht auf Gemma-4 | Medium | Phase 1 Heuristik zuerst |
| CCT nicht viable (6.6% Overlap) | High | THT zuerst, CCT optional |
| CPU-bound statt Cache-bound auf Styx | High | Benchmark vor Implementierung |
| Kein Referenzcode | Medium | Paper-Beschreibung ausreichend für THT |
| Training-Daten-Qualität | Medium | Routing-Trace aus echter Inference |

## Decision Matrix

| Option | Aufwand | Erwarteter Speedup | Risiko | Empfehlung |
|--------|---------|-------------------|--------|------------|
| **Phase 1: Heuristic** | 2-3 Tage | 5-10% | Niedrig | ✅ **Jetzt starten** |
| Phase 2: FlashMoE FFN | 2-3 Wochen | 10-20% | Medium | Nach Phase 1 Benchmark |
| Phase 3: THT Prefetch | 1-2 Wochen | 5-15% | Niedrig | Nach Phase 2 |
| Phase 3: CCT Prefetch | 1-2 Wochen | <5% | High | ❌ Skip (6.6% Overlap) |

## Next Steps

1. **Sofort:** Phase 1 Heuristic implementieren (2-3 Tage)
2. **Benchmark:** Styx mit `--moe-cache-policy heuristic` vs `lru`
3. **Entscheidung:** Wenn >5% Speedup → Phase 2 starten
4. **Parallel:** THT-Prefetch unabhängig von FlashMoE entwickelbar

## References

- FlashMoE: arXiv:2601.17063 — ML-based Cache Replacement for MoE
- ST-MoE: arXiv:2606.15453 — Spatio-Temporal Prefetching for MoE
- Belady's Algorithm: Optimal Cache Replacement (mit Zukunftswissen)
- Bestehender Code: `ggml/src/ggml-cuda/moe-cache.cu` (LRU + backfill)
- Cross-Layer-Overlap Messung: `docs/fork/2026-07-15_CROSS_LAYER_GATE_EVAL.md` (6.6%)
