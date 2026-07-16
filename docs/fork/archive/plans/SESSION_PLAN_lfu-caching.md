# SESSION_PLAN: LFU/LFRU Caching für MoE Experts (M3)

**Erstellt:** 2026-07-11
**Typ:** Forschung + Implementierung
**Status:** ⏭️ verschoben
**Meilenstein:** M3 — MoE-Offloading v2
**ROADMAP-Item:** #14 — LFU Caching Policy für MoE Experts

---

## Session-Ziel

Expert-Caching mit LFRU (Least Frequently Recently Used) Policy implementieren,
um MoE-Expert-Hit-Rate zu verbessern. Synergie mit thecodacus Memory Pinning.

## Recherche-Ergebnisse (2026-07-11)

### Paper-Evidenz
- arXiv:2511.05814: LFU zeigt "strong improvements" über LRU
- arXiv:2602.03921 (SpecMD): "Least-Stale" Policy 85× besser als LRU
- arXiv:2601.17063 (FlashMoE): ML-basiert +51% hit rate über LRU/LFU
- arXiv:2502.05370 (FineMoE): LFU +39% hit rate, 47% Latenzreduktion

### Referenz-Implementierungen
- vLLM PR #37190: LFRU eviction, production-ready, +5.2% speedup
- llama.cpp PR #24524: closed (zu groß für review)
- kTransformers HybriMoE: Score-based caching

### Empfehlung
- **Nicht LFU allein** — LFRU (freq/age hybrid) ist besser
- **Memory Pinning ist Voraussetzung** für async H2D transfers
- **SpecMD "Least-Stale"** ist 85× besser als LRU, aber komplex (predictive model)
- **Aufwand:** 3-4 Wochen für LFRU (1000-1500 LOC)

### ROI
- Styx (8GB, -ncmoe 20): 15-20% latency reduction — **lohnt sich**
- Mars (30GB, -ngl 99): <5% gain — **nicht prioritär**

## Schritte (bei Implementierung)

1. Expert access tracing implementieren (per-layer frequency counter)
2. LFRU eviction policy implementieren (score = freq / age)
3. GPU slot buffer management (fixed-address slots)
4. Integration mit --cpu-moe / --n-cpu-moe
5. Async H2D copy on cache miss (mit pinned memory)
6. Benchmark auf Styx (MoE-Modell mit -ncmoe 20)

## Abhängigkeiten

- thecodacus Memory Pinning (bereits im Fork) — Voraussetzung für async transfers
- #13 Two-Tier Expert Cache (abgelehnt) — LFRU ist die Alternative
