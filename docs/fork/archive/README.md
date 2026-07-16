# docs/fork/archive — Historische Dokumente

Dieses Verzeichnis enthält abgeschlossene, gelöste oder obsolete Dokumente die nicht mehr aktiv benötigt werden, aber als historische Referenz behalten werden.

## Struktur

| Unterverzeichnis | Inhalt |
|------------------|--------|
| `rca/` | Root Cause Analysen und Bug-Fixes (Vulkan TurboQuant, GPU-Hang, Perf-RCA) |
| `benchmarks/` | Historische Benchmark-Ergebnisse (Pascal, Vulkan KV-Cache, QAT MTP, MTP Draft) |
| `sessions/` | Solo-Session-Pläne, -Reports, Code-Reviews, Wochenrückblicke |
| `research/` | Wöchentliche Optimierungs-Recherche-Reports (2026-07-11 bis 2026-07-14) |
| `plans/` | Abgeschlossene/blockierte SESSION_PLANs (✅/❌/⏭️) |

## Index

### RCA (gelöste Bugs)
- `rca/2026-06-20_VULKAN_TURBOQUANT_KV_CACHE_FIX.md` — ✅ turbo3/4 KV-Cache Fix
- `rca/2026-06-21_VULKAN_GPU_HANG_ROOT_CAUSE.md` — ✅ amdgpu.lockup_timeout
- `rca/2026-06-21_VULKAN_PERF_RCA_BASELINE.md` — ✅ Perf-Klippe Baseline
- `rca/2026-06-21_VULKAN_PERF_RCA_MASTERPLAN.md` — ✅ Perf-Klippe Masterplan
- `rca/2026-06-21_VULKAN_COMMIT_KATEGORISIERUNG.md` — 61 Vulkan-Commits kategorisiert

### Benchmarks (historisch)
- `benchmarks/2026-07-08_BASELINE_PASCAL.md` — thecodacus MoE Baseline
- `benchmarks/2026-07-12_E4B_QAT_MTP_BENCHMARK.md` — E4B QAT MTP auf Uranus
- `benchmarks/2026-07-14_MTP_DRAFT_COMPARISON.md` — MTP Draft-Vergleich Styx vs Mars

### Sessions (historisch)
- `sessions/2026-06-21_SYNC_MERGE_MASTERPLAN.md` — AtomicBot Sync-Merge
- `sessions/2026-06-23_SOLO_SCHICHT_PLAN.md` — Solo-Schicht Plan
- `sessions/2026-07-08_CODE_REVIEW_THECODACUS.md` — thecodacus Code-Review
- `sessions/2026-07-08_FINALE_EMPFEHLUNG.md` — Finale Modell-Empfehlung
- `sessions/2026-07-08_MTP_KV_CACHE_CONTEXT_ANALYSIS.md` — MTP KV-Cache Analyse
- `sessions/2026-07-08_SOLO_SESSION_REPORT.md` — thecodacus Portierung Report
- `sessions/2026-07-14_WEEKLY_REVIEW.md` — Wochenrückblick 7.-14. Juli

### Research (wöchentlich, zusammengefasst in research/OPTIMIZATION_RESEARCH_INDEX.md)
- `research/2026-07-11_OPTIMIZATION_RESEARCH.md` — Quick Wins + Tier 2
- `research/2026-07-12_OPTIMIZATION_RESEARCH.md` — CM1, Pascal L1, UBBoost, LFRU
- `research/2026-07-13_OPTIMIZATION_RESEARCH.md` — Persistent VRAM, REAP, xKV, BSFA
- `research/2026-07-14_OPTIMIZATION_RESEARCH.md` — MMVQ Fix, Pipeline Cache, NCCL, GEAR

### Plans (abgeschlossen/blockiert)
- `plans/SESSION_PLAN_*.md` — 14 abgeschlossene oder blockierte Session-Pläne

## Nicht archiviert (bleibt in docs/fork/)

- `2026-06-20_TURBO4_SERVER_SEGV_BUG.md` — 🔴 OFFEN, blockiert turbo4 im Server-Modus
- `2026-07-15_*.md` — Aktive Design-Docs und Evals (EA, MoE Cache, NCCL, Cross-Layer Gate)
- `2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md` — Referenz für turbo3/4 Empfehlung
- `2026-07-10_QAT_MTP_Q4_0_BENCHMARK.md` — Referenz für QAT-Standard
- `2026-07-11_E4B_MTP_8GB_CRASH.md` — Referenz für 8GB Crash-Kombination
- `plans/SESSION_PLAN_20260714_persistent-vram-expert-cache.md` — ⏭️ Implementierung ausstehend
- `plans/SESSION_PLAN_20260714_pre-attention-expert-prediction.md` — ⏭️ verschoben
