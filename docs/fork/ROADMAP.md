# ROADMAP — Fork-Beschleunigung

**Erstellt:** 2026-07-11
**Quelle:** [Optimierungs-Recherche 2026-07-11](2026-07-11_OPTIMIZATION_RESEARCH.md) (Web + arXiv, 4 parallele Subagents)
**Hardware:** Mars (RDNA3/Vulkan/30GB), Styx (Pascal/CUDA/8GB), Hydra (Ampere/CUDA/8GB), Uranus (2x Ada/CUDA/32GB), Venus (GCN/Vulkan)
**Ausschluss:** Treiber-Neuimplementierung, Kernel-Rekompilierung, Hopper-spezifische Features

---

## Meilensteine

| MS | Name | Items | Status |
|----|------|-------|--------|
| **M1** | Quick-Win-Welle | #1-5 (kombiniert) | ✅ abgeschlossen (#2✅, #4✅, #5✅, #1❌) |
| **M2** | Vulkan-Offensive | #6, #7, #9, #10, #12 | ❌ blockiert (#6✅ bereits integriert, #7⏭️, #9❌, #10❌) |
| **M3** | MoE-Offloading v2 | #3, #13, #14, #15 | ⏳ teilweise (#3✅, #13❌) |
| **M4** | Speculative Decoding v2 | #11, #28 | ☐ offen |
| **M5** | Coopmat2 + Multi-GPU | #12, #20, #21 | ☐ offen |
| **M6** | Forschung | #17, #18, #19, #25, #26, #27, #29, #30 | ☐ offen |

**Regel:** Solo-Pläne werden nur für den **aktuellen und nächsten Meilenstein** erstellt. Tier 3-4 Pläne entstehen wenn der Meilenstein näher rückt (verhindert veraltete Pläne).

---

## Tier 1: Quick Wins (< 1 Woche)

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 1 | ❌ | MTP Logits Copy Optimization | Mars, Styx | 2-3h | PR #23198 | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | +20% PP mit MTP — **19 Konflikte in 10 Kern-Dateien, MTP OFF, skipped** |
| 2 | ✅ | Tensor Split Regex Optimization | Uranus | 1-2h | PR #24710 (open) | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | befreit 40% decode thread — **angewendet, Build grün auf Hydra+Mars** |
| 3 | ✅ | Pascal CUDA MMVQ Optimization | Styx | 2-3 Tage | PR #25479 (draft) | [M3](plans/SESSION_PLAN_pascal-mmvq.md) | +3-6% decode auf Pascal — **manuell portiert, +8.9% Generation auf GTX 1070 (E2B MoE), Code-Review ship-ready** |
| 4 | ✅ | n-gram / Prompt-Lookup Decoding | Alle | 0h | in mainline | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | hilft bei repetitiven Workloads — **verfügbar (`--spec-type ngram-mod`), kein Speedup auf E2B (39.1 vs 39.2 t/s)** |
| 5 | ✅ | GTT Size Tuning für Mars APU | Mars | 1h | Konfiguration | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | >15GB GPU-Speicher — **bereits konfiguriert (26GB GTT), kein Tuning nötig** |
| 6 | ✅ | Vulkan MUL_MAT_ID Subgroup Optimization | Mars, Venus | 1 Tag | PR #15524 (merged) | [M2](plans/SESSION_PLAN_mulmat-id-subgroup.md) | bis +657% MoE PP auf AMD — **bereits im Fork integriert: MatMulIdType::SUBGROUP, mul_mm_id_funcs.glsl, matmul_id_subgroup_* Shader werden generiert, Pipelines aktiv (subgroup_ballot && subgroup_require_full_support && subgroup_min_size_16)** |
| 7 | ⏭️ | Vulkan FlashAttention Refactor | Mars, Venus | 1 Woche | PR #19625 | [M2](plans/SESSION_PLAN_vulkan-fa-refactor.md) | 10-20% scalar FA improvement — **verschoben, abhängig von #6** |

## Tier 2: Mittelfristig (1-3 Wochen)

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 8 | ☐ | Mixed Precision KV Cache (Hot/Cold) | Alle | 1 Woche | commit e889fbd | später | Reduziert KV-Größe bei erhaltener Qualität |
| 9 | ❌ | Vulkan Shared Memory Staging Kernel | Mars, Venus | 2-3 Tage | PR #20897 (closed) | [M2](plans/SESSION_PLAN_vulkan-shmem-staging.md) | >2.5x TG potenziell — **PR closed ohne Merge (AI-generiert), manuelle Portierung nötig** |
| 10 | ❌ | UMA Zero-Copy für Mars APU | Mars, Venus | 1-2 Wochen | PR #22462 | [M2](plans/SESSION_PLAN_uma-zero-copy.md) | +112x transfer speed — **nicht implementieren: verwandte UMA-PRs (#22455, #22930, #23770) bereits getestet und revertiert, RCA-Masterplan zeigt System-RAM ist langsamer als GTT für GPU-Compute auf Mars, PR #22462 ist open/unstable** |
| 11 | ☐ | EAGLE-3 Speculative Decoding | Alle | 2-3 Wochen | PR #18039 | [M4](plans/SESSION_PLAN_eagle3.md) | bis 6.5x speedup |
| 12 | ☐ | Vulkan Cooperative Matrix (Coopmat2) | Mars, Uranus | 2-3 Wochen | PR #19075 | [M5](plans/SESSION_PLAN_coopmat2-rdna3.md) | 2.5x FA multiply |
| 13 | ❌ | Two-Tier GPU+RAM Expert Cache | Styx, Mars | 1-2 Wochen | FR #20757 | [M3](plans/SESSION_PLAN_two-tier-expert-cache.md) | kritisch für Styx 8GB — **nicht implementieren: alle PRs closed, thecodacus Memory Pinning deckt Tier 2 bereits ab, geringer ROI für Pascal (compute-bound nicht PCIe-bound)** |
| 14 | ☐ | LFU Caching Policy für MoE Experts | Styx, Mars | 2-3 Tage | nein (Forschung) | [M3](plans/SESSION_PLAN_lfu-caching.md) | 15-20% expert hit rate |
| 15 | ☐ | PipeShard: VRAM-Constrained MoE | Styx, Hydra, Mars | 2-3 Wochen | arXiv:2604.26334 | [M3](plans/SESSION_PLAN_pipeshard.md) | VRAM-optimierte MoE-Inference |
| 16 | ☐ | Vulkanised 2026: shmem-staging M-V Kernel | Mars, Venus | 1-2 Wochen | upstream | später | >2.5x TG auf Intel Arc |

## Tier 3: Komplex (3-6 Wochen)

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 17 | ☐ | HOBBIT: Mixed-Precision Expert Offloading | Styx, Mars | 4-6 Wochen | arXiv:2411.01433 | später | bis 9.93x MoE offloading |
| 18 | ☐ | DALI: Workload-Aware MoE Offloading | Styx | 2-3 Wochen | arXiv:2602.03495 | später | intelligentes MoE-Offloading |
| 19 | ☐ | Expected Attention KV Cache Compression | Alle | 2-3 Wochen | arXiv:2510.00636 | später | ergänzt TurboQuant |
| 20 | ☐ | Tensor Parallelism für Uranus | Uranus | 3-4 Wochen | PR #19378 | [M5](plans/SESSION_PLAN_tensor-parallelism.md) | ~40% boost für 2x 4060 Ti |
| 21 | ☐ | PagedAttention / Paged KV Cache | Uranus, Mars | 3-4 Wochen | PR #22569 | später | 2.5x aggregate throughput |
| 22 | ☐ | GWQ: Gradient-Aware Weight Quantization | Alle | 2-3 Wochen | arXiv:2411.00850 | später | 1.2x inference speedup |
| 23 | ☐ | DuoServe-MoE: Dual-Phase Expert Scheduling | Styx | 3-4 Wochen | arXiv:2509.07379 | später | phase-spezifisches Prefetch |
| 24 | ☐ | HybriMoE: Hybrid CPU-GPU Scheduling | Styx | 3-4 Wochen | kTransformers | später | 1.33x prefill, 1.70x decode |

## Tier 4: Langfristig / Forschung (6+ Wochen)

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 25 | ☐ | llama.cpp 2026 Rewrite Merge | Alle | 4-6 Wochen | mainline | später | 2.1x Durchsatz 70B, 1.4x 7B |
| 26 | ☐ | Q-Filters / LagKV / MiniCache | Alle | 2-3 Wo/Methode | GitHub (qfilters) | später | 5-32x KV compression |
| 27 | ☐ | Pre-Attention Expert Prediction | Styx, Mars | 2-3 Wochen | Forschung | später | 15% accuracy improvement |
| 28 | ☐ | Adaptive MTP Speculative Decoding | Alle | 1-2 Wochen | PR #22931 | [M4](plans/SESSION_PLAN_adaptive-mtp.md) | verhindert MTP-Regression |
| 29 | ☐ | FastKV: Token-Selective Propagation | Alle | 3-4 Wochen | GitHub (fastkv) | später | 1.82x prefill, 2.87x decode |
| 30 | ☐ | Speculative Expert Prefetching (MoE-SpeQ) | Styx, Mars | 3-4 Wochen | Forschung | später | 2.34x offloading speedup |

---

## Nicht empfohlen / nicht anwendbar

| Ansatz | Grund |
|--------|-------|
| FlashAttention-3 (arXiv:2407.08608) | Hopper-spezifisch (H100), unsere GPUs: Pascal/Ampere/Ada/RDNA3 |
| Programmatic Dependent Launch (PDL) | Hopper+ Feature, nicht auf unserer Hardware |
| GANQ (arXiv:2501.12956) | Erfordert LUT-basierte GEMM Kernels, sehr GPU-spezifisch |
| valkyr (Zig+Vulkan) | Kompletter Stack-Wechsel zu Zig, 8-12 Wochen |
| AnchorTP / Shift Parallelism | Sehr komplex (10-14 Wochen), nur für Uranus |
| Tiled Flash Linear Attention | Für zukünftige Modelle (xLSTM), nicht für aktuelle Transformer |
| NUMA-aware Optimization | Arm Neoverse spezifisch, unsere x86 CPUs haben kein relevantes NUMA |

---

## Abhängigkeiten

| Item | Hängt ab von | Grund |
|------|-------------|-------|
| #12 Coopmat2 | #7 Vulkan FA Refactor | Coopmat-FA baut auf Refactor auf |
| #20 Tensor Parallelism | #2 Tensor Split Regex | Regex-Opt ist Voraussetzung für stabiles TP |
| #11 EAGLE-3 | — | Unabhängig, aber Draft-Modell muss existieren |
| #17 HOBBIT | #14 LFU Caching | HOBBIT nutzt erweiterte Caching-Policies |
| #25 2026 Rewrite | Alle M1-M5 | Merge nach Abschluss der geplanten Features |

---

## Priorisierung nach Cost-Benefit

### Sofort umsetzbar (M1 — Quick-Win-Welle):
1. **#6 MUL_MAT_ID Subgroup** — 1 Tag, bis +657% MoE PP auf AMD
2. **#1 MTP Logits Copy** — 2-3h, +20% PP mit MTP
3. **#2 Tensor Split Regex** — 1-2h, befreit 40% decode thread
4. **#5 GTT Size Tuning** — 1h, mehr GPU-Speicher für Mars
5. **#4 n-gram Decoding** — 0h, nur aktivieren

### Diese / nächste Woche (M2 — Vulkan-Offensive):
6. **#7 Vulkan FA Refactor** — 1 Woche, 10-20% scalar FA improvement
7. **#9 Vulkan Shmem-Staging** — 2-3 Tage, >2.5x TG potenziell
8. **#10 UMA Zero-Copy** — 1-2 Wochen, +112x transfer speed für Mars

### Mittelfristig (M3 — MoE-Offloading v2):
9. **#3 Pascal MMVQ** — 2-3 Tage, +3-6% decode auf Styx
10. **#14 LFU Caching** — 2-3 Tage, 15-20% expert hit rate
11. **#13 Two-Tier Expert Cache** — 1-2 Wochen, kritisch für Styx 8GB
12. **#15 PipeShard** — 2-3 Wochen, VRAM-constrained MoE

### Mittelfristig (M4 — Speculative Decoding v2):
13. **#28 Adaptive MTP** — 1-2 Wochen, verhindert MTP-Regression
14. **#11 EAGLE-3** — 2-3 Wochen, bis 6.5x speedup

### Langfristig (M5-M6):
15. **#12 Coopmat2 RDNA3** — 2-3 Wochen, 2.5x FA multiply
16. **#20 Tensor Parallelism** — 3-4 Wochen, 40% boost für Uranus
17. **#17 HOBBIT** — 4-6 Wochen, 9.93x MoE offloading
18. **#25 2026 Rewrite** — 4-6 Wochen, system-wide 1.4-2.1x
